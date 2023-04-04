from dataclasses import dataclass
from typing import Tuple

import torch
from scipy.spatial.transform import Rotation as R

from src.latent_nerf.configs.render_config import RenderConfig
from src.latent_nerf.configs.scene_config import SceneNerfsConfig, SceneProxiesConfig
from src.latent_nerf.models.network_grid import NeRFNetwork
from .render_utils import safe_normalize, near_far_from_bound


@dataclass
class ObjectNerf:
    id: str
    model: NeRFNetwork
    text: str
    shape_path: str
    proximal_surface: float
    is_bg_nerf: bool
    shape_scale: float


@dataclass
class ObjectProxy:
    nerf: ObjectNerf
    center: Tuple[float, float, float]
    rotation_matrix: torch.Tensor
    proxy_scale: float = 1.0


class SceneNerf(NeRFNetwork):

    def __init__(self, render_cfg: RenderConfig, scene_nerfs_cfg: SceneNerfsConfig,
                 scene_proxies_cfg: SceneProxiesConfig, device):
        super().__init__(render_cfg)
        self.to(device)
        self.device = device
        self.nerfs_dict = self.init_nerfs(render_cfg, scene_nerfs_cfg)
        self.proxy_list = self.init_proxies(scene_proxies_cfg)

        self.encoder = None
        self.sigma_net = None
        self.normal_net = None

    def init_nerfs(self, render_cfg: RenderConfig, scene_nerfs_cfg: SceneNerfsConfig):
        nerfs_dict = {}
        num_nerfs = len(scene_nerfs_cfg.nerf_ids)
        for i in range(num_nerfs):
            id = scene_nerfs_cfg.nerf_ids[i]
            model = NeRFNetwork(render_cfg).to(self.device)
            object_nerf = ObjectNerf(id=id, model=model, text=scene_nerfs_cfg.nerf_texts[i],
                                     shape_path=scene_nerfs_cfg.shape_paths[i],
                                     proximal_surface=scene_nerfs_cfg.shape_proximal_surfaces[i],
                                     is_bg_nerf=scene_nerfs_cfg.is_bg_nerfs[i],
                                     shape_scale=scene_nerfs_cfg.shape_scales[i])
            nerfs_dict[id] = object_nerf
        return nerfs_dict

    def init_proxies(self, scene_proxies_cfg: SceneProxiesConfig):
        proxy_list = []
        num_proxies = len(scene_proxies_cfg.proxy_to_nerf_ids)
        for i in range(num_proxies):
            nerf_id = scene_proxies_cfg.proxy_to_nerf_ids[i]
            nerf = self.nerfs_dict[nerf_id]
            center = scene_proxies_cfg.proxy_centers[i]
            center = -1 * torch.tensor(center, device=self.device)
            proxy_scale = scene_proxies_cfg.proxy_scales[i]
            rotation_clockwise = scene_proxies_cfg.proxy_rotations_clockwise[i]
            rotation_mat = R.from_euler('y', rotation_clockwise, degrees=True).as_matrix() / proxy_scale
            rotation_mat = torch.from_numpy(rotation_mat).to(torch.float32).to(self.device)

            proxy = ObjectProxy(nerf=nerf, center=center, rotation_matrix=rotation_mat, proxy_scale=proxy_scale)
            proxy_list.append(proxy)

        return proxy_list

    def train(self, mode: bool = True):
        for nerf in self.nerfs_dict.values():
            nerf.model.train(mode)
        super().train(mode)

    def get_params(self, lr):
        params = []
        for nerf in self.nerfs_dict.values():
            cur_params_list = nerf.model.get_params(lr=lr)
            params += cur_params_list
        return params

    def run(self, rays_o, rays_d, num_steps, upsample_steps, light_d=None, ambient_ratio=1.0, shading='albedo',
            bg_color=None, perturb=False, T_thresh=1e-4, disable_background=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]
        num_proxies = len(self.proxy_list)
        num_steps = num_steps * num_proxies

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = near_far_from_bound(rays_o, rays_d, self.bound, type='cube', min_near=self.min_near)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        initial_z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        initial_z_vals = initial_z_vals.expand((N, num_steps))  # [N, T]
        initial_z_vals = nears + (fars - nears) * initial_z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            initial_z_vals = initial_z_vals + (torch.rand(initial_z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        initial_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * initial_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        initial_xyzs = torch.min(torch.max(initial_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        z_vals_list = []
        rgbs_list = []
        xyzs_list = []
        sigmas_list = []
        normals_list = []
        for i, proxy in enumerate(self.proxy_list):
            nerf = proxy.nerf
            cur_sigmas, cur_rgbs, cur_z_vals, cur_xyzs, cur_normals = nerf.model.run_single_nerf(rays_o, rays_d,
                                                                                                 initial_xyzs[:,
                                                                                                 i::num_proxies].clone(),
                                                                                                 initial_z_vals[:,
                                                                                                 i::num_proxies].clone(),
                                                                                                 sample_dist,
                                                                                                 num_steps // num_proxies,
                                                                                                 upsample_steps,
                                                                                                 light_d, ambient_ratio,
                                                                                                 shading, aabb,
                                                                                                 rotation_matrix=proxy.rotation_matrix,
                                                                                                 coord_shift=proxy.center)
            z_vals_list.append(cur_z_vals)
            rgbs_list.append(cur_rgbs)
            xyzs_list.append(cur_xyzs)
            sigmas_list.append(cur_sigmas.reshape(N, num_steps // num_proxies + upsample_steps))
            normals_list.append(cur_normals.reshape(N, num_steps // num_proxies + upsample_steps, 3))

        # concat values from all single nerfs
        unsorted_z_vals = torch.cat(z_vals_list, dim=1)
        z_vals, indices = unsorted_z_vals.sort(dim=1)
        rgbs = torch.cat(rgbs_list, dim=1)[torch.arange(N).unsqueeze(1), indices]
        for i, proxy in enumerate(self.proxy_list):
            # todo: fix this!
            nerf = proxy.nerf
            sigmas_list[i] = sigmas_list[i] * 2
            if nerf.shape_path is None:
                sigmas_list[i] = torch.exp(sigmas_list[i] / torch.max(sigmas_list[i])) * sigmas_list[i] * 2
        sigmas = torch.cat(sigmas_list, dim=1)[torch.arange(N).unsqueeze(1), indices]

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  #
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * sigmas.view(N, -1))  #
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]
        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)

        # mix background color
        if self.bg_radius > 0 and not disable_background:
            # use the bg model to calculate bg_color
            # sph = self.raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius) # [N, 2] in [-1, 1]
            bg_color = self.background(rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        depth = depth + (1 - weights_sum)
        depth = 1 - depth

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = [xyzs.reshape(-1, 3) for xyzs in xyzs_list]
        results['sigmas'] = [sigmas.flatten() for sigmas in sigmas_list]
        results['alphas'] = alphas

        return results
