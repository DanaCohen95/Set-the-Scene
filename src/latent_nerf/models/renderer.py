import math
from abc import abstractmethod

import torch
import torch.nn as nn

from src.latent_nerf.configs.render_config import RenderConfig
from .render_utils import sample_pdf, safe_normalize, near_far_from_bound


class NeRFRenderer(nn.Module):
    def __init__(self, cfg: RenderConfig, latent_mode: bool = True):
        super().__init__()

        self.opt = cfg
        self.bound = cfg.bound
        self.cascade = 1 + math.ceil(math.log2(cfg.bound))
        self.grid_size = 128
        self.cuda_ray = cfg.cuda_ray
        self.min_near = cfg.min_near
        self.density_thresh = cfg.density_thresh
        self.bg_radius = cfg.bg_radius
        self.latent_mode = latent_mode
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-cfg.bound, -cfg.bound, -cfg.bound, cfg.bound, cfg.bound, cfg.bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)

    @property
    def img_dims(self):
        return 3 + 1 if self.latent_mode else 3

    @abstractmethod
    def forward(self, x, d, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def density(self, x, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def background(self, d):
        raise NotImplementedError()

    @abstractmethod
    def get_params(self, d):
        raise NotImplementedError()

    def run(self, rays_o, rays_d, num_steps, upsample_steps, light_d=None, ambient_ratio=1.0, shading='albedo',
            bg_color=None, perturb=False, T_thresh=1e-4, disable_background=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

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

        sigmas, rgbs, z_vals, xyzs, normals = self.run_single_nerf(rays_o, rays_d, initial_xyzs, initial_z_vals,
                                                                   sample_dist,
                                                                   num_steps, upsample_steps, light_d, ambient_ratio,
                                                                   shading, aabb)

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * sigmas.view(N, -1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]
        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)

        if normals is not None:
            # orientation loss
            normals = normals.view(N, -1, 3)
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.sum(-1).mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth 
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0 and not disable_background:
            # use the bg model to calculate bg_color
            bg_color = self.background(rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, self.img_dims)
        depth = depth.view(*prefix)
        depth = depth + (1 - weights_sum)
        depth = 1 - depth
        # object_mask = depth > 1e-1
        # min_val = 0.5
        # depth[object_mask] = ((1 - min_val) * (depth[object_mask] - depth[object_mask].min()) / (
        #         depth[object_mask].max() - depth[object_mask].min())) + min_val
        # plot histogram of depth
        # import matplotlib.pyplot as plt
        # plt.hist(depth.cpu().numpy().flatten(), bins=100)
        # plt.show()

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        results['weights_sum'] = weights_sum
        results['mask'] = mask
        results['xyzs'] = xyzs.reshape(-1, 3)
        results['sigmas'] = sigmas
        results['alphas'] = alphas

        return results

    def run_single_nerf(self, rays_o, rays_d, xyzs, z_vals, sample_dist, num_steps, upsample_steps, light_d,
                        ambient_ratio, shading, aabb, rotation_matrix=None, coord_shift=None):
        N = rays_o.shape[0]  # N = B * N, in fact
        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3), rotation_tran=rotation_matrix, coord_shift=coord_shift)

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)  # add last delta

                tmp_alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))  # [N, T]
                tmp_alphas_shifted = torch.cat([torch.ones_like(tmp_alphas[..., :1]), 1 - tmp_alphas + 1e-15],
                                               dim=-1)  # [N, T+1]
                tmp_weights = tmp_alphas * torch.cumprod(tmp_alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, tmp_weights[:, 1:-1], upsample_steps,
                                        det=not self.training).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3), rotation_tran=rotation_matrix,
                                               coord_shift=coord_shift)
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), light_d, ratio=ambient_ratio,
                                     shading=shading, rotation_tran=rotation_matrix, coord_shift=coord_shift)
        rgbs = rgbs.view(N, -1, self.img_dims)  # [N, T+t, 3]

        return sigmas, rgbs, z_vals, xyzs, normals

    def render(self, rays_o, rays_d, staged=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]
        kwargs['num_steps'] = self.opt.num_steps
        kwargs['upsample_steps'] = self.opt.upsample_steps

        if self.cuda_ray:
            raise ("cuda ray is not suppurted!")
        else:
            _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, self.img_dims), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + self.opt.max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    weights_sum[b:b + 1, head:tail] = results_['weights_sum']
                    image[b:b + 1, head:tail] = results_['image']
                    head += + self.opt.max_ray_batch

            results = {}
            results['depth'] = depth
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results
