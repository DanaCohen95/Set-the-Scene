import torch
from torch import nn
import torch.nn.functional as F

from src.latent_nerf.configs.render_config import RenderConfig
from .encoding import get_encoder
from .nerf_utils import trunc_exp, MLP, NeRFType, init_decoder_layer
from .render_utils import safe_normalize
from .renderer import NeRFRenderer


def transform_to_scene_coor(x: torch.tensor, rotation_mat: torch.tensor, coord_shift: torch.tensor):
    return torch.matmul(x + coord_shift, rotation_mat).to(torch.float32)


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 cfg: RenderConfig,
                 num_layers=2,
                 hidden_dim=32,
                 num_layers_bg=2,
                 hidden_dim_bg=16,
                 encoder_num_levels=16,
                 ):

        super().__init__(cfg, latent_mode=cfg.nerf_type == NeRFType.latent)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        additional_dim_size = 1 if (cfg.nerf_type != NeRFType.rgb) else 0

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19,
                                                desired_resolution=2048 * self.bound, interpolation='smoothstep')
        self.sigma_net = MLP(self.in_dim, 4 + additional_dim_size, hidden_dim, num_layers, bias=True)
        self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg

            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3)

            self.bg_net = MLP(self.in_dim_bg, 3 + additional_dim_size, hidden_dim_bg, num_layers_bg, bias=True)

        else:
            self.bg_net = None

        if cfg.nerf_type == NeRFType.latent_tune:
            self.decoder_layer = nn.Linear(4, 3, bias=False)
            init_decoder_layer(self.decoder_layer)
        else:
            self.decoder_layer = None

    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        enc = self.encoder(x, bound=self.bound)

        h = self.sigma_net(enc)

        sigma = F.softplus(h[..., 0] + self.gaussian(x))
        albedo = h[..., 1:]
        if self.decoder_layer is not None:
            albedo = self.decoder_layer(albedo)
            albedo = (albedo + 1) / 2
        elif not self.latent_mode:
            albedo = torch.sigmoid(h[..., 1:])

        sigma = torch.where((x >= 1).any(dim=1) | (x <= -1).any(dim=1), torch.zeros_like(sigma), sigma)
        # albedo = torch.where((x >= 1).any(dim=1) | (x <= -1).any(dim=1),torch.zeros_like(albedo), albedo)
        albedo = torch.nan_to_num(albedo, nan=0.0)
        sigma = torch.nan_to_num(sigma, nan=0.0)
        normal = self.normal_net(enc)
        # normal = self.finite_difference_normal(x)
        return sigma, albedo, normal

    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward(
            (x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward(
            (x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward(
            (x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))

        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon,
            0.5 * (dy_pos - dy_neg) / epsilon,
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return normal

    def forward(self, x, d, l=None, ratio=1, shading='albedo', rotation_tran=None, coord_shift=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)
        if rotation_tran is not None or coord_shift is not None:
            x = transform_to_scene_coor(x, rotation_tran, coord_shift)

        sigma, albedo, normal = self.common_forward(x)
        if shading == 'albedo':
            # no need to query normal
            color = albedo
        else:
            lambertian = ratio + (1 - ratio) * (normal @ -l).clamp(min=0)  # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
                if self.latent_mode:
                    # pad color with a single dimension of zeros
                    color = torch.cat([color, torch.zeros((color.shape[0], 1), device=color.device)], axis=1)
            else:  # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)

        return sigma, color, normal

    def density(self, x, rotation_tran=None, coord_shift=None):
        # x: [N, 3], in [-bound, bound]
        if rotation_tran is not None or coord_shift is not None:
            x = transform_to_scene_coor(x, rotation_tran, coord_shift)

        sigma, albedo, _ = self.common_forward(x)

        return {
            'sigma': sigma,
            'albedo': albedo,
        }

    def background(self, d):

        h = self.encoder_bg(d)  # [N, C]

        rgbs = self.bg_net(h)

        if self.decoder_layer is not None:
            rgbs = self.decoder_layer(rgbs)
            rgbs = (rgbs + 1) / 2
        elif not self.latent_mode:
            rgbs = torch.sigmoid(rgbs)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.normal_net.parameters(), 'lr': lr},
        ]
        if self.decoder_layer is not None:
            params.append({'params': self.decoder_layer.parameters(), 'lr': lr})

        # if self.bg_radius > 0:
        #     params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
        #     params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

    def normal(self, x):

        normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        normal = torch.nan_to_num(normal)

        return normal
