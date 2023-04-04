import torch
from igl import read_obj
from torch import nn

from src.latent_nerf.models.mesh_utils import MeshOBJ

DELTA = 0.2


def ce_pq_loss(p, q, weight=None):
    def clamp(v, T=0.01):
        return v.clamp(T, 1 - T)

    ce = -1 * (p * torch.log(clamp(q)) + (1 - p) * torch.log(clamp(1 - q)))
    if weight is not None:
        ce *= weight
    return ce.sum()


class ShapeLoss(nn.Module):
    def __init__(self, shape_path: str, mesh_scale, proximal_surface):
        super().__init__()
        v, _, _, f, _, _ = read_obj(shape_path, float)
        self.proximal_surface = proximal_surface
        mesh = MeshOBJ(v[:, :3], f)
        self.sketchshape = mesh.normalize_mesh(mesh_scale)

    def forward(self, xyzs, sigmas):
        mesh_occ = self.sketchshape.winding_number(xyzs)
        if self.proximal_surface > 0:
            weight = 1 - self.sketchshape.gaussian_weighted_distance(xyzs, self.proximal_surface)
        else:
            weight = None
        indicator = (mesh_occ > 0.5).float()
        nerf_occ = 1 - torch.exp(-DELTA * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)
        loss = ce_pq_loss(nerf_occ, indicator,
                          weight=weight)  # order is important for CE loss + second argument may not be optimized
        return loss
