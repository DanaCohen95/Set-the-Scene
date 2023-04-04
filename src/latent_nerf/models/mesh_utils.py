import numpy as np
import torch
from igl import fast_winding_number_for_meshes, point_mesh_squared_distance


def transform_to_scene_coor(x: np.array, rotation_mat: np.array, coord_shift: np.array):
    return np.matmul(x, rotation_mat) + coord_shift


class MeshOBJ:
    dx = torch.zeros(3).float()
    dx[0] = 1
    dy, dz = dx[[1, 0, 2]], dx[[2, 1, 0]]
    dx, dy, dz = dx[None, :], dy[None, :], dz[None, :]

    def __init__(self, v: np.ndarray, f: np.ndarray):
        self.v = v
        self.f = f
        self.dx, self.dy, self.dz = MeshOBJ.dx, MeshOBJ.dy, MeshOBJ.dz
        self.v_tensor = torch.from_numpy(self.v)

        vf = self.v[self.f, :]
        self.f_center = vf.mean(axis=1)
        self.f_center_tensor = torch.from_numpy(self.f_center).float()

        e1 = vf[:, 1, :] - vf[:, 0, :]
        e2 = vf[:, 2, :] - vf[:, 0, :]
        self.face_normals = np.cross(e1, e2)
        self.face_normals = self.face_normals / np.linalg.norm(self.face_normals, axis=-1)[:, None]
        self.face_normals_tensor = torch.from_numpy(self.face_normals)

    def normalize_mesh(self, target_scale=0.5):
        verts = self.v

        # Compute center of bounding box)
        center = verts.mean(axis=0)
        verts = verts - center
        scale = np.max(np.linalg.norm(verts, axis=1))
        verts = (verts / scale) * target_scale
        return MeshOBJ(verts, self.f)

    def transform_mesh_to_scene_coord(self, rotation_mat, coord_shifts):
        transformed_verts = transform_to_scene_coor(self.v, rotation_mat, coord_shifts)
        return MeshOBJ(transformed_verts, self.f)

    def winding_number(self, query: torch.Tensor):
        device = query.device
        shp = query.shape
        query_np = query.detach().cpu().reshape(-1, 3).numpy()
        target_alphas = fast_winding_number_for_meshes(self.v.astype(np.float32), self.f, query_np)
        return torch.from_numpy(target_alphas).reshape(shp[:-1]).to(device)

    def gaussian_weighted_distance(self, query: torch.Tensor, sigma):
        device = query.device
        shp = query.shape
        query_np = query.detach().cpu().reshape(-1, 3).numpy()
        distances, _, _ = point_mesh_squared_distance(query_np, self.v.astype(np.float32), self.f)
        distances = torch.from_numpy(distances).reshape(shp[:-1]).to(device)
        weight = torch.exp(-(distances / (2 * sigma ** 2)))
        return weight
