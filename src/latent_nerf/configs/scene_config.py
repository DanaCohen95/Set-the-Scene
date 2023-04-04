from dataclasses import dataclass, field
from typing import List

from .train_config import TrainConfig


@dataclass
class SceneConfig:
    scene_theta_range = (50, 75)
    scene_phi_range = (0, 90)
    scene_iters: int = 2


@dataclass
class SceneNerfsConfig:
    nerf_ids: List[str]
    nerf_texts: List[str]
    shape_paths: List[str]
    shape_scales: List[float]
    shape_proximal_surfaces = None
    nerf_checkpoint_names: List[str] = None
    is_bg_nerfs: List[bool] = None

    def __post_init__(self):
        if len(self.nerf_ids) != len(self.nerf_texts):
            raise ValueError(r"nerf_texts list should be the length of nerf_ids list")
        if len(self.nerf_ids) != len(self.shape_scales):
            raise ValueError(r"shape_scales list should be the length of nerf_ids list")
        if self.shape_proximal_surfaces is None:
            self.shape_proximal_surfaces = [scale * 0.6 for scale in self.shape_scales[:-1]] + [
                0.05]  # lower proximal for background nerf
        elif len(self.nerf_ids) != len(self.shape_proximal_surfaces):
            raise ValueError(r"shape_proximal_surfaces list should be the length of nerf_ids list")
        if self.is_bg_nerfs is None:
            self.is_bg_nerfs = [False] * (len(self.nerf_ids) - 1) + [True]  # by default last nerf is background nerf
        elif len(self.nerf_ids) != len(self.is_bg_nerfs):
            raise ValueError(r"is_bg_nerfs list should be the length of nerf_ids list")


@dataclass
class SceneProxiesConfig:
    proxy_to_nerf_ids: List[str]
    proxy_centers: List[List[float]]
    proxy_rotations_clockwise: List[int]
    proxy_scales: List[float] = None

    def __post_init__(self):
        if len(self.proxy_to_nerf_ids) != len(self.proxy_centers):
            raise ValueError(r"proxy_center list should be the length of proxy_to_nerf_id ")
        if len(self.proxy_to_nerf_ids) != len(self.proxy_rotations_clockwise):
            raise ValueError(r"proxy_rotation_clockwise list should be the length of proxy_to_nerf_id")
        if self.proxy_scales is None:
            self.proxy_scales = [1] * len(self.proxy_to_nerf_ids)
        elif len(self.proxy_to_nerf_ids) != len(self.proxy_scales):
            raise ValueError(r"nerf_scales list should be the length of proxy_to_nerf_id")


@dataclass
class SceneTrainConfig(TrainConfig):
    """ The main configuration for the trainer """
    scene_nerfs: SceneNerfsConfig = field(default=SceneNerfsConfig)
    scene_proxies: SceneProxiesConfig = field(default=SceneProxiesConfig)
    scene: SceneConfig = field(default=SceneConfig)
