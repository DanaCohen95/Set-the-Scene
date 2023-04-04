import pyrallis
from src.latent_nerf.configs.scene_config import SceneTrainConfig
from src.latent_nerf.training.scene_trainer import SceneTrainer
import torch


@pyrallis.wrap()
def main(cfg: SceneTrainConfig):
    torch.backends.cudnn.benchmark = True
    scene_trainer = SceneTrainer(cfg)
    if cfg.log.eval_only:
        scene_trainer.full_eval()
    else:
        scene_trainer.train()


if __name__ == '__main__':
    main()
