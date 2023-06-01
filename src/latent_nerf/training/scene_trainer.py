from pathlib import Path
from typing import Tuple, Any, Dict, Callable, Union, List

import imageio
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil

from src import utils
from src.adan_optimizer import Adan
from src.latent_nerf.configs.scene_config import SceneTrainConfig
from src.latent_nerf.models.nerf_utils import NeRFType
from src.latent_nerf.models.renderer import NeRFRenderer
from src.latent_nerf.models.scene_nerf import SceneNerf
from src.latent_nerf.training.losses.shape_loss import ShapeLoss
from src.latent_nerf.training.losses.sparsity_loss import sparsity_loss
from src.latent_nerf.training.nerf_dataset import NeRFDataset
from src.stable_diffusion import StableDiffusion

from src.utils import make_path, tensor2numpy


class SceneTrainer():

    def __init__(self, cfg: SceneTrainConfig):
        self.cfg = cfg
        self.train_step = 0
        self.device = torch.device(self.cfg.optim.device)
        if self.cfg.optim.device != 'cpu' and self.cfg.optim.device != 'cuda':
            torch.cuda.set_device(int(self.cfg.optim.device[-1]))

        utils.seed_everything(self.cfg.optim.seed)

        # Make dirs
        self.exp_path = make_path(self.cfg.log.exp_dir)
        self.ckpt_path = make_path(self.exp_path / 'checkpoints')
        self.train_renders_path = make_path(self.exp_path / 'vis' / 'train')
        self.eval_renders_path = make_path(self.exp_path / 'vis' / 'eval')
        self.final_renders_path = make_path(self.exp_path / 'results')

        self.init_logger()
        if self.cfg.log.eval_only:
            pyrallis.dump(self.cfg, (self.exp_path / 'eval_config.yaml').open('w'))
        elif self.cfg.optim.resume:
            pyrallis.dump(self.cfg, (self.exp_path / 'resume_config.yaml').open('w'))
        else:
            pyrallis.dump(self.cfg, (self.exp_path / 'config.yaml').open('w'))

        self.nerf = self.init_nerf()
        self.diffusion = self.init_diffusion()
        self.text_z = self.calc_text_embeddings(self.cfg.guide.text, self.cfg.guide.append_direction)
        self.losses = self.init_losses()
        self.optimizer, self.scaler = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        self.past_checkpoints = []
        if self.cfg.optim.ckpt is not None:
            self.load_checkpoint(self.cfg.optim.ckpt, model_only=False)
        elif self.cfg.optim.resume:
            self.load_checkpoint(model_only=False)

        logger.info(f'Successfully initialized {self.cfg.log.exp_name}')

        self.optimizer, self.scaler = self.init_optimizer()
        self.nerfs_text_z_dict = self.init_nerfs_text_z_dict()
        self.nerf_to_render_idx = 1
        if cfg.render.nerf_type == NeRFType.latent_tune or cfg.render.nerf_type == NeRFType.rgb:
            self.bg_colors = torch.tensor([1, 1, 1], device=self.device)
        else:
            # todo: fix this
            # rand colors
            self.bg_colors = torch.rand((8, 4), device=self.device)

            # self.bg_colors = torch.tensor(
            #     [[0.3250, 0.0149, -0.0848, -0.1631], [0.0561, 0.0183, 0.0417, -0.0660],
            #      [0.0659, -0.0292, -0.0141, -0.0535],
            #      [0.0285, -0.0826, -0.0279, -0.0726], [-0.0463, 0.1970, -0.0373, 0.1260],
            #      [-0.0084, 0.1578, 0.0211, -0.0235], [-0.0695, 0.1445, 0.1823, -0.1139],
            #      [-0.0418, 0.1277, 0.1533, 0.0244]], device=self.device)
        if cfg.scene_nerfs.nerf_checkpoint_names is not None:
            self.load_all_nerf_models(cfg.scene_nerfs.nerf_checkpoint_names)
        self.scene_iters = self.cfg.scene.scene_iters

    def init_diffusion(self) -> nn.Module:
        diffusion_model = StableDiffusion(self.device, concept_name=self.cfg.guide.concept_name,
                                          latent_mode=self.nerf.latent_mode)
        for p in diffusion_model.parameters():
            p.requires_grad = False
        return diffusion_model

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(self.exp_path / 'log.txt', colorize=False, format=log_format)

    def calc_text_embeddings(self, ref_text, append_direction) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not append_direction:
            text_z = self.diffusion.get_text_embeds([ref_text])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom', 'zoom in']:
                text = f"{ref_text}, {d} view"
                text_z.append(self.diffusion.get_text_embeds([text]))
        return text_z

    def init_optimizer(self) -> Tuple[Optimizer, Any]:
        optimizer = Adan(self.nerf.get_params(5 * self.cfg.optim.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0,
                         foreach=False)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.optim.fp16)
        return optimizer, scaler

    def init_nerf(self) -> SceneNerf:
        model = SceneNerf(self.cfg.render, self.cfg.scene_nerfs, self.cfg.scene_proxies, self.cfg.optim.device).to(
            self.device)
        logger.info(
            f'Loaded {self.cfg.render.backbone} NeRF, #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
        logger.info(model)
        return model

    def init_nerfs_text_z_dict(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        text_z_dict = {}
        for nerf_id, nerf in self.nerf.nerfs_dict.items():
            text_z_dict[nerf_id] = self.calc_text_embeddings(nerf.text, append_direction=True)
        return text_z_dict

    def init_losses(self) -> Dict[str, Dict[str, Callable]]:
        losses = {}
        # losses for separate iterations
        for nerf_id, nerf in self.nerf.nerfs_dict.items():
            cur_nerf_loss = {}
            if self.cfg.optim.lambda_shape > 0 and nerf.shape_path:
                cur_nerf_loss['shape_loss'] = ShapeLoss(nerf.shape_path, nerf.shape_scale, nerf.proximal_surface)

            if self.cfg.optim.lambda_sparsity > 0:
                cur_nerf_loss['sparsity_loss'] = sparsity_loss
            losses[nerf_id] = cur_nerf_loss

        if self.cfg.optim.lambda_sparsity > 0:
            losses['sparsity_loss'] = sparsity_loss

        return losses

    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_single_loader = NeRFDataset(self.cfg.render, device=self.device, type='train', H=self.cfg.render.train_h,
                                          W=self.cfg.render.train_w, size=10).dataloader()
        train_scene_loader = NeRFDataset(self.cfg.render, device=self.device, type='train', H=self.cfg.render.train_h,
                                         W=self.cfg.render.train_w, size=10, scene_view=True).dataloader()
        val_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                 W=self.cfg.render.eval_w,
                                 size=1, scene_view=True).dataloader()

        val_scene_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.eval_h,
                                       W=self.cfg.render.eval_w,
                                       size=3, scene_view=True).dataloader()
        # Will be used for creating the final video
        val_large_loader = NeRFDataset(self.cfg.render, device=self.device, type='val', H=self.cfg.render.full_eval_h,
                                       W=self.cfg.render.full_eval_w,
                                       size=self.cfg.log.full_eval_size).dataloader()
        # Will be used for creating the final video
        val_large_scene_loader = NeRFDataset(self.cfg.render, device=self.device, type='val',
                                             H=self.cfg.render.full_eval_h,
                                             W=self.cfg.render.full_eval_w,
                                             size=self.cfg.log.full_eval_size // 4, scene_view=True).dataloader()
        dataloaders = {'train_scene': train_scene_loader, 'train_single': train_single_loader, 'val': val_loader,
                       'val_scene': val_scene_loader, 'val_large': val_large_loader,
                       'val_large_scene': val_large_scene_loader}
        return dataloaders

    def get_params_for_train_step(self):
        kwargs = {}
        nerf_to_render_idx = self.nerf_to_render_idx % (
                len(self.nerf.nerfs_dict) + self.scene_iters)
        if nerf_to_render_idx >= len(self.nerf.nerfs_dict):
            logger.info(f"rendering all scene together")
            train_dataloader = self.dataloaders['train_scene']
            model = self.nerf
            losses = self.losses
            text_z = self.text_z
            is_tv_loss = False
        else:
            single_nerf_to_render = list(self.nerf.nerfs_dict.values())[nerf_to_render_idx]
            logger.info(f"rendering {single_nerf_to_render.id}")
            if single_nerf_to_render.is_bg_nerf:
                train_dataloader = self.dataloaders['train_scene']
            else:
                train_dataloader = self.dataloaders['train_single']
            model = single_nerf_to_render.model
            losses = self.losses[single_nerf_to_render.id]
            text_z = self.nerfs_text_z_dict[single_nerf_to_render.id]
            is_tv_loss = True
        self.nerf_to_render_idx += 1
        kwargs['disable_background'] = False
        kwargs['bg_color'] = self.bg_colors[np.random.randint(len(self.bg_colors))]
        return model, losses, text_z, train_dataloader, is_tv_loss, kwargs

    def train(self):
        logger.info('Starting training ^_^')
        # Evaluate the initialization
        # self.evaluate(self.dataloaders['val'], self.eval_renders_path)
        self.nerf.train()

        pbar = tqdm(total=self.cfg.optim.iters, initial=self.train_step,
                    bar_format='{desc}: {percentage:3.0f}% training step {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        while self.train_step < self.cfg.optim.iters:
            # Keep going over dataloader until finished the required number of iterations
            model, losses, text_z, train_dataloader, is_tv_loss, kwargs = self.get_params_for_train_step()
            for i, data in enumerate(train_dataloader):
                self.train_step += 1
                pbar.update(1)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    pred_rgbs, pred_ws, loss = self.train_render(data, model, losses, text_z, **kwargs)

                self.scaler.scale(loss).backward()
                if is_tv_loss:
                    self.post_train_step(model)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.train_step % self.cfg.log.save_interval == 0:
                    self.save_checkpoint(full=True)
                    self.evaluate()
                    self.nerf.train()

                if np.random.uniform(0, 1) < 0.05:
                    # Randomly log rendered images throughout the training
                    self.log_train_renders(pred_rgbs)
        logger.info('Finished Training ^_^')
        logger.info('Evaluating the last model...')
        self.save_checkpoint(full=True)
        self.full_eval()
        logger.info('\tDone!')

    def log_train_renders(self, pred_rgbs: torch.Tensor):
        if self.nerf.latent_mode:
            pred_rgb_vis = self.diffusion.decode_latents(pred_rgbs).permute(0, 2, 3,
                                                                            1).contiguous()  # [1, 3, H, W]
        else:
            pred_rgb_vis = pred_rgbs.permute(0, 2, 3,
                                             1).contiguous().clamp(0, 1)  #
        save_path = self.train_renders_path / f'step_{self.train_step:05d}.jpg'
        save_path.parent.mkdir(exist_ok=True)

        pred = tensor2numpy(pred_rgb_vis[0])

        Image.fromarray(pred).save(save_path)

    def train_render(self, data: Dict[str, Any], nerf: NeRFRenderer, losses, text_z, **kwargs):
        rays_o, rays_d = data['rays_o'], data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if self.cfg.optim.start_shading_iter is None or self.train_step < self.cfg.optim.start_shading_iter:
            shading = 'albedo'
            ambient_ratio = 1.0
        else:
            shading = 'lambertian'
            ambient_ratio = 0.1

        # with torch.autograd.graph.save_on_cpu():
        outputs = nerf.render(rays_o, rays_d, staged=False, perturb=True, ambient_ratio=ambient_ratio, shading=shading,
                              force_all_rays=True, **kwargs)
        # torch.cuda.empty_cache()
        pred_rgb = outputs['image'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)
        depth = outputs['depth'].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # text embeddings
        if type(text_z) == list:
            dirs = data['dir']  # [B,]
            text_z = text_z[dirs]
        else:
            text_z = text_z

        # calc loss
        loss = self.diffusion.train_step(text_z, pred_rgb, self.train_step, self.cfg.optim.iters)

        if 'sparsity_loss' in losses:
            loss += self.cfg.optim.lambda_sparsity * losses['sparsity_loss'](pred_ws)
            # logger.info('lambda sparsity: {}'.format(self.get_lambda_sparsity() * losses['sparsity_loss'](pred_ws)))

        if 'shape_loss' in losses:
            loss += self.cfg.optim.lambda_shape * losses['shape_loss'](outputs['xyzs'], outputs['sigmas'])

        if self.cfg.optim.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss += self.cfg.optim.lambda_orient * loss_orient

        return pred_rgb, pred_ws, loss

    def eval_render(self, data: Dict[str, Any], nerf: NeRFRenderer, bg_color=None, perturb=False, render_normals=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        outputs = nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                              ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, bg_color=bg_color)

        pred_depth = outputs['depth'].reshape(B, H, W)
        if nerf.latent_mode:
            pred_latent = outputs['image'].reshape(B, H, W, 3 + 1).permute(0, 3, 1, 2).contiguous()
            if self.cfg.log.skip_rgb:
                # When rendering in a size that is too large for decoding
                pred_rgb = torch.zeros(B, H, W, 3, device=pred_latent.device)
            else:
                pred_latent = F.interpolate(pred_latent, (128, 128), mode='bicubic')
                pred_rgb = self.diffusion.decode_latents(pred_latent).permute(0, 2, 3, 1).contiguous()
        else:
            pred_rgb = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)

        pred_depth = pred_depth.unsqueeze(-1).repeat(1, 1, 1, 3)

        # Render again for normals
        pred_normals = None
        if render_normals:
            shading = 'normal'
            outputs_normals = nerf.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                          ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                          disable_background=True)
            pred_normals = outputs_normals['image'][:, :, :3].reshape(B, H, W, 3).contiguous()

        return pred_rgb, pred_depth, pred_normals

    def post_train_step(self, nerf):
        if self.cfg.render.backbone == 'grid':
            lambda_tv = min(1.0, self.train_step / 1000) * self.cfg.optim.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            nerf.encoder.grad_total_variation(lambda_tv, None, nerf.bound)

    def full_eval(self):
        logger.info('evaluating scene nerf')
        self.evaluate_nerf(self.dataloaders['val_large_scene'], self.final_renders_path, nerf_to_render=self.nerf,
                           nerf_name="",
                           save_as_video=True)
        logger.info('evaluating single nerfs')
        # for singe_nerf in self.nerf.single_object_nerfs_list:
        #     logger.info(f'evaluating nerf {singe_nerf.name}')
        #     self.evaluate_nerf(self.dataloaders['val_large'], self.final_renders_path, nerf_to_render=singe_nerf.model,
        #                        nerf_name=singe_nerf.name, save_as_video=True)

    def evaluate(self, save_as_video: bool = False):
        logger.info('evaluating scene nerf')
        self.evaluate_nerf(self.dataloaders['val_scene'], self.eval_renders_path, nerf_to_render=self.nerf,
                           nerf_name="",
                           save_as_video=save_as_video)
        logger.info('evaluating single nerfs')
        for nerf in self.nerf.nerfs_dict.values():
            logger.info(f'evaluating nerf {nerf.id}')
            self.evaluate_nerf(self.dataloaders['val'], self.eval_renders_path, nerf_to_render=nerf.model,
                               nerf_name=nerf.id, save_as_video=save_as_video)

    def evaluate_nerf(self, dataloader: DataLoader, save_path: Path, save_as_video: bool = False,
                      nerf_to_render: NeRFRenderer = None, nerf_name=""):
        logger.info(f'Evaluating and saving model, iteration #{self.train_step}...')
        save_path.mkdir(exist_ok=True)
        if nerf_to_render is None:
            nerf_to_render = self.nerf
        nerf_to_render.eval()

        if save_as_video:
            all_preds = []
            all_preds_normals = []
            all_preds_depth = []

        for i, data in enumerate(dataloader):
            if i % 10 == 0:
                logger.info(f'{i}/{len(dataloader)}')
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.cfg.optim.fp16):
                    preds, preds_depth, preds_normals = self.eval_render(data, nerf_to_render, render_normals=True)

            pred, pred_depth, pred_normals = tensor2numpy(preds[0]), tensor2numpy(preds_depth[0]), tensor2numpy(
                preds_normals[0])

            if save_as_video:
                (save_path / 'images').mkdir(exist_ok=True)
                Image.fromarray(pred.copy()).save(save_path / f"images/{self.train_step}_{i:04d}_rgb_{nerf_name}.png")
                all_preds.append(pred)
                all_preds_normals.append(pred_normals)
                all_preds_depth.append(pred_depth)
            else:
                if not self.cfg.log.skip_rgb:
                    Image.fromarray(pred).save(save_path / f"{self.train_step}_{i:04d}_rgb_{nerf_name}.png")
                Image.fromarray(pred_normals).save(save_path / f"{self.train_step}_{i:04d}_normals_{nerf_name}.png")
                Image.fromarray(pred_depth).save(save_path / f"{self.train_step}_{i:04d}_depth_{nerf_name}.png")

        if save_as_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_normals = np.stack(all_preds_normals, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            dump_vid = lambda video, name: imageio.mimsave(save_path / f"{self.train_step}_{name}.mp4", video, fps=15,
                                                           quality=8, macro_block_size=1)

            if not self.cfg.log.skip_rgb:
                dump_vid(all_preds, f'rgb_{nerf_name}')
            dump_vid(all_preds_normals, f'normals_{nerf_name}')
            dump_vid(all_preds_depth, f'depth_{nerf_name}')
        logger.info('Done!')

    def load_all_nerf_models(self, ckpt_list):
        for i, nerf in enumerate(self.nerf.nerfs_dict.values()):
            state_dict = torch.load(ckpt_list[i], map_location=self.device)
            missing_keys, unexpected_keys = nerf.model.load_state_dict(state_dict, strict=False)
            logger.info(f"checkpoint {ckpt_list[i]} was loaded")
            if len(missing_keys) > 0:
                logger.warning(f"missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logger.warning(f"unexpected keys: {unexpected_keys}")

    def save_all_nerf_models(self, cur_ckpt_dir: Path):
        for model_id, nerf in self.nerf.nerfs_dict.items():
            state_dict = nerf.model.state_dict()
            torch.save(state_dict, cur_ckpt_dir / f'{model_id}.pth')

    def save_checkpoint(self, full=False):
        name = f'step_{self.train_step:06d}'

        state = {
            'train_step': self.train_step,
            'checkpoints': self.past_checkpoints,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['scaler'] = self.scaler.state_dict()

        cur_ckpt_dir = self.ckpt_path / name
        cur_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_all_nerf_models(cur_ckpt_dir)

        self.past_checkpoints.append(cur_ckpt_dir)

        if len(self.past_checkpoints) > self.cfg.log.max_keep_ckpts:
            old_ckpt = self.past_checkpoints.pop(0)
            if old_ckpt.exists():
                shutil.rmtree(old_ckpt)

        torch.save(state, cur_ckpt_dir / 'train_params.pth')

    def load_checkpoint(self, ckpt_dir: str = None, model_only=False):
        if ckpt_dir is None:
            checkpoint_list = sorted(self.ckpt_path.glob('*'))
            if checkpoint_list:
                ckpt_dir = checkpoint_list[-1]
                logger.info(f"Latest checkpoint is {ckpt_dir}")
            else:
                logger.info(f"No checkpoint found for {self.ckpt_path}, model randomly initialized.")
                return

        nerfs_ckpt_list = [ckpt_dir / f'{model_id}.pth' for model_id in self.nerf.nerfs_dict.keys()]
        self.load_all_nerf_models(nerfs_ckpt_list)

        train_params_path = Path(ckpt_dir / 'train_params.pth')
        train_params_dict = torch.load(train_params_path, map_location=self.device)
        if model_only:
            return

        self.past_checkpoints = train_params_dict['checkpoints']
        self.train_step = train_params_dict['train_step'] + 1
        logger.info(f"load at step {self.train_step}")

        if self.optimizer and 'optimizer' in train_params_dict:
            try:
                self.optimizer.load_state_dict(train_params_dict['optimizer'])
                logger.info("loaded optimizer.")
            except:
                logger.warning("Failed to load optimizer.")

        if self.scaler and 'scaler' in train_params_dict:
            try:
                self.scaler.load_state_dict(train_params_dict['scaler'])
                logger.info("loaded scaler.")
            except:
                logger.warning("Failed to load scaler.")
