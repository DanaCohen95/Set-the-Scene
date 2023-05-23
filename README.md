# Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes

<a href="https://arxiv.org/abs/2303.13450"><img src="https://img.shields.io/badge/arXiv-2211.07600-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>

> Recent breakthroughs in text-guided image generation have led to remarkable progress in the field of 3D synthesis from text. By optimizing neural radiance fields (NeRF) directly from text, recent methods are able to produce remarkable results. Yet, these methods are limited in their control of each object's placement or appearance, as they represent the scene as a whole. This can be a major issue in scenarios that require refining or manipulating objects in the scene. To remedy this deficit, we propose a novel Global-Local training framework for synthesizing a 3D scene using object proxies. A proxy represents the object's placement in the generated scene and optionally defines its coarse geometry. The key to our approach is to represent each object as an independent NeRF. We alternate between optimizing each NeRF on its own and as part of the full scene. Thus, a complete representation of each object can be learned, while also creating a harmonious scene with style and lighting match. We show that using proxies allows a wide variety of editing options, such as adjusting the placement of each independent object, removing objects from a scene, or refining an object. Our results show that Set-the-Scene offers a powerful solution for scene synthesis and manipulation, filling a crucial gap in controllable text-to-3D synthesis.

<img src="figures/teaser_video.gif" width="800px"/>

## Description

Official Implementation for "Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes".

> TL;DR - we create scenes from text and proxies, such that each object in the scene is represented by a separate NeRF, enabling us to manipulate each object independently.

#### Installation:

1. Install the common dependencies from the requirements.txt file
   `pip install -r requirements.txt`
2. Install igl `conda install -c conda-forge igl`
3. Create a :hugs: token for StableDiffusion

#### Training an existing scene layout:

1. Use the relevant yaml config file in the `demo_configs` folder, and possibly change the nerf_texts parameters to
   match the desired prompts.
2. Run
```bash
 python -m scripts.train_scene_nerf --config_path demo_configs/scene_name.yaml
```
3. The results will be saved in the `results` folder.
4. In order to edit the scene post-training, edit proxies parameters in the config yaml and change the parameter
   `log.eval_only: True`.

#### Creating a new scene layout:

1. Build a scene using Blender or other 3D software, name each object in the scene and save the entire scene as an obj
   file.
2. In the script  `scripts/obj_scene_to_config.py` update rotation_clockwise_dict with the orientation of each object in
   the scene (how much the object is rotated from its from view, in degress)
3. Run  `scripts/obj_scene_to_config.py` script. The obj file for each object will be saved on the scene directory and
   the config parameters will be printed to the console.
4. Copy the printed config parameters to the config file.
5. By default, each proxy is represented using a different nerf. If you want different proxies to use the same model
   change the parameters `scene_nerfs`, `scene_proxies.proxy_to_nerf_ids` accordingly.
6. Run the training as described above.

## Acknowledgments

This code is heavily based on the [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion)
and [latent-nerf](https://github.com/eladrich/latent-nerf).

## Citation

If you use this code for your research, please cite our
paper [Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes](https://arxiv.org/abs/2303.13450)

```
@article{cohenbar2023setthescene,
title={Set-the-Scene: Global-Local Training for Generating Controllable NeRF Scenes},
author={Dana Cohen-Bar and Elad Richardson and Gal Metzer and Raja Giryes and Daniel Cohen-Or},
year={2023},
eprint={2303.13450},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```

