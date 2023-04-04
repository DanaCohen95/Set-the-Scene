from pathlib import Path

import numpy as np
from igl import read_obj, write_obj
from scipy.spatial.transform import Rotation as R


def create_obj_files_from_scene_files(scene_dir):
    """
    Converts a scene directory with multiple OBJ files to a single OBJ file.
    """
    scene_path = scene_dir / 'scene.obj'
    with open(scene_path) as f:
        lines = f.readlines()
    objects_data = {}
    current_object_name = None
    current_object_data = []
    current_vertex_count = 0
    vertex_before_object = 0
    for line in lines:
        if line.startswith(('o ', 'g ')):
            # If a new object is encountered, save the previous object to a dictionary
            if current_object_name is not None:
                objects_data[current_object_name] = current_object_data
                current_object_data = []
                vertex_before_object = current_vertex_count
            current_object_name = line.strip().split(' ')[1].split('.')[0]
        elif line.startswith('v '):
            current_vertex_count += 1
        elif line.startswith('f '):
            # Shift the indices of the face by the total number of vertices for previous objects
            indices = line.strip().split(' ')[1:]
            indices = [int(i.split('/')[0]) - vertex_before_object for i in indices]
            line = 'f ' + ' '.join([str(i) for i in indices]) + '\n'
        current_object_data.append(line)
    # Save the last object to the dictionary
    if current_object_name is not None:
        objects_data[current_object_name] = current_object_data

    # Save each object to a file
    for object_name, object_lines in objects_data.items():
        with open(f'{scene_dir}/{object_name}.obj', 'w') as object_file:
            object_file.writelines(object_lines)
    return objects_data.keys()


def calc_obj_params(obj_path):
    """
    Calculates the scale of an object.
    """
    v, _, _, f, _, _ = read_obj(str(obj_path), float)
    v[:, 0] = -v[:, 0]
    center = v.mean(axis=0)
    v = v - center
    proxy_scale = 1
    shape_scale = np.max(np.linalg.norm(v, axis=1))
    if shape_scale < 0.25:
        shape_scale = shape_scale * 2
        proxy_scale = proxy_scale / 2
    shape_scale = np.round(shape_scale, 2)
    center = np.round(center, 2)
    return shape_scale, center, proxy_scale


def transform_to_coordinate_system(obj_path, rotation_clockwise_dict):
    """
    Transforms the coordinate system of an object.
    """
    v, _, _, f, _, _ = read_obj(str(obj_path), float)
    v[:, 0] = -v[:, 0]
    rotation_clockwise = rotation_clockwise_dict[object_name]
    rotation = R.from_euler('y', -rotation_clockwise, degrees=True)
    v = rotation.apply(v)
    write_obj(str(obj_path), v, f)


if __name__ == '__main__':
    scene_dir = Path("scenes/dining")
    rotation_clockwise_dict = {'chair1': 90, 'chair2': 90, 'chair3': 270, 'chair4': 270, 'table': 90, 'room': 0}
    # rotation_clockwise_dict = {'sofa_for_one': 90, 'sofa_for_three': 0, 'table': 0, 'room': 0}
    objects_names = create_obj_files_from_scene_files(scene_dir)

    shape_scales = []
    centers = []
    proxy_scales = []
    for object_name in objects_names:
        obj_path = scene_dir / f'{object_name}.obj'
        transform_to_coordinate_system(obj_path, rotation_clockwise_dict)
        shape_scale, center, proxy_scale = calc_obj_params(obj_path)
        shape_scales.append(shape_scale)
        centers.append(center)
        proxy_scales.append(proxy_scale)

    print("**********************************************************")
    print('scene_proxies:')
    print('\tproxy_to_nerf_ids:', list(objects_names))
    print('\tproxy_centers:', [[x[0], x[1], x[2]] for x in centers])
    print('\tproxy_rotations_clockwise:', [rotation_clockwise_dict[object_name] for object_name in objects_names])
    print('\tproxy_scales:', proxy_scales)
    print('scene_nerfs:')
    print('\tnerf_ids:', list(objects_names))
    print('\tshape_scales:', shape_scales)
    print('\tshape_paths:', [str(scene_dir / f'{object_name}.obj') for object_name in objects_names])
    print('\tnerf_texts:', [f'a {x}' for x in objects_names])
