#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB
import imageio
import os as _os_suffix
def get_suffix(p):  # local impl (removed private dmfq dep)
    return _os_suffix.splitext(p)[1]

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    event: np.array
    hdr: np.array
    image_norImg: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    expourse_time: int


class SceneInfo(NamedTuple):
    point_cloud: Optional[BasicPointCloud]
    train_cameras: List
    test_cameras: List
    nerf_normalization: Dict
    ply_path: str


def getNerfppNorm(cam_info: List[CameraInfo]) -> Dict:
    def get_center_and_diag(cam_centers: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}



def fetchPly(path: str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


DIVIDINGLINEDIVIDINGLINEDIVIDINGLINE0=None

def readColmapCameras(
    cam_extrinsics: Dict, cam_intrinsics: Dict, images_folder: str
) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx + 1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # print(f"==>> intr.model: {intr.model}")   # LLNeRF 的 still4 是 SIMPLE_RADIAL  SIMPLE_RADIAL
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), f"Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported! Now is {intr.model}"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            event=None,  # 默认为 None
            hdr=None,    # 默认为 None
            image_norImg=None,  # 默认为 None
            expourse_time=None  # 默认为 None
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos



def readColmapSceneInfo(path: str, images: str, eval: bool, llffhold: int = 8) -> SceneInfo:
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    # print(f"==>> len(train_cam_infos): {len(train_cam_infos)}")
    # print(f"==>> len(test_cam_infos): {len(test_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


DIVIDINGLINEDIVIDINGLINEDIVIDINGLINE1=None

def readCamerasFromTransforms(
    path: str, transformsfile: str, white_background: bool, extension: str = ".png"
) -> List[CameraInfo]:
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        # im_data = np.array(image.convert("RGBA"))

        # bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        # norm_data = im_data / 255.0
        # arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        # image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
                event=None, 
                hdr=None, 
                image_norImg=None,
                expourse_time=None
            )
        )

    return cam_infos


def readNerfSyntheticInfo(
    path: str, white_background: bool, eval: bool, extension: str = ".png"
) -> SceneInfo:
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info

DIVIDINGLINEDIVIDINGLINEDIVIDINGLINE2=None


def readColmapCameras_ev(
    cam_extrinsics: Dict, cam_intrinsics: Dict, images_folder: str
) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):   # y: 后面有排序，所以这里ok
        # print(f"==>> key: {key}")
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx + 1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        print(f"==>> image_path: {image_path}")    # 这里还没排序，外面有排序

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        image_path_norImg = image_path.replace('images_lowImg', 'images_norImg')   # add 250322 
        image_norImg = Image.open(image_path_norImg)
        # add
        if get_suffix(image_path)=='.png':
            event_npy_path = image_path.replace('.png', '.npy').replace('images_lowImg', 'event_npy')  #! event_npy
        elif get_suffix(image_path)=='.JPG':
            event_npy_path = image_path.replace('.JPG', '.npy').replace('images_lowImg', 'event_npy')  #! event_npy
        else:
            raise NotImplementedError
        
        # print(image_path)
        # print('event_npy_path', event_npy_path)
        # print(a)
        if os.path.exists(event_npy_path):
            event = np.load(event_npy_path)
        else:
            print(f"==>> Skip this cam_info, Event path does not exist:{event_npy_path}")
            continue  # 250614 跳过此次循环，可能引入逻辑上的 bug
        # if os.path.exists(event_npy_path):
        #     event = np.load(event_npy_path)
        # else:
        #     event = None
        #     print(f"==>> Event path does not exist, set event=None, path: {event_npy_path}")
        
        # hdr_image = image_path.replace('.png', '.exr').replace('images', 'hdr')
        # if os.path.exists(hdr_image):
        #     hdr = imageio.imread(hdr_image)
        #     hdr = hdr[..., 0:3]
        # else:
        # hdr = None
        # expourse_time_file = images_folder.replace('/images', '/exposure_train.json')
        # if os.path.exists(expourse_time_file):
        #     with open(expourse_time_file) as json_file:
        #         exp_dict = json.load(json_file)
        #         key = './ldr/{}'.format(image_path.split('/')[-1])
        #         expourse_time = exp_dict[key]
        #         # print('expourse_time', expourse_time)
        # else:
        # expourse_time = None
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            event=event,  # 默认为 None
            hdr=None,    # 默认为 None
            image_norImg=image_norImg,  # 默认为 None
            expourse_time=None  # 默认为 None
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos

def readColmapSceneInfo_ev(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # reading_dir = "images" if images == None else images
    reading_dir = "images_lowImg" if images == None else images
    reading_dir = "images_lowImg"   # 
    cam_infos_unsorted = readColmapCameras_ev(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                                images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)  # 这里排序了

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    train_cam_infos = cam_infos
    # 
    # test_cam_infos = []
    test_cam_infos = train_cam_infos
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

DIVIDINGLINEDIVIDINGLINEDIVIDINGLINE3=None


def readCamerasFromTransforms_ev(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_path = os.path.join(path, frame["file_path"].replace('event', 'train') + '.png')
            # image_path = image_path.replace('event_vis', 'train')  # '/home/hjy/ws/EvGS/0_ref/no_ev/GI-GS/datasets/wengflow/robust_e_nerf_data_3dgs_czh/lego/./train/r_0.png'
            # event_path = image_path.replace('train', 'event_npy').replace('.png', '.npy')
            
            
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            event_npy_path = os.path.join(path, path, frame["file_path"].replace('event', 'event_npy') + '.npy')
            if os.path.exists(event_npy_path):
                event = np.load(event_npy_path)
            else:
                event = None
                print(f"==>> Event path does not exist, set event=None, path: {event_npy_path}")

            # EVENTLOAD=None
            # if os.path.exists(event_path):
            #     print(f"==>> will load from event_path: {event_path}")
            #     event = np.load(event_path)
            # else:
            #     print("no event file")
            #     event = None
            # if os.path.exists(os.path.join(path, frame["file_path"].replace('event', 'hdr') + '.exr')):
            #     hdr = imageio.imread(os.path.join(path, frame["file_path"].replace('event', 'hdr') + '.exr'))
            #     hdr = hdr[..., 0:3]
            #     hdr = tonemap(hdr)
            # if os.path.exists(os.path.join(path, frame["file_path"].replace('event', 'hdr_tm') + '.png')):
            #     hdr = imageio.imread(os.path.join(path, frame["file_path"].replace('event', 'hdr_tm') + '.png'))
            #     hdr = hdr / 255.0
            if os.path.exists(os.path.join(path, frame["file_path"].replace('event', 'hdr') + '.exr')):
                hdr = imageio.imread(os.path.join(path, frame["file_path"].replace('event', 'hdr') + '.exr'))
                hdr = hdr[..., 0:3]
            else:
                # print("no hdr file")
                hdr = None
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            event=event, hdr=hdr, image_norImg=None, expourse_time=None))
            
    return cam_infos







def readNerfSyntheticInfo_ev(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms_ev(path, "transforms_event.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d_new.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # xyz = np.random.random((num_pts, 3)) * 1.4 - 0.7
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        # pcd = fetchPly(ply_path)
        xyz = positions
        num_pts = len(xyz)
        # print(num_pts)
        shs = np.random.random((num_pts, 3)) / 255.0
        # rgb = ((pcd.colors)*255).astype(np.uint8)
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        # try:
        #     pcd = fetchPly(ply_path)
        # except:
        #     pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

DIVIDINGLINEDIVIDINGLINEDIVIDINGLINE4=None

# 被 scene_info = sceneLoadTypeCallbacks["Colmap_ev"] 调用
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo, 
    "Blender": readNerfSyntheticInfo, 
    "Colmap_ev" : readColmapSceneInfo_ev,
    "Blender_ev": readNerfSyntheticInfo_ev,           # 240220 实际上也可以重命名为 "Blender_event": readNerfSyntheticInfo_event  
}