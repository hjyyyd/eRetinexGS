'''
eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement
            via Event Streams and 3D Gaussian Splatting.
Training entry point.
'''

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Optional, Tuple, Union
import math
import time

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange
from diff_gaussian_rasterization import Gaussian_SSR
import gc
from lpips import LPIPS

from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from pbr import CubemapLight, get_brdf_lut, pbr_shading, pbr_shading_retinex
from scene import GaussianModel, Scene, Camera
from utils.general_utils import safe_state
from utils.image_utils import psnr, turbo_cmap, erode
from utils.loss_utils import l1_loss, ssim, get_img_grad_weight
from utils.graphics_utils import normal_from_depth_image

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def g(*args, **kwargs):  # debug no-op (removed private dmfq dep)
    return None
from eRetinexGS.code.utils import mkdir_and_rename,copyAllFoldersAndFiles
from eRetinexGS.code.utils import CSVWriter_y
from eRetinexGS.code.utils import stack_grid_to_cmap

from eRetinexGS.code.utils import gamma_brten_before_crf_y,get_brightness_weights,img_loss_weight_brht_fn,variation_similarity_loss_func, cosine_color_loss_func,chromaticity_loss_func,compute_white_balance_loss
from eRetinexGS.code.utils import gainVis, linear_to_sRGB_y
from eRetinexGS.code.utils import color_const_loss_fn,llnerf_gray_loss_y,gray_loss,get_v_n3_enhanced,ltv_loss,log_data_loss_func,RawNeRF_loss_func
from utils_crf import HDR_NeRF_CRF, EventDegrade, draw_CRF_xLinear_tbwriter_add_figure, get_crf_regularization_loss
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import scipy.io as scio
import os



import torch.nn as nn




def ev_masked_tv_loss_fn(img, event_mask):
    """
    img: [3, H, W] - image tensor (R or I)
    event_mask: [1, H, W] - 0: smooth area (apply loss), 1: edge (ignore)
    """
    grad_x = img[:, :, :-1] - img[:, :, 1:]  # [3, H, W-1]
    grad_y = img[:, :-1, :] - img[:, 1:, :]  # [3, H-1, W]

    # Prepare inverse mask: 1 where we want to apply loss (i.e., mask==0)
    inv_mask_x = (1 - event_mask[:, :, :-1])  # [1, H, W-1]
    inv_mask_y = (1 - event_mask[:, :-1, :]) # [1, H-1, W]

    loss_x = (torch.abs(grad_x) * inv_mask_x).mean()
    loss_y = (torch.abs(grad_y) * inv_mask_y).mean()

    return loss_x + loss_y



def no_ev_loss_fn(diff_image, event_threshold, no_ev_mask):

    diff_abs = torch.abs(diff_image)  # (B, T-1, H, W)

    no_ev_loss = torch.sum( F.relu(diff_abs - event_threshold) * no_ev_mask) / torch.sum(no_ev_mask)

    return no_ev_loss






def l1_mask_loss(network_output, gt, mask=None):
    if mask == None:
        return torch.abs((network_output - gt)).mean()
    else:
        # print(torch.sum(mask))
        return torch.sum(torch.abs((network_output - gt)) * mask) / torch.sum(mask)


def ev_loss_weight_dark_fn(weight_dark, network_output, gt, mask=None):
    if mask == None:
        # return torch.abs((network_output - gt)).mean()
        raise NotImplementedError
    else:
        # print(torch.sum(mask))
        return torch.sum(torch.abs((network_output - gt)*weight_dark) * mask) / torch.sum(mask)


def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale],
                                            intrinsic_matrix.to(depth.device),
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def resize_tensorboard_img(
    img: torch.Tensor,  # [C, H, W]
    max_res: int = 800,
) -> torch.Tensor:
    _, H, W = img.shape
    ratio = min(max_res / H, max_res / W)
    target_size = (int(H * ratio), int(W * ratio))
    transform = T.Resize(size=target_size)
    img = transform(img)  # [C, H', W']
    return img


def get_enhc_loss_img(args, llnerf_r_albedo_n3_map,llnerf_v_viewdep_n1_map,llnerf_alpha_n1_map,llnerf_gamma_n3_map):
    loss_dict_enhc={}

    R = llnerf_r_albedo_n3_map
    L = llnerf_v_viewdep_n1_map
    L_enhc = get_v_n3_enhanced(L,llnerf_alpha_n1_map,llnerf_gamma_n3_map)
    render_retinex_enhc = pbr_shading_retinex(R, L_enhc)["render_retinex"]    # rgb_enhc
    rgb_enhc = render_retinex_enhc
    assert not torch.isnan(rgb_enhc).any(), "rgb_enhc contains NaN values"
    assert not torch.isinf(rgb_enhc).any(), "rgb_enhc contains inf values"
    assert rgb_enhc.shape[-1] > 0, "Last dimension of rgb_enhc is empty"
    mean_rgb_enhc = rgb_enhc.mean(dim=-1)
    assert torch.all(mean_rgb_enhc < 1e6), "mean_rgb_enhc is too large"

    # 1. Exposure Loss (曝光损失)
    # print(f"==>> rgb_enhc.shape: {rgb_enhc.shape}")  # ([3, 770, 1156])
    if "loss_exp" in args.enhc_loss_list:
        hyper_exposure_loss_mult = 0.1
        hyper_fixed_exposure = 0.55
        loss_exp = hyper_exposure_loss_mult * torch.mean((rgb_enhc.mean(dim=-1) - hyper_fixed_exposure)**2)
        loss_dict_enhc['loss_exp'] = loss_exp
        assert not torch.isnan(loss_exp).any(), f"loss_exp contains NaN values after loss_exp: {loss_exp}"

    # 2: Gray Loss (灰度损失)：基于灰度世界假设，约束增强后的 RGB 图像的颜色分布，使其接近灰度。
    #* 对应 llnerf paper, eq(9):        rgb_enhc.permute(1, 2, 0) 将 3hw -> hw3
    if "loss_gray" in args.enhc_loss_list:
        hyper_gray_loss_mult = 0.1
        R_norm = (R / R.max()).detach()
        loss_gray = hyper_gray_loss_mult * gray_loss(rgb_enhc.permute(1, 2, 0), ref=R_norm.permute(1, 2, 0))
        loss_dict_enhc['loss_gray'] = loss_gray
        assert not torch.isnan(loss_gray).any(), f"loss_gray contains NaN values after loss_gray: {loss_gray}"

    # 4-5  smooth prior
    L_sg = L.detach()
    # print(f"==>> L_sg.shape: {L_sg.shape}")  # ([1, 770, 1156])
    coeff_ref_alpha = L_sg[ 0:1,:,:]   # 1,H,W  #  原本 llnerf 这里 L_sg[..., 0:1]
    coeff_ref_gamma = L_sg

    # 5. Alpha LTV Loss (Alpha 局部总变差损失)：约束 Alpha 参数（alpha）的局部变化一致性。
    if "loss_Gltv" in args.enhc_loss_list:
        hyper_alpha_ltv_loss_mult = 0.1
        loss_Altv = hyper_alpha_ltv_loss_mult * ltv_loss(llnerf_alpha_n1_map, coeff_ref_alpha)
        loss_dict_enhc['loss_Altv'] = loss_Altv
        assert not torch.isnan(loss_Altv).any(), f"loss_Altv contains NaN values after loss_Altv: {loss_Altv}"

    # 4. Gamma LTV Loss (Gamma 局部总变差损失)：约束 Gamma 校正参数（gamma）的局部变化一致性。
    if "loss_Altv" in args.enhc_loss_list:
        hyper_gamma_ltv_loss_mult = 0.1
        loss_Gltv = hyper_gamma_ltv_loss_mult * ltv_loss(llnerf_gamma_n3_map, coeff_ref_gamma)
        loss_dict_enhc['loss_Gltv'] = loss_Gltv
        assert not torch.isnan(loss_Gltv).any(), f"loss_Gltv contains NaN values after loss_Gltv: {loss_Gltv}"


    # add
    if "loss_gamma_norm" in args.enhc_loss_list:
        hyper_gamma_norm_loss_mult = 0.01
        loss_gamma_norm = hyper_gamma_norm_loss_mult * torch.mean(llnerf_gamma_n3_map**2)
        loss_dict_enhc['loss_gamma_norm'] = loss_gamma_norm

    return loss_dict_enhc

def get_enhc2(render_image, alpha_scalar=1.0 , beta_scalar=0.2, gamma_scalar=3.5, epsilon=1e-6 ): # todo
    render_image = torch.clamp(render_image, 0.0, 1.0)
    # 加入 epsilon 防止 render_image / beta_scalar 为 0
    safe_input = render_image / beta_scalar + epsilon   # 250326 之前没有 加 epsilon， 报NaN Bug, 因为这个操作，梯度在趋于0时为无穷大
    render_image_enhc2 = ( safe_input **(1/gamma_scalar) )/ alpha_scalar  # 1/3.5 不直接导致当前值 NaN， 但会影响反传之后， albedo 变成 NaN
    render_image_enhc2 = torch.clamp(render_image_enhc2, 0.0, 1.0)
    return render_image_enhc2


def get_degrade(render_image, alpha_scalar=1.0 , beta_scalar=0.2, gamma_scalar=3.5, epsilon=1e-6 ): # todo
    render_image = torch.clamp(render_image, 0.0, 1.0)
    render_image_degrade = beta_scalar * (alpha_scalar * render_image + epsilon)** gamma_scalar
    # 限制结果在 [0,1] 内
    render_image_degrade = torch.clamp(render_image_degrade, 0.0, 1.0)
    return render_image_degrade


    #! 曾用名 brightness_loss_func
def brightness_mean_pixel_loss_func(rendered_image, target_brightness=0.4):  # llnerf: 0.55
    avg_brightness = torch.mean(rendered_image) # 是 一个float  #! render_image: 3HW
    return torch.abs(avg_brightness - target_brightness)

def brightness_mean_gray_loss_func(rendered_image, target_brightness=0.4):  # 0.55
    """
    ChatGPT:
    计算基于灰度的平均亮度损失。
    :param rendered_image: torch.Tensor, 形状为 (3, H, W)，RGB 格式，值域 [0, 1]
    :param target_brightness: float，期望的灰度亮度
    :return: 单个标量 loss
    """
    assert rendered_image.shape[0] == 3, "输入图像应为 3xHxW 的 RGB 图像"
    # 计算灰度图：Y = 0.299 * R + 0.587 * G + 0.114 * B
    gray = 0.299 * rendered_image[0] + 0.587 * rendered_image[1] + 0.114 * rendered_image[2]
    # 计算平均灰度亮度
    avg_gray = gray.mean()      # 是 一个float
    return torch.abs(avg_gray - target_brightness)


def training(
    dataset: GroupParams,
    opt: GroupParams,
    pipe: GroupParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_iterations: int,
    signal_supervise: str,
    ev_align_pre_or_post: str,
    checkpoint_path: Optional[str] = None,
    pbr_iteration: int = 30_000,
    debug_from: int = -1,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    normal_tv_weight: float = 1.0,
    brdf_tv_weight: float = 1.0,
    env_tv_weight: float = 0.01,
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8,
    indirect: bool = False,
    args: Namespace = None
) -> None:
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    print(f"==>> opt.densify_grad_threshold: {opt.densify_grad_threshold}")
    # add crf 250329
    if args.HDR_NeRF_CRF_split == 'True':
        spilit = True
    elif args.HDR_NeRF_CRF_split == 'False':
        spilit = False
    hdr_nerf_crf = HDR_NeRF_CRF(spilit=spilit)
    hdr_nerf_crf.train()
    # Event-branch degradation model F (frame branch uses the CRF above as G).
    event_degrade = EventDegrade()
    event_degrade.train()
    grad_vars_crf = list(hdr_nerf_crf.parameters()) + list(event_degrade.parameters())
    lrate_crf=5e-4
    optimizer_crf = torch.optim.Adam(params=grad_vars_crf, lr=lrate_crf, betas=(0.9, 0.999))
    hdr_nerf_crf.to(device="cuda")
    event_degrade.to(device="cuda")

    gamma_ot = args.gamma_ot
    # if signal_supervise=='img':
    #     scene = Scene(dataset, gaussians, shuffle=False)
    # elif signal_supervise=='ev':
    #     scene = Scene(dataset, gaussians, shuffle=False)
    # elif signal_supervise=='img_ev':
    #     scene = Scene(dataset, gaussians, shuffle=False)
    # else:
    #     raise NotImplementedError
    scene = Scene(dataset, gaussians, shuffle=False)

    gaussians.training_setup(opt)
    tb_writer = prepare_output_and_logger(dataset)

    # add save code to output folders
    newFolder = f'{args.model_path}/src'
    mkdir_and_rename(newFolder)
    copyAllFoldersAndFiles(srcRoot= './',
            desRoot = newFolder,
            exclude_txt_path='.exclude.txt')  #
            # ignorePatterns=set(('archive','data','outputs','submodules','utils_git','utils_sync','0_del'))'_*' ,'.*', '_*'   #,'*.sh'
    print(f'Done! maked new copy to folder: {newFolder}')


    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    if args.parser_background == 'White':
        bg_color = [1, 1, 1]
    elif args.parser_background == 'Black':
        bg_color = [0, 0, 0]
    else:
        raise NotImplementedError
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # NOTE: prepare for PBR
    brdf_lut = get_brdf_lut().cuda()
    envmap_dirs = get_envmap_dirs()
    cubemap = CubemapLight(base_res=256).cuda()
    cubemap.train()
    # print(torch.isnan(cubemap.base).any())

    param_groups = [
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.opacity_lr}
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    canonical_rays = scene.get_canonical_rays()

    # load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        first_iter = checkpoint["iteration"]
        gaussians.restore(model_params, opt)

        print(f"Load checkpoint from {checkpoint_path}")

    # define progress bar
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = trange(first_iter, opt.iterations, desc="Training progress")  # For logging
    viewpoint_stack = scene.getTrainCameras().copy()

    t_start = time.time()
    ev_skip_max = args.ev_skip_max
    # print(f"==>> ev_skip_max: {ev_skip_max}")
    for iteration in range(first_iter + 1, opt.iterations + 1):  # the real iteration (1 shift)
        loss_dict={}
        iter_start.record()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        POPEVENT=None
        # ev_skip = randint(0, 1)
        # ev_skip = randint(0, 12)    # (0, 3)
        ev_skip = randint(0, ev_skip_max)

        img_index = randint(0, len(viewpoint_stack)-2-ev_skip) #左闭右闭  0,..,397
        # print(f"==>> img_index: {img_index}")
        viewpoint_cam = viewpoint_stack[img_index]


        event_list = [viewpoint_cam.event]

        for i in range(ev_skip+1):  # range(2)时 , i=0,1
            viewpoint_cam_1 = viewpoint_stack[img_index+1+i]
            event_list.append(viewpoint_cam_1.event)

        # changed from exp 220b
        # event_list = event_list[0:-1]
        if ev_align_pre_or_post=='post':
            event_list = event_list[1:]
        elif ev_align_pre_or_post=='pre':
            event_list = event_list[:-1]
        event = torch.sum(torch.stack(event_list), dim=0, keepdim=True).cuda() # 没有event file 是置0？


        try:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
            c2w_1 = torch.inverse(viewpoint_cam_1.world_view_transform.T)
        except:
            continue

        pipe.debug

        RENDERINGRESULT=None
        rendering_result = render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            pad_normal=False,
            derive_normal=True,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )

        rendering_result_1 = render(
            viewpoint_camera=viewpoint_cam_1,
            pc=gaussians,
            pipe=pipe,
            bg_color=background,
            pad_normal=False,
            derive_normal=True,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )

        render_image = rendering_result["render"]  # [3, H, W]      # rendering_result["render"]对应valinna 3DGS直出的结果  pbr_result["render_retinex"] 对应shading后的结果
        viewspace_point_tensor = rendering_result["viewspace_points"]
        visibility_filter = rendering_result["visibility_filter"]
        radii = rendering_result["radii"]
        normal_map_from_depth = rendering_result["normal_map_from_depth"]  # [3, H, W]
        normal_map = rendering_result["normal_map"]  # [3, H, W]
        albedo_map = rendering_result["albedo_map"]  # [3, H, W]
        # 250222
        llnerf_r_albedo_n3_map  = rendering_result["llnerf_r_albedo_n3_map"]  # [3, H, W]
        llnerf_v_viewdep_n1_map = rendering_result["llnerf_v_viewdep_n1_map"]  # [1, H, W]
        llnerf_alpha_n1_map  = rendering_result["llnerf_alpha_n1_map"]  # [3, H, W]
        llnerf_gamma_n3_map = rendering_result["llnerf_gamma_n3_map"]  # [1, H, W]

        roughness_map = rendering_result["roughness_map"]  # [1, H, W]
        metallic_map = rendering_result["metallic_map"]  # [1, H, W]
        # allmap = rendering_result["allmap"]

        render_image_1 = rendering_result_1["render"]
        normal_map_1 = rendering_result_1["normal_map"]  # [3, H, W]
        albedo_map_1 = rendering_result_1["albedo_map"]  # [3, H, W]
        llnerf_r_albedo_n3_map_1  = rendering_result_1["llnerf_r_albedo_n3_map"]  # [3, H, W]
        llnerf_v_viewdep_n1_map_1 = rendering_result_1["llnerf_v_viewdep_n1_map"]  # [1, H, W]
        roughness_map_1 = rendering_result_1["roughness_map"]  # [1, H, W]
        metallic_map_1 = rendering_result_1["metallic_map"]  # [1, H, W]
        #
        rmax, rmin = 1.0, 0.04
        roughness_map = roughness_map * (rmax - rmin) + rmin

        # NOTE: mask normal map by view direction to avoid skip value
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width

        # Loss
        alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        if args.which_img_gt=='norImg':
            gt_image = viewpoint_cam.original_image_norImg[0:3, :, :].cuda()   # 250403 gt_image 是 RGB 通道，  不是 BGR通道
        elif args.which_img_gt=='lowImg':
            gt_image = viewpoint_cam.original_image[0:3, :, :].cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        alpha_mask_1 = viewpoint_cam_1.gt_alpha_mask.cuda()
        if args.which_img_gt=='norImg':
            gt_image_1 = viewpoint_cam_1.original_image_norImg[0:3, :, :].cuda()
        elif args.which_img_gt=='lowImg':
            gt_image_1 = viewpoint_cam_1.original_image[0:3, :, :].cuda()
        gt_image_1 = (gt_image_1 * alpha_mask_1 + background[:, None, None] * (1.0 - alpha_mask_1)).clamp(0.0, 1.0)

        loss: torch.Tensor

        color_mask = torch.zeros_like(gt_image).double() # 3,H,W
        color_mask[0, 0::2, 0::2] = 1  # r
        color_mask[1, 0::2, 1::2] = 1  # g
        color_mask[1, 1::2, 0::2] = 1  # g
        color_mask[2, 1::2, 1::2] = 1  # b
        event_mask = event.clone()
        event_mask[event_mask!=0] = 1  # event_mask: 1HW， 0表示没有事件的地方，1表示有事件的地方
        # event_mask 取反
        no_ev_mask = 1-event_mask

        loss = 0
        #* iteration <= pbr_iteration
        if iteration <= pbr_iteration:

            LOSSFIRSTSTAGE=None
            #* img loss
            #! add crf
            render_image_sRGB = hdr_nerf_crf(gamma_brten_before_crf_y(render_image))   # render_image 3hw

            #! confidence-guided complementarity: the grayscale of the recovered
            #  radiance is a per-pixel confidence — frames are trusted where it is
            #  bright, events where it is dark.
            weight_brht, weight_dark = get_brightness_weights(render_image.detach())  # [1,H,W]

            #! img loss (frame branch, weighted by frame confidence)
            img_loss = img_loss_weight_brht_fn(weight_brht, render_image_sRGB, gt_image, use_mask=True)
            img_loss = (1.0 - opt.lambda_dssim) * img_loss   # opt.lambda_dssim=0.2
            loss += args.lambda_img_loss * img_loss
            loss_dict['img_loss'] = img_loss

            img_ssim_loss = (1.0 - ssim(render_image_sRGB, gt_image))
            img_ssim_loss = opt.lambda_dssim * img_ssim_loss
            loss += args.lambda_img_loss_ssim * img_ssim_loss
            loss_dict['img_ssim_loss'] = img_ssim_loss


            #! event loss (event branch, F models the event-camera degradation)
            diff_image = torch.log(event_degrade(render_image_1) + 1e-8) - torch.log(event_degrade(render_image) + 1e-8)
            event_threshold = 0.2
            diff_gt = event * event_threshold
            event_loss = ev_loss_weight_dark_fn(weight_dark, diff_image * color_mask, diff_gt * color_mask, event_mask)
            tmp_event_loss = (1.0 - opt.lambda_dssim) * event_loss
            loss += args.lambda_ev_loss * tmp_event_loss
            loss_dict['event_loss'] = event_loss

            event_ssim_loss = (1.0 - ssim(diff_image * color_mask, diff_gt * color_mask))     # float tensor
            tmp_event_ssim_loss = opt.lambda_dssim * event_ssim_loss
            loss += args.lambda_ev_loss_ssim * tmp_event_ssim_loss
            loss_dict['event_ssim_loss'] = event_ssim_loss

            #! brightness loss
            if args.args_brightness_loss=='brightness_mean_pixel_loss':     #! 403系列 合成数据实验认为 好
                brightness_loss = brightness_mean_pixel_loss_func(linear_to_sRGB_y(render_image,gamma=gamma_ot), target_brightness=args.target_brightness)
                loss += args.lambda_brightness_loss * brightness_loss
                loss_dict['brightness_loss'] = brightness_loss
            elif args.args_brightness_loss=='brightness_mean_gray_loss':   #! 403系列 合成数据实验认为 差
                brightness_loss = brightness_mean_gray_loss_func(linear_to_sRGB_y(render_image,gamma=gamma_ot), target_brightness=args.target_brightness)
                loss += args.lambda_brightness_loss * brightness_loss
                loss_dict['brightness_loss'] = brightness_loss




        #* iteration > pbr_iteration
        else:  # NOTE: Retinex
            LOSSSECONDSTAGE=None
            # img_loss = F.l1_loss(render_image, gt_image)

            R = llnerf_r_albedo_n3_map
            L = llnerf_v_viewdep_n1_map
            R_1 = llnerf_r_albedo_n3_map_1
            L_1 = llnerf_v_viewdep_n1_map_1
            render_retinex = pbr_shading_retinex(R, L)["render_retinex"]
            render_retinex_1 = pbr_shading_retinex(R_1, L_1)["render_retinex"]

            #! add crf
            render_retinex_degrade = hdr_nerf_crf(gamma_brten_before_crf_y(render_retinex))   # render_image 3hw
            render_retinex_1_degrade = hdr_nerf_crf(gamma_brten_before_crf_y(render_retinex_1))   # render_image 3hw


            #! confidence-guided complementarity (recovered radiance grayscale -> per-pixel confidence)
            weight_brht, weight_dark = get_brightness_weights(render_retinex.detach())  # [1,H,W]

            #! img loss retinex (frame branch G, weighted by frame confidence)
            retinex_render_loss = img_loss_weight_brht_fn(weight_brht, render_retinex_degrade, gt_image, use_mask=True)
            loss += args.lambda_retinex_render_loss * retinex_render_loss
            loss_dict['retinex_render_loss'] = retinex_render_loss

            #! ev loss retinex (event branch, F models the event-camera degradation)
            diff_render_retinex = torch.log(event_degrade(render_retinex_1) + 1e-8) - torch.log(event_degrade(render_retinex) + 1e-8)
            event_threshold = 0.2
            diff_gt = event * event_threshold

            retinex_render_ev_loss = ev_loss_weight_dark_fn(weight_dark, diff_render_retinex * color_mask, diff_gt * color_mask, event_mask)
            retinex_render_ev_loss = (1.0 - opt.lambda_dssim) * retinex_render_ev_loss

            loss += args.lambda_retinex_render_ev_loss * retinex_render_ev_loss
            loss_dict['retinex_render_ev_loss'] = retinex_render_ev_loss
            assert not torch.isnan(retinex_render_ev_loss).any(), f"retinex_render_ev_loss contains NaN values"


            retinex_render_ev_loss_ssim = (1.0 - ssim(diff_render_retinex * color_mask, diff_gt * color_mask))
            retinex_render_ev_loss_ssim = opt.lambda_dssim * retinex_render_ev_loss_ssim
            loss += args.lambda_retinex_render_ev_loss_ssim * retinex_render_ev_loss_ssim
            loss_dict['retinex_render_ev_loss_ssim'] = retinex_render_ev_loss_ssim
            assert not torch.isnan(retinex_render_ev_loss_ssim).any(), f"retinex_render_ev_loss_ssim contains NaN values"


            #! brightness loss
            if args.args_brightness_loss=='brightness_mean_pixel_loss':     #! 403系列 合成数据实验认为 好
                brightness_loss = brightness_mean_pixel_loss_func(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), target_brightness=args.target_brightness)
                loss += args.lambda_brightness_loss * brightness_loss
                loss_dict['brightness_loss'] = brightness_loss
            elif args.args_brightness_loss=='brightness_mean_gray_loss':   #! 403系列 合成数据实验认为 差
                brightness_loss = brightness_mean_gray_loss_func(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), target_brightness=args.target_brightness)
                loss += args.lambda_brightness_loss * brightness_loss
                loss_dict['brightness_loss'] = brightness_loss

            #! event-guided reflectance smoothness (masked TV on reflectance R)
            if args.args_ev_masked_tv_loss=='ev_masked_tv_loss':
                loss_evMasked_tv_R = ev_masked_tv_loss_fn(R, event_mask)  # event_mask: 1HW
                loss += args.lambda_ev_masked_tv_loss_R * loss_evMasked_tv_R
                loss_dict['loss_evMasked_tv_R'] = loss_evMasked_tv_R

            #! gray-world prior: channel-wise variance of the recovered radiance,
            #  relaxed where the reflectance itself is genuinely colorful.
            if args.args_color_const_loss == "llnerf_gray_loss":
                var_I = render_retinex.var(dim=0)            # [H,W]
                var_R = R.var(dim=0).detach()                # [H,W]
                loss_gray = (var_I / (0.5 + var_R)).mean()
                loss += args.lambda_color_const_loss * loss_gray
                loss_dict['color_const_loss'] = loss_gray
                assert not torch.isnan(loss_gray).any(), f"loss_gray contains NaN values: {loss_gray}"


        loss_dict['total_loss'] = loss
        LOSSBACKWARD=None
        loss.backward()
        iter_end.record()

        elapsed = (time.time() - t_start)/60   # unit: minute
        with torch.no_grad():
            # Progress bar
            # ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            input_exps_i = hdr_nerf_crf.input_exps.item()
            # Log and save
            training_report(
                tb_writer=tb_writer,
                iteration=iteration,
                first_iter=first_iter,
                input_exps_i=input_exps_i,
                loss_dict=loss_dict,
                elapsed=elapsed,
                testing_iterations=testing_iterations,
                scene=scene,
                light=cubemap,
                brdf_lut=brdf_lut,
                canonical_rays=canonical_rays,
                pbr_iteration=pbr_iteration,
                metallic=metallic,
                tone=tone,
                gamma=gamma,
                radius=radius,
                bias=bias,
                thick=thick,
                delta=delta,
                step=step,
                start=start,
                renderArgs=(pipe, background),
                indirect=indirect,
                args=args,
                hdr_nerf_crf=hdr_nerf_crf,
                color_mask=color_mask,
                event_threshold=event_threshold
            )
            # NOTE: we save .pth instead of point cloud for additional irradiance volumes and cubemap
            # uncomment
            SAVING=None
            if iteration in saving_iterations:
               print(f"\n[ITER {iteration}] Saving Gaussians")
               scene.save(iteration)   # 保存 point_cloud 子文件夹
            if iteration in checkpoint_iterations: # checkpoint_iterations: [30000, 40000]
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "cubemap": cubemap.state_dict(),
                        # "irradiance_volumes": irradiance_volumes.state_dict(),
                        "light_optimizer": light_optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )



            if iteration in saving_iterations:    # save_iterations 100 200 300 1000 2000 3000 7000 30000 32000 36000 40000
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "cubemap": cubemap.state_dict(),
                        # "irradiance_volumes": irradiance_volumes.state_dict(),
                        "light_optimizer": light_optimizer.state_dict(),
                        "iteration": iteration,
                    },
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

                torch.save({
                        'iteration': iteration,
                        'hdr_nerf_crf': hdr_nerf_crf.state_dict(),
                        'event_degrade': event_degrade.state_dict(),
                        'optimizer_crf': optimizer_crf.state_dict()
                        },
                        scene.model_path + "/chkpnt_hdr_nerf_crf" + str(iteration) + ".pth"
                )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                gaussians.update_learning_rate(iteration)
                if iteration >= pbr_iteration:
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0)

            if iteration >= 300:
                # crf add 250329
                CRF=None
                optimizer_crf.step()
                optimizer_crf.zero_grad(set_to_none=True)
                decay_rate = 0.1
                # lrate_decay = 250
                decay_steps = 30000 # 40k 指数衰减到 lrate_crf 的 0.1
                global_step = iteration
                new_lrate = lrate_crf * (decay_rate ** (global_step / decay_steps))
                for param_group in optimizer_crf.param_groups:
                    param_group['lr'] = new_lrate

                if iteration % 200 == 0:
                    crf_savedir = os.path.join(dataset.model_path, "CRF")
                    draw_CRF_xLinear_tbwriter_add_figure(tb_writer, crf_savedir,input_exps_i, iteration, hdr_nerf_crf, x_lim=1.0,y_lim=args.crf_y_lim,crf_x_domain=args.crf_x_domain)

        # time.sleep(0.15)
        torch.cuda.empty_cache()


def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer: Optional[SummaryWriter],
    iteration: int,
    first_iter: int,
    input_exps_i: float,
    loss_dict: Dict[str, torch.Tensor],
    elapsed: float,
    testing_iterations: List[int],
    scene: Scene,
    light: CubemapLight,
    brdf_lut: torch.Tensor,
    canonical_rays: torch.Tensor,
    pbr_iteration: int,
    metallic: bool,
    tone: bool,
    gamma: bool,
    radius: float,
    bias: float,
    thick: float,
    delta: float,
    step: int,
    start: int,
    renderArgs: Tuple[GroupParams, torch.Tensor],
    indirect: bool = False,
    args: Namespace = None,
    hdr_nerf_crf: HDR_NeRF_CRF = None,
    color_mask: torch.Tensor= None,
    event_threshold: float= 0.2
) -> None:
    if tb_writer:
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(f"input_exps", input_exps_i, iteration)  # add 250331

        for k, v in loss_dict.items():
            tb_writer.add_scalar(f"train_loss/{k}", v, iteration)


    # Report test and samples of training set
    # print(f"==>> testing_iterations: {testing_iterations}") # [100, 300, 500, 1000, 2000, 3000, 7000, 10000, 15000, 20000, 25000, 30000, 30100, 30500, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 40000]
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test",
             "cameras": [
                scene.getTestCameras()[idx % len(scene.getTrainCameras())] for idx in range(80, 100, 1)  # (40, 60, 1)
                ],
            },
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 200, 10)   #! range(5, 30, 5)
                ],
            },
        )
        lpips_fn = LPIPS(net="vgg").cuda()
        pipe, background = renderArgs
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                # l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                # l1_test_norImg = 0.0
                psnr_test_norImg = 0.0
                ssim_test_norImg = 0.0
                lpips_test_norImg = 0.0

                psnr_test_norImg_enhc = 0.0
                ssim_test_norImg_enhc = 0.0
                lpips_test_norImg_enhc = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    viewpoint: Camera
                for idx in range(len(config["cameras"])):
                    viewpoint = config["cameras"][idx]

                    if idx == (len(config["cameras"]) -1):
                        last_view=True
                    else:
                        last_view=False
                    # 在这里使用 viewpoint 和 viewpoint_1
                    render_result = render(
                        viewpoint_camera=viewpoint,
                        pc=scene.gaussians,
                        pipe=pipe,
                        bg_color=background,
                        inference=True,
                        pad_normal=False,
                        derive_normal=True,
                        radius=radius,
                        bias=bias,
                        thick=thick,
                        delta=delta,
                        step=step,
                        start=start)


                    render_image = torch.clamp(render_result["render"], 0.0, 1.0)


                    if not last_view:
                        viewpoint_1 = config["cameras"][idx + 1]
                        render_result_1 = render(
                            viewpoint_camera=viewpoint_1,
                            pc=scene.gaussians,
                            pipe=pipe,
                            bg_color=background,
                            inference=True,
                            pad_normal=False,
                            derive_normal=True,
                            radius=radius,
                            bias=bias,
                            thick=thick,
                            delta=delta,
                            step=step,
                            start=start)
                        render_image_1 = torch.clamp(render_result_1["render"], 0.0, 1.0)
                    depth_img = (
                        torch.from_numpy(
                            turbo_cmap(render_result["depth_map"].cpu().numpy().squeeze())
                        )
                        .to(render_image.device)
                        .permute(2, 0, 1)
                    )
                    normal_map_from_depth = render_result["normal_map_from_depth"]
                    normal_map = render_result["normal_map"]
                    normal_img = torch.cat([normal_map, normal_map_from_depth], dim=-1)
                    # 250308
                    llnerf_r_albedo_n3_map  = render_result["llnerf_r_albedo_n3_map"]  # [3, H, W]
                    llnerf_v_viewdep_n1_map = render_result["llnerf_v_viewdep_n1_map"]  # [1, H, W]
                    llnerf_alpha_n1_map  = render_result["llnerf_alpha_n1_map"]  # [3, H, W]
                    llnerf_gamma_n3_map = render_result["llnerf_gamma_n3_map"]  # [1, H, W]


                    if args.which_img_gt=='norImg':
                        gt_image = viewpoint.original_image_norImg[0:3, :, :].cuda()
                    elif args.which_img_gt=='lowImg':
                        gt_image = viewpoint.original_image[0:3, :, :].cuda()

                    gt_image_norImg = viewpoint.original_image_norImg[0:3, :, :].cuda()

                    alpha_mask = viewpoint.gt_alpha_mask.cuda()
                    gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)
                    albedo_map = render_result["albedo_map"]  # [3, H, W]
                    roughness_map = render_result["roughness_map"]  # [1, H, W]
                    metallic_map = render_result["metallic_map"]  # [1, H, W]
                    out_normal_view = render_result["out_normal_view"]
                    depth_pos = render_result["depth_pos"]
                    normal_mask = render_result["normal_mask"]  # [1, H, W]


                    # NOTE: PBR record
                    if iteration <= pbr_iteration:
                        zero_pad = torch.zeros_like(render_image)
                        render_retinex = zero_pad
                        # render_retinex = torch.cat([zero_pad, zero_pad, zero_pad], dim=2)  # [3, H, 3W]
                    else:
                        LOGSECONDSAGE=None
                        render_retinex = pbr_shading_retinex(llnerf_r_albedo_n3_map, llnerf_v_viewdep_n1_map)["render_retinex"]
                        render_retinex = render_retinex
                        # pbr_image = torch.cat(
                        #     [render_retinex, render_retinex], dim=2
                        # )

                        if args.llnerf_enhc == 'enhc1':
                            L = llnerf_v_viewdep_n1_map
                            R = llnerf_r_albedo_n3_map
                            L_enhc = get_v_n3_enhanced(L,llnerf_alpha_n1_map,llnerf_gamma_n3_map)
                            llnerf_v_viewdep_n1_map_enhc = L_enhc
                            render_retinex_enhc =  R * L_enhc    # rgb_enhc  # add 250311

                    if tb_writer and (idx < 11):  #  40,41,...50, 共11个，   ori: 5
                        # if iteration == testing_iterations[0]:
                        if iteration == testing_iterations[0] or iteration == testing_iterations[-1] or iteration == 30000:  #iteration == first_iter:
                            #! 0_gt
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/0a_gt",
                                (gainVis(gt_image,args.tonemap_gain))[None],        # 250403 gt_image 是 RGB 通道
                                global_step=iteration,
                            )

                            #! 0_norImg
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/0b_norImg",
                                (gt_image_norImg)[None],
                                global_step=iteration,
                            )

                            if not last_view:
                                event_list = [viewpoint.event]
                                event_list.append(viewpoint_1.event)

                                #! 0c_ev_gt
                                # event_list = event_list[0:-1]
                                if args.ev_align_pre_or_post =='post':
                                    event_list = event_list[1:]
                                elif args.ev_align_pre_or_post=='pre':
                                    event_list = event_list[:-1]
                                event_gt_1hw = torch.sum(torch.stack(event_list), dim=0, keepdim=True).cuda() ## event: ([1, 352, 480])    # 没有event file 是置0？
                                stack_grid_colormap_ev_gt = np.transpose(stack_grid_to_cmap(event_gt_1hw.cpu().numpy(), vmax=2), (0, 3, 1, 2))   # ([1, 3, 352, 480])
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/0c_ev_gt",
                                    stack_grid_colormap_ev_gt,        # 250403 gt_image 是 RGB 通道
                                    global_step=iteration,
                                )

                        gamma_ot=args.gamma_ot
                        # gamma_ot=2.2
                        #! 1_render
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/1b_render",
                            (linear_to_sRGB_y(render_image,gamma=gamma_ot))[None],
                            global_step=iteration,
                        )

                        #! 1b_render_image_sRGB
                        with torch.no_grad():
                            render_image_sRGB = hdr_nerf_crf(gamma_brten_before_crf_y(render_image))

                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/1a_render_image_sRGB_brighten",
                                (gainVis(render_image_sRGB,args.tonemap_gain))[None],
                                global_step=iteration,
                            )


                        #! 1c_ev_pre
                        if not last_view:
                            diff_image = torch.log(render_image_1 + 1e-8) - torch.log(render_image + 1e-8)  # {diff_image.shape}") # ([3, 352, 480]) 第一个 3 是RGB 3通道
                            valid_color = diff_image * color_mask  # valid_color: [3, H, W]  ## diff_image: ([3, 352, 480]) 3 是3通道
                            diff_bayer_1HW = valid_color.sum(dim=0, keepdim=True)  # diff_bayer_1HW: [1, H, W]
                            event_pre_1hw = diff_bayer_1HW / event_threshold ## event_pre_1hw: ([1, 352, 480])
                            stack_grid_colormap_ev_pre = np.transpose(stack_grid_to_cmap(event_pre_1hw.cpu().numpy(), vmax=2), (0, 3, 1, 2))  # ([1, 3, 352, 480]) # # numpy用 np.transpose;   torch 用 .permute(0, 3, 1, 2)  1,3,H,W
                            tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/1c_ev_pre",
                                                stack_grid_colormap_ev_pre,
                                                iteration)

                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/depth",
                            (depth_img)[None],
                            global_step=iteration,
                        )
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/normal",
                            (resize_tensorboard_img(normal_img, 1600)[None] + 1.0) / 2.0,
                            global_step=iteration,
                        )
                        if iteration > pbr_iteration:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/2a_render_retinex",
                                (linear_to_sRGB_y(render_retinex,gamma=gamma_ot))[None],
                                global_step=iteration,
                            )

                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/2b_retinex_r_n3",
                                (llnerf_r_albedo_n3_map)[None],
                                global_step=iteration,
                            )
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/2c_retinex_v_n1",
                                (torch.tile(llnerf_v_viewdep_n1_map, (3, 1, 1)))[None],
                                global_step=iteration,
                            )

                            if args.llnerf_enhc == 'enhc1':
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/3_render_retinex_enhc", #3hw
                                    (render_retinex_enhc)[None],
                                    global_step=iteration,
                                )

                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/b2_retinex_v_n1_enhc",                 # 3h2w
                                    (llnerf_v_viewdep_n1_map_enhc)[None],
                                    global_step=iteration,
                                )
                                tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/b3_retinex_alpha_n1",
                                (llnerf_alpha_n1_map)[None],
                                global_step=iteration,
                                )
                                tb_writer.add_images(
                                    f"{config['name']}_view_{viewpoint.image_name}/b4_retinex_gamma_n3",
                                    (llnerf_gamma_n3_map)[None],
                                    global_step=iteration,
                                )

                    if iteration <= pbr_iteration:
                        # l1_test += F.l1_loss(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image).mean().double()
                        psnr_test += psnr(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image).mean().double()
                        ssim_test += ssim(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image).mean().double()
                        lpips_test += lpips_fn(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image).mean().double()

                        # l1_test_norImg += F.l1_loss(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image_norImg).mean().double()
                        psnr_test_norImg += psnr(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image_norImg).mean().double()
                        ssim_test_norImg += ssim(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image_norImg).mean().double()
                        lpips_test_norImg += lpips_fn(linear_to_sRGB_y(render_image,gamma=gamma_ot), gt_image_norImg).mean().double()

                    else:
                        # l1_test += F.l1_loss(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image).mean().double()
                        psnr_test += psnr(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image).mean().double()
                        ssim_test += ssim(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image).mean().double()
                        lpips_test += lpips_fn(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image).mean().double()

                        # l1_test_norImg += F.l1_loss(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image_norImg).mean().double()
                        psnr_test_norImg += psnr(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image_norImg).mean().double()
                        ssim_test_norImg += ssim(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image_norImg).mean().double()
                        lpips_test_norImg += lpips_fn(linear_to_sRGB_y(render_retinex,gamma=gamma_ot), gt_image_norImg).mean().double()

                        if args.llnerf_enhc == 'enhc1':
                            # l1_test_norImg_enhc += F.l1_loss(render_retinex, gt_image_norImg).mean().double()
                            psnr_test_norImg_enhc += psnr(render_retinex_enhc, gt_image_norImg).mean().double()
                            ssim_test_norImg_enhc += ssim(render_retinex_enhc, gt_image_norImg).mean().double()
                            lpips_test_norImg_enhc += lpips_fn(render_retinex_enhc, gt_image_norImg).mean().double()


                # l1_test /= len(config["cameras"])
                psnr_test  /= len(config["cameras"])
                ssim_test  /= len(config["cameras"])
                lpips_test /= len(config["cameras"])

                # l1_test_norImg /= len(config["cameras"])
                psnr_test_norImg  /= len(config["cameras"])
                ssim_test_norImg  /= len(config["cameras"])
                lpips_test_norImg /= len(config["cameras"])

                if args.llnerf_enhc == 'enhc1':
                    psnr_test_norImg_enhc  /= len(config["cameras"])
                    ssim_test_norImg_enhc  /= len(config["cameras"])
                    lpips_test_norImg_enhc /= len(config["cameras"])

                print(len(config["cameras"]))
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: PSNR {psnr_test:.3f} SSIM {ssim_test:.5f} LPIPS {lpips_test:.5f}")
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: PSNR_norImg {psnr_test_norImg:.3f} SSIM_norImg {ssim_test_norImg:.5f} LPIPS_norImg {lpips_test_norImg:.5f}")
                if args.llnerf_enhc == 'enhc1':
                    print(f"\n[ITER {iteration}] Evaluating {config['name']}: PSNR_norImg_enhc {psnr_test_norImg_enhc:.3f} SSIM_norImg_enhc {ssim_test_norImg_enhc:.5f} LPIPS_norImg_enhc {lpips_test_norImg_enhc:.5f}")

                if tb_writer:
                    # tb_writer.add_scalar(config["name"] + "_mtrc/l1Err", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/psnr", psnr_test, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/ssim", ssim_test, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/lpips", lpips_test, iteration)

                    # tb_writer.add_scalar(config["name"] + "_mtrc/l1Err_norImg", l1_test_norImg, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/psnr_norImg", psnr_test_norImg, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/ssim_norImg", ssim_test_norImg, iteration)
                    tb_writer.add_scalar(config["name"] + "_mtrc/lpips_norImg", lpips_test, iteration)

                    if args.llnerf_enhc == 'enhc1':
                        tb_writer.add_scalar(config["name"] + "_mtrc/psnr_norImg_enhc", psnr_test_norImg_enhc, iteration)
                        tb_writer.add_scalar(config["name"] + "_mtrc/ssim_norImg_enhc", ssim_test_norImg_enhc, iteration)
                        tb_writer.add_scalar(config["name"] + "_mtrc/lpips_norImg_enhc", lpips_test_norImg_enhc, iteration)

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity.reshape(-1), iteration
            )
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[100,300,500,1000,2000,3000,4000,5000,6000,7000,10000,15000,20000,25000,30000,30100,30500,31000,32000,33000,34000,35000,36000,37000,38000,39000,40000] # 250326: [100, 200, 300, 500, 1_000, 2_000, 3_000, 7_000, 30000, 32_000, 36000, 40000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[40000]     # 250326:[100, 200, 300, 500, 1_000, 2_000, 3_000, 7_000, 30000, 32_000, 36000, 40000],
    )
    # parser.add_argument("--args_data_type_synthsis_or_real", type=str, default='synthesis',help="synthesis,  real")
    #! ============== synthetic 同 real data 的 parser ==============
    parser.add_argument("--signal_supervise",  type=str, default='img', help="img or ev or img_ev")
    parser.add_argument("--which_img_gt",  type=str, default='lowImg', help="")
    parser.add_argument("--args_ev_loss", type=str, default='ev_loss_l1',help="ev_loss_l1,  ev_loss_weight_dark")
    parser.add_argument("--parser_background", type=str, default='White',help="White Black")
    parser.add_argument("--HDR_NeRF_CRF_split", type=str, default='True',help="True, False")
    # ============== synthetic 同 real data 的 parser ==============


    #! ============== synthetic 不同于 real data 的 parser ==============
    parser.add_argument("--ev_skip_max", default=12, type=int, help="1 for synthsis, 12 for real")

    parser.add_argument("--args_no_ev_loss", type=str, default=None,help="no_ev_loss")
    parser.add_argument("--lambda_no_ev_loss", default=0.0, type=float, help="")
    parser.add_argument("--args_vs_loss", type=str, default=None,help="None, vs_loss")  #  no_vs_loss
    parser.add_argument("--lambda_vs_loss", default=0.0, type=float, help="")
    parser.add_argument("--args_chromaticity_loss", type=str, default=None,help="None, chromaticity_loss") # no_chromaticity_loss

    parser.add_argument("--gamma_ot", type=float, default=2.2, help="2.2 or 1.0")

    parser.add_argument("--crf_y_lim", type=float, default=0.5, help="0.5 or 1.0")
    parser.add_argument("--crf_x_domain", type=str, default='linear',help="linear, log")
    # ============== synthetic 不同于 real data 的 parser ==============


    #! ============== synthetic, real 都要设置的 parser ==============
    parser.add_argument("--args_data_loss", type=str, default='l1_loss',help="RawNeRF_loss,  log_data_loss, img_loss_weight_brht")
    # 仅用 lambda 来控制 loss 是否起作用，默认为0
    parser.add_argument("--args_brightness_loss", type=str, default=None, help="None, brightness_mean_pixel_loss,  brightness_mean_gray_loss")
    parser.add_argument("--target_brightness", default=0.4, type=float, help="")
    parser.add_argument("--lambda_brightness_loss", default=0.0, type=float, help="")   #! brightness_loss

    parser.add_argument("--lambda_img_loss", default=0.0, type=float, help="")
    parser.add_argument("--lambda_img_loss_ssim", default=0.0, type=float, help="")
    parser.add_argument("--lambda_ev_loss", default=0.0, type=float, help="")
    parser.add_argument("--lambda_ev_loss_ssim", default=0.0, type=float, help="")

    parser.add_argument("--lambda_retinex_render_loss", default=0.0, type=float, help="")               #! img
    parser.add_argument("--lambda_retinex_render_loss_pseudo", default=0.0, type=float, help="")        #! img
    parser.add_argument("--lambda_retinex_render_ev_loss", default=0.0, type=float, help="")
    parser.add_argument("--lambda_retinex_render_ev_loss_ssim", default=0.0, type=float, help="")
    # ============== synthetic, real 都要设置的 parser ==============




    #! ============== unused parser ==============

    parser.add_argument("--args_color_const_loss", type=str, default=None,help="no_color_const_loss, color_const_loss, llnerf_gray_loss")
    parser.add_argument("--lambda_color_const_loss", default=0.0, type=float, help="") # 1.0

    parser.add_argument("--args_crf_loss", type=str, default=None,help="crf_loss")
    parser.add_argument("--lambda_loss_crf_zero", default=0.0, type=float, help="")
    parser.add_argument("--lambda_loss_crf_grad", default=0.0, type=float, help="")

    parser.add_argument("--args_ev_masked_tv_loss", type=str, default=None,help="ev_masked_tv_loss")
    parser.add_argument("--lambda_ev_masked_tv_loss_R", default=0.0, type=float, help="")  # 0.1
    parser.add_argument("--lambda_ev_masked_tv_loss_L", default=0.0, type=float, help="")  # 0.1

    parser.add_argument("--args_dark_evssim_loss", type=str, default=None,help="None, dark_evssim_loss") # no_dark_evssim_loss,
    parser.add_argument("--args_cosine_color_loss", type=str, default=None,help="None, cosine_color_loss") # no_cosine_color_loss


    parser.add_argument("--llnerf_enhc",  type=str, default=None, help="None, enhc1:llnerf , enhc2: alpha(beta*x)**gamma")

    #  ============== unused parser ==============


    parser.add_argument("--tonemap_gain",  type=int, default=-1, help="-1: normal light,   1,2,3: ")
    parser.add_argument("--enhc_loss_list", type=str, nargs='+', help="loss_exp loss_gray loss_Gltv loss_Altv loss_gamma_norm")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--ev_align_pre_or_post", type=str, default='post',
                        help="post: 000001.npy contains events between images 000000.png and 000001.png  or  pre: 000000.npy contains events between images 000000.png and 000001.png.")
    parser.add_argument("--start_checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr_iteration", default=30_000, type=int, help="The iteration to begin the pb.r learning (Deomposition Stage in the paper)")
    parser.add_argument("--normal_tv", default=5.0, type=float, help="The weight of TV loss on predicted normal map.")
    parser.add_argument("--brdf_tv", default=1.0, type=float, help="The weight of TV loss on predicted BRDF (material) map.")
    parser.add_argument("--env_tv", default=0.01, type=float, help="The weight of TV loss on Environment Map.")
    parser.add_argument("--radius", default=0.8, type=float, help="Path tracing range")
    parser.add_argument("--bias", default=0.01, type=float, help="ensure hit the surface")
    parser.add_argument("--thick", default=0.05, type=float, help="thickness of the surface")
    parser.add_argument("--delta", default=0.0625, type=float, help="angle interval to control the num-sample")
    parser.add_argument("--step", default=16, type=int, help="Path tracing steps")
    parser.add_argument("--start", default=64, type=int, help="Path tracing starting point")    # 250326: 8
    parser.add_argument("--degree", default=1, type=int, help="sh_degree")                      # 250326: 3
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"==>> args.images: {args.images}")
    # Start GUI server, configure and run training
    # with torch.autograd.detect_anomaly():
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    dataset.sh_degree = args.degree
    training(
        dataset=dataset,
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        signal_supervise=args.signal_supervise,
        ev_align_pre_or_post=args.ev_align_pre_or_post,
        checkpoint_path=args.start_checkpoint,
        pbr_iteration=args.pbr_iteration,
        debug_from=args.debug_from,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        normal_tv_weight=args.normal_tv,
        brdf_tv_weight=args.brdf_tv,
        env_tv_weight=args.env_tv,
        radius=args.radius,
        bias=args.bias,
        thick=args.thick,
        delta=args.delta,
        step=args.step,
        start=args.start,
        indirect=args.indirect,
        args=args
    )

    # All done
    print("\nTraining complete.")
