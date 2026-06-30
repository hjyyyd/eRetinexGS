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
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from arguments import GroupParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
import kornia
from utils.sh_utils import eval_sh
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def g(*args, **kwargs):  # debug no-op (removed private dmfq dep)
    return None



def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: GroupParams,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
    inference: bool = False,
    pad_normal: bool = False,
    derive_normal: bool = False,
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8
) -> Dict:
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        radius=radius,
        bias=bias,
        thick=thick,
        delta=delta,
        step=step,
        start=start,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        inference=inference,
        argmax_depth=False,
    )

    # entrance
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # normal = pc.get_normal(viewpoint_camera)
    # normal = pc.get_rot_normal(viewpoint_camera)
    normal = pc.get_normal
    albedo = pc.get_albedo
    roughness = pc.get_roughness
    roughness_alpha = pc.get_roughness_alpha
    albedo_gamma = pc.get_albedo_gamma   # albedo_gamma
    metallic = pc.get_metallic
    assert albedo.shape[0] == roughness.shape[0] and albedo.shape[0] == metallic.shape[0]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:   #* False   # 
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)  # 根据视角方向计算SH基函数值
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # v_precomp
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color  
    
    """
    pc.get_features_v
    pc.max_sh_degree_v 
    pc.active_sh_degree_v
    """
    # 
    shs_view_v = pc.get_features_v.transpose(1, 2).view(-1, 3, (pc.max_sh_degree_v + 1) ** 2)
    dir_pp_v = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features_v.shape[0], 1)
    dir_pp_normalized_v = dir_pp_v / dir_pp_v.norm(dim=1, keepdim=True)
    sh2rgb_v = eval_sh(pc.active_sh_degree_v, shs_view_v, dir_pp_normalized_v)  # 根据视角方向计算SH基函数值
    colors_precomp_v = torch.clamp_min(sh2rgb_v + 0.5, 0.0)  # v_precomp
    
    shs_v = None
    # print(f"==>> shs: {shs}")       # not None
    # print(f"==>> shs_v: {shs_v}")   # None
    colors_precomp_v = colors_precomp_v[:,:1] # torch.Size([6276, 3])
    # print(f"==>> colors_precomp_v: {colors_precomp_v}")   #  not None
    # print(f"==>> colors_precomp_v.shape: {colors_precomp_v.shape}")
    # print(f"==>> roughness.shape: {roughness.shape}")
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    #! d224a
    if torch.isnan(albedo).any():
            print("albedo is nan")   # 这里 会出现NaN
    (
        rendered_image,  # _V
        radii,
        opacity_map,
        depth_map,
        normal_map_from_depth,
        normal_map,
        occlusion_map,
        albedo_map,
        roughness_map_v,               #* changed from roughness_map
        roughness_map_alpha,        #*add
        albedo_map_gamma,           #*add
        metallic_map,
        out_normal_view,
        depth_pos
    ) = rasterizer(      
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        normal=normal,
        shs=shs,                        # no changed
        colors_precomp=colors_precomp,  # no changed
        albedo=albedo,
        roughness=colors_precomp_v,   #* changed from roughness,
        roughness_alpha=roughness_alpha, #*add
        albedo_gamma=albedo_gamma, #*add
        metallic=metallic,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        derive_normal=derive_normal)
    roughness_map = torch.zeros_like(metallic_map) #add

    #!d228a
    # (
    #     rendered_image,  # _V
    #     radii,
    #     opacity_map,
    #     depth_map,
    #     normal_map_from_depth,
    #     normal_map,
    #     occlusion_map,
    #     albedo_map,
    #     roughness_map,          #* changed from roughness_map
    #     metallic_map,
    #     out_normal_view,
    #     depth_pos
    # ) = rasterizer(      
    #     means3D=means3D,
    #     means2D=means2D,
    #     opacities=opacity,
    #     normal=normal,
    #     shs=shs,         # 
    #     colors_precomp=colors_precomp, # 
    #     albedo=albedo,
    #     roughness=roughness,   #* changed from roughness,
    #     metallic=metallic,
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=cov3D_precomp,
    #     derive_normal=derive_normal)
    normal_from_depth_mask = (normal_map_from_depth != 0).all(0)
    normal_mask = (normal_map != 0).all(0, keepdim=True)
    if pad_normal:
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map < 0.004,
            torch.zeros_like(opacity_map),
            opacity_map,
        )
        opacity_map = torch.where(  # NOTE: a trick to filter out 1 / 255
            opacity_map > 1.0 - 0.004,
            torch.ones_like(opacity_map),
            opacity_map,
        )
        normal_bg = torch.tensor([0.0, 0.0, 1.0], device=normal_map.device)
        normal_map = normal_map * opacity_map + (1.0 - opacity_map) * normal_bg[:, None, None]
        mask_from_depth = (normal_map_from_depth == 0.0).all(0, keepdim=True).float()
        normal_map_from_depth = normal_map_from_depth * (1.0 - mask_from_depth) + mask_from_depth * normal_bg[:, None, None]

    normal_map_from_depth = torch.where(
        torch.norm(normal_map_from_depth, dim=0, keepdim=True) > 0,
        F.normalize(normal_map_from_depth, dim=0, p=2),
        normal_map_from_depth,
    )
    normal_map = torch.where(
        torch.norm(normal_map, dim=0, keepdim=True) > 0,
        F.normalize(normal_map, dim=0, p=2),
        normal_map,
    )

    normal_map = kornia.filters.median_blur(normal_map[None, ...], (3, 3))[0]

    out_normal_view = torch.where(
        torch.norm(out_normal_view, dim=0, keepdim=True) > 0,
        F.normalize(out_normal_view, dim=0, p=2),
        out_normal_view,
    )

    out_normal_view = kornia.filters.median_blur(out_normal_view[None, ...], (3, 3))[0]




    # add 250222
    llnerf_r_albedo_n3_map = albedo_map # [3,H,W] ([3, 770, 1155])
    llnerf_r_albedo_n3_map = torch.clamp(llnerf_r_albedo_n3_map,0.0,1.0)  # 250326 加入之后会 解决NaN
    if torch.isnan(llnerf_r_albedo_n3_map).any():
        print("llnerf_r_albedo_n3_map is nan")   # 这里 会出现NaN
    llnerf_v_viewdep_n1_map = roughness_map_v   #todo!!!!! # ([1, 770, 1155])
    llnerf_v_viewdep_n1_map = torch.clamp(llnerf_v_viewdep_n1_map,0.0,1.0)
    # print(f"==>> llnerf_r_albedo_n3_map.shape: {llnerf_r_albedo_n3_map.shape}") 
    # print(f"==>> llnerf_v_viewdep_n1_map.shape: {llnerf_v_viewdep_n1_map.shape}")
    
    llnerf_alpha_n1_map = roughness_map_alpha
    llnerf_gamma_n3_map = albedo_map_gamma
    llnerf_alpha_n1_map = torch.clamp(llnerf_alpha_n1_map,0.0, 0.5)  #todo hyparam
    llnerf_gamma_n3_map = torch.clamp(llnerf_gamma_n3_map,0.0,1.0)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,                       # rendering_result["render"]对应valinna 3DGS直出的结果
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity_map": opacity_map,
        "depth_map": depth_map,
        "normal_map_from_depth": normal_map_from_depth,
        "normal_from_depth_mask": normal_from_depth_mask,
        "normal_map": normal_map,
        "normal_mask": normal_mask,
        "albedo_map": albedo_map,
        "llnerf_r_albedo_n3_map":llnerf_r_albedo_n3_map,
        "llnerf_v_viewdep_n1_map":llnerf_v_viewdep_n1_map,
        "llnerf_alpha_n1_map": llnerf_alpha_n1_map,
        "llnerf_gamma_n3_map": llnerf_gamma_n3_map,
        "roughness_map": roughness_map,
        "metallic_map": metallic_map,
        "occlusion_map": occlusion_map,
        "out_normal_view": out_normal_view,
        "depth_pos": depth_pos
    }
