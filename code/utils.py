import torch
import numpy as np
import cv2
import torchvision
from pbr.shade import rgb_to_srgb, srgb_to_rgb
def g(*args, **kwargs):  # debug no-op (removed private dmfq dep)
    return None
from typing import Dict, List, Optional, Tuple, Union

import os
from datetime import datetime
import pandas as pd
import argparse




def get_err_map(img1_BGR_01,img2_BGR_01,vmin=0.0, vmax=0.4, colormap='COLORMAP_VIRIDIS'):
    '''
    '''
    # print(f"==>> img2_BGR_01.shape: {img2_BGR_01.shape}")
    # print(f"==>> img1_BGR_01.shape: {img1_BGR_01.shape}")
    error_hw3 = np.abs(img1_BGR_01 - img2_BGR_01)
    error_hw = np.mean(error_hw3, axis=2)
    errmap_hw3_BGR255  = get_err_color_map(error_hw,vmin=vmin, vmax=vmax, colormap=colormap)
    return errmap_hw3_BGR255

def get_err_color_map(error_hw_01,vmin=0.0, vmax=1.0, colormap='COLORMAP_VIRIDIS'): # vmin, vmax 有用
    '''
    old name: get_color_map
    y 20240712 经过测试，这种方法, 速度不受影响
    param:
        error_hw: h,w      0.-1. float32
    return:
        errmap_hw3_BGR255: h,w,3    0-255 uint8   BGR

    example:
        real255=cv2.imread(pngs_path_list_gt[i])
        fake255=cv2.imread(pngs_path_list_pre1[i])
        error_hw3 = np.abs(real255 / 255.0 - fake255 / 255.0)
        error_hw = np.mean(error_hw3, axis=2)
        erro255  = get_color_map(error_hw,vmin=vmin, vmax=vmax, colormap=colormap)
    '''
    assert vmax>vmin
    error_hw_01 = np.clip(error_hw_01, vmin, vmax)  # 将错误矩阵限制在 [vmin, vmax] 区间
    error_hw_01 = (error_hw_01-vmin)/(vmax-vmin)  # 将错误矩阵归一化到 [0, 1] 区间
    error_hw_255 = np.clip(error_hw_01*255,0,255).astype(np.uint8)   # 生成一个线性的错误矩阵
    # print(f"==>> error_hw.shape: {error_hw.shape}")

    # 将灰度图像转换为Jet样式的伪彩色图像
    if colormap=='COLORMAP_VIRIDIS':
        error_hw_virids = cv2.applyColorMap(error_hw_255, cv2.COLORMAP_VIRIDIS)  # 颜色映射
    elif colormap=='COLORMAP_JET':
        error_hw_virids = cv2.applyColorMap(error_hw_255, cv2.COLORMAP_JET)
    elif colormap=='COLORMAP_HOT':
        error_hw_virids = cv2.applyColorMap(error_hw_255, cv2.COLORMAP_HOT)
    errmap_hw3_BGR255 = error_hw_virids
    # print(f"==>> errmap_hw3_BGR255.shape: {errmap_hw3_BGR255.shape}")
    # print(f"==>> errmap_hw3_BGR255.dtype: {errmap_hw3_BGR255.dtype}")
    return errmap_hw3_BGR255 # hw3 255 BGR



def get_err_colorbar(colorbar_h, colorbar_w=80, bar_inner_w=20, bar_inner_h_ratio=0.8,
                 bar_x_offset=None, bar_x_offset_ratio=0.2, vmin=0.0, vmax=1.0, num_ticks=5, colormap='COLORMAP_VIRIDIS'):
    """
    生成颜色条（可单独使用）

    :param colorbar_h: 颜色条高度（应与误差图 h 一致）
    :param colorbar_w: 整个颜色条区域的宽度（包括白色背景 & 数值标注）
    :param bar_inner_w: 仅颜色条本身的宽度
    :param bar_inner_height_ratio: 颜色条本身的高度比例（相对于 h ，默认 0.8）
    :param bar_x_offset: 颜色条距离左侧的偏移量（若 None，则默认居中）
    :param bar_x_offset_ratio: 颜色条距离左侧的偏移量。bar_x_offset为None时，根据ratio设置。 如果ratio也为 None ，则默认居中）
    :param vmin: 颜色条最小值
    :param vmax: 颜色条最大值
    :param num_ticks: 颜色条上的刻度数量
    :return: (colorbar_h, colorbar_w, 3) 颜色条（BGR 格式）
    """
    assert bar_inner_w < colorbar_w, "bar_inner_w 应小于 colorbar_w"
    assert 0 < bar_inner_h_ratio <= 1, "bar_inner_height_ratio 应在 (0,1] 之间"

    colorbar_w = colorbar_w  # 颜色条总宽度
    # 计算颜色条的高度
    bar_inner_h = int(colorbar_h * bar_inner_h_ratio)
    bar_inner_w = bar_inner_w  # 仅颜色条的宽度

    # 创建白色背景
    colorbar_hw3_BGR = np.ones((colorbar_h, colorbar_w, 3), dtype=np.uint8) * 255  # 全白背景

    # 颜色条的垂直起始位置（确保在图像中居中）
    y_offset = (colorbar_h - bar_inner_h) // 2


    # 颜色条的水平起始位置（根据 bar_x_offset 调整）
    if bar_x_offset is None:
        if bar_x_offset_ratio:
            x_offset = int(colorbar_w * bar_x_offset_ratio)
        else:
            x_offset = (colorbar_w - bar_inner_w) // 2  # 默认让bar 中轴线 居中
    else:
        x_offset = min(max(0, bar_x_offset), colorbar_w - bar_inner_w)  # 限制范围，防止超出边界

    # 生成颜色条的渐变
    gradient = np.linspace(1, 0, bar_inner_h).reshape(bar_inner_h, 1)
    gradient = (gradient * 255).astype(np.uint8)

    colormap_dict = {
        'COLORMAP_VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'COLORMAP_JET': cv2.COLORMAP_JET,
        'COLORMAP_HOT': cv2.COLORMAP_HOT
    }
    gradient_color = cv2.applyColorMap(gradient, colormap_dict.get(colormap, cv2.COLORMAP_VIRIDIS))

    # 将颜色条插入白色背景
    colorbar_hw3_BGR[y_offset:y_offset + bar_inner_h, x_offset:x_offset + bar_inner_w] = gradient_color

    # ========== 添加刻度标注 ==========
    tick_positions = np.linspace(y_offset, y_offset + bar_inner_h - 1, num_ticks).astype(int)  # 刻度位置
    tick_values = np.linspace(vmax, vmin, num_ticks)  # 误差数值

    for i, (pos, val) in enumerate(zip(tick_positions, tick_values)):
        text = f"{val:.2f}"  # 格式化数值
        text_position = (x_offset + bar_inner_w + 5, pos + 5)  # 数值放在颜色条右侧
        cv2.putText(colorbar_hw3_BGR, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return colorbar_hw3_BGR  # colorbar: [colorbar_h,colorbar_w,3]  255 BGR




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

LOSS_FUNC=None

# , no_ev_mask: torch.Tensor=None
def RawNeRF_loss_func(image: torch.Tensor, gt_image: torch.Tensor, clip_min: float = 0.0, use_mask: bool = False) -> torch.Tensor:
    '''
    adapted from:
    /home/hjy/ws/EvGS/0_ref/no_ev/multinerf_RawNeRF_CVPR24/internal/train_utils.py
    '''
    # Clip raw values against 1 to match sensor overexposure behavior.
    img_clip = torch.clamp(image, clip_min, 1.0)
    gt_image = torch.clamp(gt_image, clip_min, 1.0)

    # 计算残差平方误差
    resid_sq_clip = (img_clip - gt_image) ** 2
    # 梯度重加权
    scaling_grad = 1.0 / (1e-3 + img_clip.detach())

    if use_mask:
        # 生成 mask，仅在 gt_image < 0.1 的地方屏蔽损失
        mask = gt_image >= 0.01 #!509  0.01  # H, W, 3 的布尔张量
        # 应用 mask，仅计算有效区域的 loss

        # if no_ev_mask is not None:
        #     # mask 为 mask 为True 或者 no_ev_mask 为 True
        #     mask = mask | no_ev_mask.repeat(3, 1, 1).bool()

        masked_loss = (resid_sq_clip * (scaling_grad ** 2))[mask].mean()

        # if no_ev_mask is not None:
        #     # mask 为 mask 为True 或者 no_ev_mask 为 True
        #     # mask = mask | no_ev_mask.bool()
        #     masked_loss = 0.5*masked_loss + 0.5*(resid_sq_clip)[no_ev_mask.repeat(3, 1, 1).bool()].mean()

        if torch.isnan(masked_loss).any():
            print("RawNeRF_loss_func: masked_loss is nan")
        return masked_loss
    else:
        # 不使用 mask，计算所有区域的 loss
        unmasked_loss = (resid_sq_clip * (scaling_grad ** 2)).mean()
        if torch.isnan(unmasked_loss).any():
            print("RawNeRF_loss_func: unmasked_loss is nan")
        return unmasked_loss


def log_data_loss_func(image: torch.Tensor, gt_image: torch.Tensor) -> torch.Tensor:
    '''
    和 llnerf 中的文献接近。
    '''
    rtn = torch.abs((torch.log(image+1e-3) - torch.log(gt_image+1e-3))).mean()
    return rtn


def variation_similarity_loss_func(pre, gt, eps=1e-3):
    """
    pre: [3, H, W] - 预测图像
    gt:  [3, H, W] - Ground truth 图像
    """

    # 计算梯度：水平和垂直方向
    def gradient(x):
        dh = x[:, :, 1:] - x[:, :, :-1]  # [3, H, W-1]
        dv = x[:, 1:, :] - x[:, :-1, :]  # [3, H-1, W]
        return dh, dv

    pre_dh, pre_dv = gradient(pre)
    gt_dh, gt_dv = gradient(gt)

    # 结构引导的权重（每个位置上的 GT 梯度强度 + eps）
    weight_h = torch.sum(gt_dh ** 2, dim=0, keepdim=True) + eps  # [1, H, W-1]
    weight_v = torch.sum(gt_dv ** 2, dim=0, keepdim=True) + eps  # [1, H-1, W]

    # 梯度差归一化
    loss_h = torch.sum((pre_dh - gt_dh) ** 2, dim=0, keepdim=True) / weight_h
    loss_v = torch.sum((pre_dv - gt_dv) ** 2, dim=0, keepdim=True) / weight_v

    # 平均作为最终 loss
    return 0.5 * (loss_h.mean() + loss_v.mean())


def cosine_color_loss_func(pre, gt, eps=1e-6, threshold=0.004):
    """
    计算颜色余弦损失，仅在 GT 暗区域（亮度 < threshold）内进行。

    Args:
        pre: [3, H, W] - 预测图像，范围 0~1
        gt:  [3, H, W] - Ground truth 图像，范围 0~1
        eps: float - 防止除零
        threshold: float - 亮度阈值，默认0.004

    Returns:
        loss: scalar，暗区余弦颜色损失
    """
    assert pre.shape == gt.shape and pre.shape[0] == 3, "输入尺寸必须为 [3, H, W]"

    # 计算 luminance: Y = 0.299 * R + 0.587 * G + 0.114 * B
    luminance = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]  # [H, W]
    mask = (luminance < threshold)                             # [H, W], bool mask

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pre.device)  # 避免空 mask 的情况

    # 拉平成 [3, H*W]
    pre_vec = pre.view(3, -1)
    gt_vec  = gt.view(3, -1)
    mask_flat = mask.view(-1)

    # 仅保留 mask 中的像素
    pre_vec = pre_vec[:, mask_flat]
    gt_vec  = gt_vec[:, mask_flat]

    # 归一化
    pre_norm = pre_vec / (pre_vec.norm(dim=0, keepdim=True) + eps)
    gt_norm  = gt_vec  / (gt_vec.norm(dim=0, keepdim=True)  + eps)

    # 余弦相似度
    cos_sim = (pre_norm * gt_norm).sum(dim=0)  # [N]
    loss = 1 - cos_sim.mean()
    return loss

def chromaticity_loss_func(pre, gt, eps=1e-6, threshold=0.004):
    """
    计算预测图像和 GT 图像在暗区域的色度（chromaticity）L1损失。

    Args:
        pre: [3, H, W] - 预测图像
        gt:  [3, H, W] - Ground truth 图像
        eps: float - 防止除以0
        threshold: float - 暗区阈值（默认 0.004）

    Returns:
        loss: scalar，暗区色度差异损失
    """
    assert pre.shape == gt.shape and pre.shape[0] == 3, "输入尺寸必须为 [3, H, W]"

    # 计算 luminance
    luminance = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]  # [H, W]
    mask = (luminance < threshold)                             # [H, W], bool mask

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pre.device)

    # [3, H*W]
    pre_vec = pre.view(3, -1)
    gt_vec  = gt.view(3, -1)
    mask_flat = mask.view(-1)

    # 只选中 mask 区域
    pre_vec = pre_vec[:, mask_flat]
    gt_vec  = gt_vec[:, mask_flat]

    # 归一化到色度空间（除以亮度）
    pre_luminance = pre_vec.sum(dim=0, keepdim=True) + eps  # [1, N]
    gt_luminance  = gt_vec.sum(dim=0, keepdim=True) + eps   # [1, N]

    pre_chroma = pre_vec / pre_luminance  # [3, N]
    gt_chroma  = gt_vec / gt_luminance    # [3, N]

    # L1 loss
    loss = torch.abs(pre_chroma - gt_chroma).mean()
    return loss






def ltv_loss(L_e, L,  beta=1.5, alpha=2, eps=1e-4):
    """ adapted from /home/hjy/ws/EvGS/0_ref/no_ev/LLNeRF/internal/train_utils.py

    原本的代码 L_e 可能是 (batch_size, num_rays, num_samples, 3) 对应 NeRF 采样光线的背景
    这里 L_e 得到 相邻 pixel, 要重新写一下

    改完之后
    L_e: 3HW or 1HW
    L:   3HW or 1HW
    """
    # # get the gray scale image of L_enhanced and split it.
    # assert rendering['L_enhanced'].shape[0] % 3 == 0 and config.sample_neighbor_num > 0
    # L_chunks = jnp.array_split(rendering['L'].mean(axis=-1), 3)
    # Le_chunks = jnp.array_split(rendering['L_enhanced'].mean(axis=-1), 3)
    # pix_L, right_pix_L, down_pix_L = L_chunks
    # pix_Le, right_pix_Le, down_pix_Le = Le_chunks

    hyper_sample_neighbor_num = 4
    # 确保输入形状和条件
    # print(f"==>> L_e.shape: {L_e.shape}")       # ([3, 770, 1156]) 或者 ([1, 770, 1156])
    # assert L_e.shape[1] == 3 and hyper_sample_neighbor_num > 0
    assert hyper_sample_neighbor_num > 0

    # 对 L 取对数
    L = torch.log(L + eps)

    # # 提取像素值及其相邻像素值
    pix_L, right_pix_L, down_pix_L = L[:, 0, ...], L[:, 1, ...], L[:, 2, ...]
    pix_Le, right_pix_Le, down_pix_Le = L_e[:, 0, ...], L_e[:, 1, ...], L_e[:, 2, ...]

    #* powered by deepseek_v3 提取像素值及其相邻像素值
    # 提取当前像素的 RGB 值（裁剪边界以对齐右侧和下方像素）
    pix_L = L[:, :-1, :-1]  # 形状: (3, H-1, W-1)
    pix_Le = L_e[:, :-1, :-1]  # 形状: (3, H-1, W-1)
    # 提取右侧像素的 RGB 值
    right_pix_L = L[:, :-1, 1:]  # 形状: (3, H-1, W-1)
    right_pix_Le = L_e[:, :-1, 1:]  # 形状: (3, H-1, W-1)
    # 提取下方像素的 RGB 值
    down_pix_L = L[:, 1:, :-1]  # 形状: (3, H-1, W-1)
    down_pix_Le = L_e[:, 1:, :-1]  # 形状: (3, H-1, W-1)


    # 计算水平和垂直方向的差异
    dx_L = pix_L - right_pix_L
    dy_L = pix_L - down_pix_L
    dx_Le = pix_Le - right_pix_Le
    dy_Le = pix_Le - down_pix_Le

    # 计算 LTV 损失
    ltv_x = (beta * dx_Le ** 2) / (dx_L ** alpha + eps)
    ltv_y = (beta * dy_Le ** 2) / (dy_L ** alpha + eps)

    # 返回平均值
    return ((ltv_x + ltv_y) / 2).mean()











# def RawNeRF_loss_func(image: torch.Tensor, gt_image: torch.Tensor, clip_min: float = 0.0) -> torch.Tensor:
#     '''
#     adapted from:
#     /home/hjy/ws/EvGS/0_ref/no_ev/multinerf_RawNeRF_CVPR24/internal/train_utils.py
#     '''
#     # Clip raw values against 1 to match sensor overexposure behavior.
#     # 传感器过曝模拟（与JAX版本相同逻辑）
#     # img_clip = torch.clamp(image, 0.0, 1.0)
#     img_clip = torch.clamp(image, clip_min, 1.0)
#     gt_image = torch.clamp(gt_image, clip_min, 1.0)
#     resid_sq_clip = (img_clip - gt_image)**2  # 形状保持相同
#     # 梯度重加权（关键变化在梯度停止处理）
#     scaling_grad = 1.0 / (1e-3 + img_clip.detach())  # 使用detach()替代stop_gradient
#     # scaling_grad = 1.0 / (1e-3 + gt_image.detach())  # 使用detach()替代stop_gradient
#     # 最终损失计算（保持相同数学操作）
#     rtn = (resid_sq_clip * (scaling_grad ** 2)).mean()
#     # rtn = resid_sq_clip.mean()
#     if torch.isnan(rtn).any():
#         print("RawNeRF_loss_func: rtn is nan")
#     return rtn






def get_brightness_weights(render_image):
    """
    render_image: [3, H, W], float, range [0, 1]
    returns:
        weight_brht: [1, H, W], brighter pixel → higher weight
        weight_dark: [1, H, W], darker pixel → higher weight
    """
    # 防止梯度反传
    render_image = render_image.detach()

    # 转换为灰度图
    gray = 0.299 * render_image[0] + 0.587 * render_image[1] + 0.114 * render_image[2]

    # 归一化到 [0, 1]
    gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

    # 生成权重图
    weight_brht = gray_norm.unsqueeze(0)        # [1, H, W]
    weight_dark = 1.0 - weight_brht             # [1, H, W]

    return weight_brht, weight_dark

def img_loss_weight_brht_fn(weight_brht: torch.Tensor, image: torch.Tensor, gt_image: torch.Tensor, clip_min: float = 0.0, use_mask: bool = False) -> torch.Tensor:
    '''
    adapted from:
    /home/hjy/ws/EvGS/0_ref/no_ev/multinerf_RawNeRF_CVPR24/internal/train_utils.py
    '''
    # Clip raw values against 1 to match sensor overexposure behavior.
    img_clip = torch.clamp(image, clip_min, 1.0)
    gt_image = torch.clamp(gt_image, clip_min, 1.0)

    # L1 residual
    resid_clip = torch.abs(img_clip - gt_image)

    if use_mask:
        # 生成 mask，仅在 gt_image < 0.1 的地方屏蔽损失
        mask = gt_image >= 0.01  # H, W, 3 的布尔张量
        # 应用 mask，仅计算有效区域的 loss
        masked_loss = ((resid_clip * weight_brht)[mask] ).mean()
        if torch.isnan(masked_loss).any():
            print("RawNeRF_loss_func: masked_loss is nan")
        return masked_loss
    else:
        # # 不使用 mask，计算所有区域的 loss
        # unmasked_loss = (resid_sq_clip * (scaling_grad ** 2)).mean()
        # if torch.isnan(unmasked_loss).any():
        #     print("RawNeRF_loss_func: unmasked_loss is nan")
        # return unmasked_loss
        raise NotImplementedError("不使用 mask 的情况未实现")



COLOR_LOSS=None
def color_const_loss_fn(img):
    """
    计算色彩恒常性损失，使 R、G、B 颜色通道的均值相近
    :param img: 形状为 (3, H, W) 的输入图像
    :return: 标量 loss
    """
    mean_r = torch.mean(img[0])  # R 通道均值
    mean_g = torch.mean(img[1])  # G 通道均值
    mean_b = torch.mean(img[2])  # B 通道均值

    loss = torch.abs(mean_r - mean_g) + torch.abs(mean_g - mean_b) + torch.abs(mean_b - mean_r)
    return loss



def llnerf_gray_loss_y(rgb, ref=None, hyper_gray_loss_clip = 0):
    """ adapted from /home/hjy/ws/EvGS/0_ref/no_ev/LLNeRF/internal/train_utils.py

    rgb 应该是 hw3。   只要保证最后一维 对应 RGB 3通道即可
    ref 应该是 hw3
    """
    #* hyper param
    hyper_gray_variance_bias = 0.5

    # print(f"==>> rgb.shape: {rgb.shape}")  # ([770, 1156, 3])
    assert rgb.shape[-1] == 3,  rgb.shape  # 确保最后一个维度是 RGB 通道

    assert ref is not None
    # 计算 weight2， ref 的方差加上 bias
    # weight2 = ref.var(dim=-1, keepdim=True) + hyper_gray_variance_bias

    weight2 = ref.var(dim=-1, keepdim=True) + hyper_gray_variance_bias

    # 计算相邻像素的差异
    diffs = (rgb - torch.roll(rgb, 1, dims=-1)) ** 2  # shape: [..., 3]

    # 如果设置了 gray_loss_clip，则对差异进行裁剪
    if hyper_gray_loss_clip and hyper_gray_loss_clip > 0:
        diffs = torch.clamp(diffs, max=hyper_gray_loss_clip)

    # 计算最终的灰度损失
    return torch.sqrt( torch.relu( diffs.sum(dim=-1, keepdim=True) / 3 / weight2   ) ).mean()


def gray_loss(rgb, ref=None, hyper_gray_loss_clip = 0):
    """ adapted from /home/hjy/ws/EvGS/0_ref/no_ev/LLNeRF/internal/train_utils.py

    rgb 应该是 hw3。   只要保证最后一维 对应 RGB 3通道即可
    ref 应该是 hw3
    """
    #* hyper param
    hyper_gray_variance_bias = 0.5

    # print(f"==>> rgb.shape: {rgb.shape}")  # ([770, 1156, 3])
    assert rgb.shape[-1] == 3,  rgb.shape  # 确保最后一个维度是 RGB 通道


    assert ref is not None
    # 计算 weight2， ref 的方差加上 bias
    weight2 = ref.var(dim=-1, keepdim=True) + hyper_gray_variance_bias

    # 计算相邻像素的差异
    diffs = (rgb - torch.roll(rgb, 1, dims=-1)) ** 2  # shape: [..., 3]

    # 如果设置了 gray_loss_clip，则对差异进行裁剪
    if hyper_gray_loss_clip and hyper_gray_loss_clip > 0:
        diffs = torch.clamp(diffs, max=hyper_gray_loss_clip)

    # 计算最终的灰度损失
    return torch.sqrt(diffs.sum(dim=-1, keepdim=True) / 3 / weight2).mean()



def compute_white_balance_loss(x, alpha=20, epsilon=1e-6):
    '''
    copy from 0_ref/ev/IncEventGS/loss_utils.py
    '''
    """
    Custom loss function.
    x: Input value
    alpha: Parameter that controls the rate of gradient change
    epsilon: Smoothing parameter

    white_balance_weight = 0.01
    loss_white_balance = white_balance_weight*compute_white_balance_loss(img_ev_end.mean())
    """
    smooth_abs = torch.sqrt((x - 0.5) ** 2 + epsilon)
    return torch.sigmoid(alpha * (smooth_abs - 0.25))



OTHER_FUNC=None

def get_v_n3_enhanced(L,llnerf_alpha_n1_map,llnerf_gamma_n3_map):
    """
    adapted from:
    /home/hjy/ws/EvGS/0_ref/no_ev/LLNeRF/internal/models.py:
        L_enhanced = (L_sg / (coeff_alpha + 0.0001)) ** final_gamma + residual
    """

    gamma_base = fixed_gamma_value = 2.2
    residual = 0
    coeff_gamma = llnerf_gamma_n3_map

    # print(f"==>> llnerf_gamma_n3_map.max(): {llnerf_gamma_n3_map.max()}")  # llnerf_gamma_n3_map.max(): nan
    # print(f"==>> llnerf_gamma_n3_map.min(): {llnerf_gamma_n3_map.min()}")  # llnerf_gamma_n3_map.min(): nan
    # print(f"==>> llnerf_gamma_n3_map.shape: {llnerf_gamma_n3_map.shape}")
    # #todo 权宜之计：如果 llnerf_gamma_n3_map 有 nan 值，就将其全部置为 0
    # if torch.isnan(llnerf_gamma_n3_map).any():
    #     llnerf_gamma_n3_map = torch.zeros_like(llnerf_gamma_n3_map)

    # llnerf_gamma_n3_map = torch.zeros_like(llnerf_gamma_n3_map)

    assert not torch.isnan(llnerf_alpha_n1_map).any(), f"llnerf_alpha_n1_map contains NaN values,{llnerf_alpha_n1_map.min()}"
    assert not torch.isnan(llnerf_gamma_n3_map).any(), f"llnerf_gamma_n3_map contains NaN values,{llnerf_alpha_n1_map.min()}"


    llnerf_gamma_n3_map = torch.clamp(llnerf_gamma_n3_map,-1.0,1.0)
    #todo
    final_gamma = 1 / (2.2 + llnerf_gamma_n3_map)  # 1 / (gamma_base + coeff_gamma)
    # final_gamma = 1 / (2.2 )  # 1 / (gamma_base + coeff_gamma)


    assert not torch.isnan(L).any(), f"L contains NaN values,{llnerf_alpha_n1_map.min()}"
    # print(f"==>> L:")
    # g(L)
    # g(llnerf_alpha_n1_map)

    L_enhanced = (L / (llnerf_alpha_n1_map + 0.0001)) ** final_gamma    # + residual
    assert not torch.isnan(L_enhanced).any(), f"L_enhanced contains NaN values,{llnerf_alpha_n1_map.min()}"
    #todo 是否要注释？
    L_enhanced = torch.clamp(L_enhanced,0.0,1.0)
    return L_enhanced



def gainVis(img, tonemap_gain = 10.0):
    """
    img: torch   0.0~1.0,  hw3 or  3hw 均可
    """
    # tonemap_gain = 3.0
    # img = np.clip((img * tonemap_gain),0,255).astype(np.uint8)
    # gamma = 1/2.2
    # img = np.clip(np.power(img/255, gamma)*255,0,255).astype(np.uint8)
    if tonemap_gain==-1:
        rtn = img
    else:
        rtn = torch.clamp((img*tonemap_gain),0,1)
    return rtn



def gainVis_3h2w(gt_image):
    """将原始图像与增益可视化后的图像水平拼接
    Args:
        gt_image (Tensor): 原始图像张量，形状为 [C, H, W]   3hw or 1hw, 0.0~1.0
    Returns:
        Tensor: 水平拼接后的图像，形状为 [C, H, 2*W]
    """
    return torch.cat([gt_image, gainVis(gt_image)], dim=2)


def gainVis_gamm(img, gamma_corr=False, tonemap_gain = 10.0):
    """
    img: torch   0.0~1.0, 如 hw3 or  hw4, shape.shape[-1] == 3,

    """
    # tonemap_gain = 3.0
    # img = np.clip((img * tonemap_gain),0,255).astype(np.uint8)

    # gamma = 1/2.2
    # img = np.clip(np.power(img/255, gamma)*255,0,255).astype(np.uint8)

    if gamma_corr:
        img = srgb_to_rgb(img)

    img = torch.clamp((img*tonemap_gain),0,1)

    if gamma_corr:
        img = rgb_to_srgb(img)
    return img


def sRGB_to_linear_y(x):
    x = x ** 2.2
    return x


def gamma_brten_before_crf_y(x,epsilon=1e-6,gamma=2.2):
    # gamma=2.2
    x = (x + epsilon) ** (1/gamma)
    return x

def linear_to_sRGB_y(x,epsilon=1e-6,gamma=2.2):
    # gamma=2.2
    x = (x + epsilon) ** (1/gamma)
    return x

def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    ori: GI-GS
    '''
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        # srgb1 = 1.055 * torch.pow(torch.clamp(linear, min=eps), 1.0/2.4) - 0.055
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError

def srgb_to_linear(srgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    '''
    ori: GI-GS
    '''
    if isinstance(srgb, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError



def get_color_map(error_hw,vmin=0.0, vmax=1.0, colormap='COLORMAP_VIRIDIS'): # vmin, vmax 有用
    '''
    y 20240712 经过测试，这种方法, 速度不受影响
    param:
        error_hw: h,w      0.-1. float32
    return:
        errmap_hw3_BGR255: h,w,3    0-255 uint8   BGR

    example:
        real255=cv2.imread(pngs_path_list_gt[i])
        fake255=cv2.imread(pngs_path_list_pre1[i])
        error_hw3 = np.abs(real255 / 255.0 - fake255 / 255.0)
        error_hw = np.mean(error_hw3, axis=2)
        erro255  = get_color_map(error_hw,vmin=vmin, vmax=vmax, colormap=colormap)
    '''
    assert vmax>vmin
    error_hw = np.clip(error_hw, vmin, vmax)  # 将错误矩阵限制在 [vmin, vmax] 区间
    error_hw = (error_hw-vmin)/(vmax-vmin)  # 将错误矩阵归一化到 [0, 1] 区间
    error_hw = np.clip(error_hw*255,0,255).astype(np.uint8)   # 生成一个线性的错误矩阵
    # print(f"==>> error_hw.shape: {error_hw.shape}")

    # 将灰度图像转换为Jet样式的伪彩色图像
    if colormap=='COLORMAP_VIRIDIS':
        error_hw_virids = cv2.applyColorMap(error_hw, cv2.COLORMAP_VIRIDIS)  # 颜色映射
    elif colormap=='COLORMAP_JET':
        error_hw_virids = cv2.applyColorMap(error_hw, cv2.COLORMAP_JET)
    elif colormap=='COLORMAP_HOT':
        error_hw_virids = cv2.applyColorMap(error_hw, cv2.COLORMAP_HOT)
    errmap_hw3_BGR255 = error_hw_virids
    # print(f"==>> errmap_hw3_BGR255.shape: {errmap_hw3_BGR255.shape}")
    # print(f"==>> errmap_hw3_BGR255.dtype: {errmap_hw3_BGR255.dtype}")
    return errmap_hw3_BGR255 # hw3 255 BGR


def get_colorbar(colorbar_h, colorbar_w=80, bar_inner_w=20, bar_inner_h_ratio=0.8,
                 bar_x_offset=None, bar_x_offset_ratio=0.2, vmin=0.0, vmax=1.0, num_ticks=5, colormap='COLORMAP_VIRIDIS'):
    """
    生成颜色条（可单独使用）

    :param colorbar_h: 颜色条高度（应与误差图 h 一致）
    :param colorbar_w: 整个颜色条区域的宽度（包括白色背景 & 数值标注）
    :param bar_inner_w: 仅颜色条本身的宽度
    :param bar_inner_height_ratio: 颜色条本身的高度比例（相对于 h ，默认 0.8）
    :param bar_x_offset: 颜色条距离左侧的偏移量（若 None，则默认居中）
    :param bar_x_offset_ratio: 颜色条距离左侧的偏移量。bar_x_offset为None时，根据ratio设置。 如果ratio也为 None ，则默认居中）
    :param vmin: 颜色条最小值
    :param vmax: 颜色条最大值
    :param num_ticks: 颜色条上的刻度数量
    :return: (colorbar_h, colorbar_w, 3) 颜色条（BGR 格式）
    """
    assert bar_inner_w < colorbar_w, "bar_inner_w 应小于 colorbar_w"
    assert 0 < bar_inner_h_ratio <= 1, "bar_inner_height_ratio 应在 (0,1] 之间"

    colorbar_w = colorbar_w  # 颜色条总宽度
    # 计算颜色条的高度
    bar_inner_h = int(colorbar_h * bar_inner_h_ratio)
    bar_inner_w = bar_inner_w  # 仅颜色条的宽度

    # 创建白色背景
    colorbar = np.ones((colorbar_h, colorbar_w, 3), dtype=np.uint8) * 255  # 全白背景

    # 颜色条的垂直起始位置（确保在图像中居中）
    y_offset = (colorbar_h - bar_inner_h) // 2


    # 颜色条的水平起始位置（根据 bar_x_offset 调整）
    if bar_x_offset is None:
        if bar_x_offset_ratio:
            x_offset = int(colorbar_w * bar_x_offset_ratio)
        else:
            x_offset = (colorbar_w - bar_inner_w) // 2  # 默认让bar 中轴线 居中
    else:
        x_offset = min(max(0, bar_x_offset), colorbar_w - bar_inner_w)  # 限制范围，防止超出边界

    # 生成颜色条的渐变
    gradient = np.linspace(1, 0, bar_inner_h).reshape(bar_inner_h, 1)
    gradient = (gradient * 255).astype(np.uint8)

    colormap_dict = {
        'COLORMAP_VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'COLORMAP_JET': cv2.COLORMAP_JET,
        'COLORMAP_HOT': cv2.COLORMAP_HOT
    }
    gradient_color = cv2.applyColorMap(gradient, colormap_dict.get(colormap, cv2.COLORMAP_VIRIDIS))

    # 将颜色条插入白色背景
    colorbar[y_offset:y_offset + bar_inner_h, x_offset:x_offset + bar_inner_w] = gradient_color

    # ========== 添加刻度标注 ==========
    tick_positions = np.linspace(y_offset, y_offset + bar_inner_h - 1, num_ticks).astype(int)  # 刻度位置
    tick_values = np.linspace(vmax, vmin, num_ticks)  # 误差数值

    for i, (pos, val) in enumerate(zip(tick_positions, tick_values)):
        text = f"{val:.2f}"  # 格式化数值
        text_position = (x_offset + bar_inner_w + 5, pos + 5)  # 数值放在颜色条右侧
        cv2.putText(colorbar, text, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return colorbar  # colorbar: [colorbar_h,colorbar_w,3]  255 BGR


# def linear_to_srgb(linear: _Array,
#                    eps: Optional[float] = None,
#                    xnp: types.ModuleType = jnp) -> _Array:
#   """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
#   if eps is None:
#     eps = xnp.finfo(xnp.float32).eps
#   srgb0 = 323 / 25 * linear
#   srgb1 = (211 * xnp.maximum(eps, linear)**(5 / 12) - 11) / 200
#   return xnp.where(linear <= 0.0031308, srgb0, srgb1)


# def srgb_to_linear(srgb: _Array,
#                    eps: Optional[float] = None,
#                    xnp: types.ModuleType = jnp) -> _Array:
#   """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
#   if eps is None:
#     eps = xnp.finfo(xnp.float32).eps
#   linear0 = 25 / 323 * srgb
#   linear1 = xnp.maximum(eps, ((200 * srgb + 11) / (211)))**(12 / 5)
#   return xnp.where(srgb <= 0.04045, linear0, linear1)



# def linear_to_srgb(linear: torch.Tensor,
#                    eps: Optional[float] = None) -> torch.Tensor:
#     """
#     adapted from /home/hjy/ws/EvLLGS/0_ref/no_ev/LLNeRF_ICCV23/internal/image.py
#     chatGPT: jax to pytorch
#     假设 linear 在 [0, 1] 范围内，按照 sRGB 标准转换。
#     标准公式：
#       if linear <= 0.0031308: srgb = linear * 12.92
#       else: srgb = 1.055 * (linear^(1/2.4)) - 0.055
#     此处写法中：
#       323/25 = 12.92,
#       211/200 = 1.055,
#       11/200 = 0.055,
#       1/γ = 5/12  等价于 1/2.4
#     """
#     if eps is None:
#         eps = torch.finfo(torch.float32).eps
#     # 保证 eps 与 linear 数据类型和设备一致
#     eps_tensor = torch.tensor(eps, device=linear.device, dtype=linear.dtype)

#     # 计算两部分
#     srgb0 = (323 / 25) * linear
#     srgb1 = (211 * torch.maximum(eps_tensor, linear) ** (5 / 12) - 11) / 200

#     return torch.where(linear <= 0.0031308, srgb0, srgb1)


# def srgb_to_linear(srgb: torch.Tensor,
#                    eps: Optional[float] = None) -> torch.Tensor:
#     """
#     chatGPT:
#     假设 srgb 在 [0, 1] 范围内，按照 sRGB 标准转换回 linear。
#     标准公式：
#       if srgb <= 0.04045: linear = srgb / 12.92
#       else: linear = ((srgb + 0.055)/1.055)^2.4
#     此处写法中：
#       25/323 = 1/12.92,
#       ((200 * srgb + 11)/211) 等价于 (srgb + 0.055)/1.055,
#       12/5 = 2.4
#     """
#     if eps is None:
#         eps = torch.finfo(torch.float32).eps
#     eps_tensor = torch.tensor(eps, device=srgb.device, dtype=srgb.dtype)

#     linear0 = (25 / 323) * srgb
#     # 注意确保内部不会出现负数，加上 eps_tensor 保证数值稳定性
#     linear1 = torch.maximum(eps_tensor, ((200 * srgb + 11) / 211)) ** (12 / 5)

#     return torch.where(srgb <= 0.04045, linear0, linear1)




if __name__ == "__main__":
    root='../outputs/329d_lowImg_ev_linear_crf/nerf_llff_data_evllgs_v2e_linear/fern'
    print(f"==>> root: {root}")

    # img = cv2.imread("/home/hjy/ws/EvLLGS/outputs/d311a_lowlight/LLNeRF/still4_lowlight_y/train_40000/ours_None/retinex_rgb/00000_retinex.png")
    # out_path = "/home/hjy/ws/EvLLGS/outputs/d311a_lowlight/LLNeRF/still4_lowlight_y/train_40000/ours_None/retinex_rgb/00000_retinex_gainVis.png"
    # print(f"==>> out_path: {out_path}")

    # root = "/home/hjy/ws/EvLLGS/outputs/d311a_lowlight/LLNeRF/still4_lowlight_y/train_40000/ours_None"
    # in_path = f"{root}/gt/00000_gt.png"


    # img = cv2.imread(in_path)
    # # 将图像从 BGR 转换为 RGB（如果需要） 将图像转换为浮点数并归一化到 [0.0, 1.0]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # # 将 NumPy 数组转换为 PyTorch Tensor
    # img = torch.from_numpy(img)


    # # out_path = f"{root}/gt/00000_gt_gainVis_gain80_gamma.png"
    # # out = gainVis_gamm(img, gamma_corr=True, tonemap_gain = 80.0)
    # # out = out.permute(2, 0, 1)    # 调整维度顺序从 (H, W, C) 到 (C, H, W)


    # out_path = f"{root}/gt/00000_gt_gainVis_gain10.png"
    # out = gainVis_gamm(img, gamma_corr=False, tonemap_gain = 10.0) # img torch HW3 float1   # 可视化效果最好
    # out = out.permute(2, 0, 1)    # 调整维度顺序从 (H, W, C) 到 (C, H, W)


    # torchvision.utils.save_image(out, out_path)



class CSVWriter_y:
    '''
    csv_writer = CSVWriter_y(file_path="output_without_header.csv")
    for i in range(6):
        new_data = [i, i*10, i*20, i*30]
        csv_writer.write(new_data)
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.current_row = 0
        self._initialize_csv()

    def _initialize_csv(self):
        CSVWriter_y.rename_if_file_exists(self.file_path)
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)
        open(self.file_path, 'w').close()   # 创建空文件或清空现有文件
        self.current_row = 0                # 初始化时不包含header行

    def write(self, new_data):
        new_data_df = pd.DataFrame([new_data])          # 将新数据转换为 DataFrame
        new_data_df.to_csv(self.file_path, index=False, mode='a', header=False)  # 动态写入 CSV 文件
        self.current_row += len(new_data_df)            # 更新当前行数
        print(f"数据写入完成: self.file_path: {self.file_path}, new_data:\n", new_data)

    def get_row_count(self):
        return self.current_row

    @staticmethod
    def rename_if_file_exists(file_path):
        if os.path.isfile(file_path):                   # 如果存在， 且为文件路径
            _, suffix = os.path.splitext(file_path)     # eg.  '.csv'
            backup_file_path = f"{file_path.rsplit('.', 1)[0]}-archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"+ suffix
            os.rename(file_path, backup_file_path)
            print(f"文件已存在，重命名为: {backup_file_path}")
        else:
            print(f'文件不存在: {file_path}')


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def copyAllFoldersAndFiles(srcRoot,desRoot,exclude_txt_path):
    '''
    from utils_Sync_trainNet import copyAllFoldersAndFiles
    description:
        copy all files and folders in srcRoot to desRoot.
            if desRoot exists:  will overwrite desRoot
            if desRoot not exist: will create desRoot
    parameters:
        srcDir:
            src code direction
        desDir:
            dest direction
    # ignorePatterns=set(('0_del','0_src_bak','0_src*','logs','data', 'del*'))
    '''
    import os
    import shutil

    with open(exclude_txt_path, 'r') as f:
        lines = f.readlines()
        ignorePatterns = [line.strip() for line in lines]

    if os.path.exists(desRoot):
        shutil.rmtree(desRoot)
    os.makedirs(os.path.dirname(desRoot),exist_ok=True)
    shutil.copytree(srcRoot, desRoot,symlinks = True ,ignore=shutil.ignore_patterns(*ignorePatterns))




def stack_grid_to_cmap(stack_grid, vmax = 2):  # stack_grid:  N,H,W   dtype:int    -10 ~ 12等

    def data2d_to_bgr(data,cmap='seismic_r',vmin=0,vmax=2):   #  'jet'   seismic_r   seismic_r
        '''H,W 的数组，取值范围是 -min ~ max , 用 cmap 颜色来区别数值的大小
        args:
            data: H,W
            cmap: seismic_r 表示 红色低，蓝色高
        return:
            color: H,W,3   BGR uint8

        '''
        import matplotlib
        # norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
        norm = matplotlib.colors.Normalize(vmin, vmax)
        cmap = matplotlib.cm.get_cmap(cmap)  #   # viridis
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        # color = (sm.to_rgba(vmax-data[:,:])*255).astype(np.uint8)
        colorRGBA = (sm.to_rgba(data[:,:])*255).astype(np.uint8)
        # color = color[:,:,:-1]
        color = cv2.cvtColor(colorRGBA,cv2.COLOR_RGBA2BGR)
        return color

    #* save stack frame to png
    # print(f"==>> stack_grid minval,maxval: {stack_grid.min()},{stack_grid.max()}")
    # print(f"==>> minNum: {-vmax}")
    # print(f"==>> maxNum: {vmax}")

    n,h,w = stack_grid.shape
    stack_grid_colormap = np.zeros((n,h,w,3),dtype=np.uint8)
    frameNum = stack_grid.shape[0]
    for i in range(frameNum):
        stack_grid_colormap[i,:,:,:] = data2d_to_bgr(stack_grid[i,:,:]+vmax,vmin=0,vmax=2*vmax)

    return stack_grid_colormap  #  N,H,W,3