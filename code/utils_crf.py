import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
import scipy.io as scio
import numpy as np
def g(*args, **kwargs):  # debug no-op (removed private dmfq dep)
    return None



# Model
class HDR_NeRF_CRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_exps=1, output_ch=7, skips=[4], use_viewdirs=True, spilit=True):
        """ 
        """
        super(HDR_NeRF_CRF, self).__init__()
        # self.D = D
        self.W = W
        # self.input_ch = input_ch
        # self.input_ch_views = input_ch_views
        # self.input_ch_exps = input_ch_exps
        # self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.spilit = spilit
        # self.pts_linears = nn.ModuleList(
            # [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # add crf 定义 input_exps 作为可学习参数，初始化值为 0.2
        self.input_exps = nn.Parameter(torch.tensor(0.2))  

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        if self.spilit:
            self.exps_linears_r = nn.ModuleList([nn.Linear(1, W//2)])
            self.exps_linears_g = nn.ModuleList([nn.Linear(1, W//2)])
            self.exps_linears_b = nn.ModuleList([nn.Linear(1, W//2)])
            
            self.r_l_linner = nn.Linear(W//2, 1)
            self.b_l_linner = nn.Linear(W//2, 1)
            self.g_l_linner = nn.Linear(W//2, 1)
        else:
            self.exps_linears_r = nn.ModuleList([nn.Linear(1, W//2)])
            self.r_l_linner = nn.Linear(W//2, 1)
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # if use_viewdirs:
        #     # self.feature_linear = nn.Linear(W, W)
        #     # self.alpha_linear = nn.Linear(W, 1)
        #     # self.rgb_linear = nn.Linear(W//2, 3)
        #     # self.b_l_linner = nn.Linear(W//2, 1)
        #     # self.r_l_linner = nn.Linear(W//2, 1)
        #     # self.g_l_linner = nn.Linear(W//2, 1)
        #     # self.rgb_l_linner = nn.Linear(W//2, 1)
        #     pass
        # else:
        #     raise NotImplementedError
            # self.output_linear = nn.Linear(W, output_ch)

    def forward(self, rgb_h_3hw):
        input_exps = torch.clamp(self.input_exps, min=0.01, max=0.5)  # 限制 input_exps 取值范围
        # input_exps = torch.clamp(self.input_exps, min=0.2, max=0.5)  # 限制 input_exps 取值范围

        rgb_h_3hw = torch.log(rgb_h_3hw + 1e-6)

        rgb_h_hw3 = rgb_h_3hw.permute(1, 2, 0)
        height,width,_ = rgb_h_hw3.shape
        rgb_h_n3 = rgb_h_hw3.reshape(-1, 3)
        rgb_h = rgb_h_n3    # 空间， shape 是 N,3,       
        # rgb_h = self.rgb_linear(h)   # HDR 图片  rgb_h: (N, 3)
        # rgb_h = rgb_h   
        if self.spilit:
            ## Split RGB channels
            r_h_s = rgb_h[:,0:1] + torch.log(input_exps)
            g_h_s = rgb_h[:,1:2] + torch.log(input_exps)
            b_h_s = rgb_h[:,2:3] + torch.log(input_exps)

            lnx = torch.cat([r_h_s, g_h_s, b_h_s], -1)

            r_h = lnx[:,0:1]
            g_h = lnx[:,1:2]
            b_h = lnx[:,2:3]
            
            for i, l in enumerate(self.exps_linears_r):
                r_h = self.exps_linears_r[i](r_h)
                r_h = F.relu(r_h)
            r_l = self.r_l_linner(r_h)

            for i, l in enumerate(self.exps_linears_g):
                g_h = self.exps_linears_g[i](g_h)
                g_h = F.relu(g_h)
            g_l = self.g_l_linner(g_h)

            for i, l in enumerate(self.exps_linears_b):
                b_h = self.exps_linears_b[i](b_h)
                b_h = F.relu(b_h)
            b_l = self.b_l_linner(b_h)

            rgb_l = torch.cat([r_l, g_l, b_l], -1)  # LDR 图片

        else:
            # raise NotImplementedError
            # rgb_h_r = rgb_h[:,:,None]
            # input_exps = input_exps[:,:,None].repeat(1,3,1)
            # h = rgb_h_r + torch.log(input_exps)
            # lnx = torch.reshape(h, [-1, 3])
            # h = torch.reshape(h, [-1, 1])
            # for i, l in enumerate(self.exps_linears_r):
            #     h = self.exps_linears_r[i](h)
            #     h = F.relu(h)
            # rgb_l = self.rgb_l_linner(h)
            # rgb_l = torch.reshape(rgb_l, [-1, 3])
            r_h_s = rgb_h[:,0:1] + torch.log(input_exps)
            g_h_s = rgb_h[:,1:2] + torch.log(input_exps)
            b_h_s = rgb_h[:,2:3] + torch.log(input_exps)

            lnx = torch.cat([r_h_s, g_h_s, b_h_s], -1)

            r_h = lnx[:,0:1]
            g_h = lnx[:,1:2]
            b_h = lnx[:,2:3]
            
            for i, l in enumerate(self.exps_linears_r):
                r_h = self.exps_linears_r[i](r_h)
                r_h = F.relu(r_h)
            r_l = self.r_l_linner(r_h)

            for i, l in enumerate(self.exps_linears_r):
                g_h = self.exps_linears_r[i](g_h)
                g_h = F.relu(g_h)
            g_l = self.r_l_linner(g_h)

            for i, l in enumerate(self.exps_linears_r):
                b_h = self.exps_linears_r[i](b_h)
                b_h = F.relu(b_h)
            b_l = self.r_l_linner(b_h)

            rgb_l = torch.cat([r_l, g_l, b_l], -1)  # LDR 图片
        
        rgb_l_n3 = rgb_l
        rgb_l_hw3 = rgb_l_n3.reshape(height, width, 3)
        rgb_l_3hw = rgb_l_hw3.permute(2, 0, 1)

        rgb_l_3hw = torch.exp(rgb_l_3hw)
        return rgb_l_3hw



class EventDegrade(nn.Module):
    """Event-branch degradation model F.

    The event camera does not observe the clean scene radiance directly: its
    response is shaped by photoreceptor band-limiting, leak/shot noise and
    delayed responses. This module learns that per-channel mapping F so that the
    predicted events can be formed as  E_hat = Delta log( F(I_r) ).

    The mapping is parameterized as a small MLP acting on each radiance value and
    applied multiplicatively in the log domain, which keeps the output strictly
    positive and initialized close to the identity.
    """

    def __init__(self, W: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, W), nn.ReLU(inplace=True),
            nn.Linear(W, W), nn.ReLU(inplace=True),
            nn.Linear(W, 1),
        )

    def forward(self, rgb_3hw: torch.Tensor) -> torch.Tensor:
        # rgb_3hw: [3, H, W], non-negative scene radiance
        C, H, W = rgb_3hw.shape
        x = rgb_3hw.reshape(C, H * W).permute(1, 0).reshape(-1, 1)  # [(C*H*W), 1]
        delta = torch.clamp(self.net(x), min=-2.0, max=2.0)        # bounded log-gain
        out = x * torch.exp(delta)                                  # positive, ~identity at init
        out = out.reshape(H * W, C).permute(1, 0).reshape(C, H, W)
        return out



def get_crf_regularization_loss(model, eps=1e-6, H=32, W=32):
    """
    综合正则项：
    1. 限制 ln(0+eps) -> ln(0)，防止低亮度映射出大值（loss_zero）
    2. 防止输出对输入的负梯度（即亮度越小输出越大）（loss_grad）

    Args:
        model: 接受 (3, H, W) 输入的 CRF 网络，输出也是 (3, H, W)
        eps: 避免 log(0) 的极小值
        H, W: 用于构造虚拟图像的高度和宽度

    Returns:
        loss_zero: 输入为黑图时，输出也应接近0
        loss_grad: 输入亮度递增时，输出应单调递增（梯度非负）
    """

    device = next(model.parameters()).device

    # ==== 约束0输入映射到0输出 ====
    # === 1. 零亮度输入约束 ===
    ln_x0 = torch.ones([3, H, W], device=device, requires_grad=True) # torch.tensor([[(eps)]], device=device, requires_grad=True)
    out_zero = model(ln_x0)
    loss_zero = torch.abs(out_zero - 0.0).mean()

    # ==== 限制梯度方向，防止“输入越小输出反而更大” ====
    with torch.enable_grad():
        x = torch.linspace(0, 1.5, 1000, device=device).reshape([-1, 1])
        rgb_h_3hw = x ** (1/2.2)
        ln_x = torch.log(rgb_h_3hw + eps).detach().clone().requires_grad_(True)
        ln_x.requires_grad_(True)

        # forward through model

        r_h = ln_x
        g_h = ln_x
        b_h = ln_x
        for i, l in enumerate(model.exps_linears_r):
            r_h = model.exps_linears_r[i](r_h)
            r_h = F.relu(r_h)
        r_l = model.r_l_linner(r_h)
        
        for i, l in enumerate(model.exps_linears_g):
            g_h = model.exps_linears_g[i](g_h)
            g_h = F.relu(g_h)
        g_l = model.g_l_linner(g_h)

        for i, l in enumerate(model.exps_linears_b):
            b_h = model.exps_linears_b[i](b_h)
            b_h = F.relu(b_h)
        b_l = model.g_l_linner(b_h)


        rgb_l = torch.cat([r_l, g_l, b_l], dim=-1)
        y = torch.exp(rgb_l)

        crf_grad = torch.autograd.grad(
            outputs=y,
            inputs=ln_x,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True
        )[0]

        loss_grad = F.relu(-crf_grad).mean()

    return loss_zero, loss_grad




def draw_CRF_xLinear_tbwriter_add_figure(tb_writer, basedir, input_exps, iteration, model, x_lim=1.5,y_lim=None, crf_x_domain='linear'):
    os.makedirs(basedir, exist_ok=True)
    print(f"input_exps: {input_exps}")
    device = next(model.parameters()).device  # 获取 model 的设备
    
    x = torch.linspace(0, 1.5, 1000, device=device).reshape([-1, 1])  # 直接在 CUDA 上创建
    # x = torch.linspace(0, 1.5, steps=H * W, device=device).reshape(1, H, W) 
    x_sRGB=x**(1/2.2)
    # ln_x = torch.linspace(-5, 3, 1000, device=device).reshape([-1, 1])  # 直接在 CUDA 上创建
    ln_x = torch.log(x_sRGB + 1e-6)

    r_h = ln_x + torch.log(torch.tensor(input_exps))
    g_h = ln_x + torch.log(torch.tensor(input_exps))
    b_h = ln_x + torch.log(torch.tensor(input_exps))

    # r_h = ln_x #+ torch.log(torch.tensor(input_exps))
    # g_h = ln_x #+ torch.log(torch.tensor(input_exps))
    # b_h = ln_x #+ torch.log(torch.tensor(input_exps))

    if model.spilit:
        for i, l in enumerate(model.exps_linears_r):
            r_h = model.exps_linears_r[i](r_h)
            r_h = F.relu(r_h)
        r_l = model.r_l_linner(r_h)

        for i, l in enumerate(model.exps_linears_g):
            g_h = model.exps_linears_g[i](g_h)
            g_h = F.relu(g_h)
        g_l = model.g_l_linner(g_h)

        for i, l in enumerate(model.exps_linears_b):
            b_h = model.exps_linears_b[i](b_h)
            b_h = F.relu(b_h)
        b_l = model.b_l_linner(b_h)
    else:
        for i, l in enumerate(model.exps_linears_r):
            r_h = model.exps_linears_r[i](r_h)
            r_h = F.relu(r_h)
        r_l = model.r_l_linner(r_h)

        for i, l in enumerate(model.exps_linears_r):
            g_h = model.exps_linears_r[i](g_h)
            g_h = F.relu(g_h)
        g_l = model.r_l_linner(g_h)

        for i, l in enumerate(model.exps_linears_r):
            b_h = model.exps_linears_r[i](b_h)
            b_h = F.relu(b_h)
        b_l = model.r_l_linner(b_h)

    # rgb_l = torch.sigmoid(torch.cat([r_l, g_l, b_l], -1))
    rgb_l = (torch.cat([r_l, g_l, b_l], -1))
    print(f"==>> rgb_l: ")
    g(rgb_l)
    # 先将数据移回 CPU，再进行 NumPy 转换
    x = x.cpu().numpy()

    rgb_l=torch.clamp(rgb_l, max=0.0)
    y = torch.exp(rgb_l).detach().cpu().numpy()
    xLinear=x  #**(2.2)
    # # simple tone mapper for synthetic dataset
    # def tonemapSimple(x):
    #     return (x / (x + 1)) ** (1 / 2.2)

    # z_simple = np.clip(tonemapSimple(np.exp(x)), 0, 1)



    plt.rcParams.update({
    'font.size': 16  # 整体字体大小，例如改为16，可按需调大调小
    })
    fig, ax = plt.subplots()

    if crf_x_domain == 'linear':
        ax.set_xlabel("Real Radiance")
        ax.set_ylabel("Low-Light Pixel")
        # plt.plot(x, y[:, 0:1], color='r', label='red')
        # plt.plot(x, y[:, 1:2], color='g', label='green')
        # plt.plot(x, y[:, 2:3], color='b', label='blue')
        ax.plot(xLinear, y[:, 0:1], color='r', label='red')
        ax.plot(xLinear, y[:, 1:2], color='g', label='green')
        ax.plot(xLinear, y[:, 2:3], color='b', label='blue')

        # print(f"==>> xLinear: ")
        # g(xLinear)
        # print(f"==>> color b: ")
        # g(y[:, 2:3])
        if x_lim is not None:
            ax.set_xlim(-(x_lim/10), x_lim)
        if y_lim is not None:
            ax.set_ylim(-(y_lim/10), y_lim)
        # plt.plot(x, z_simple, color='y', label='CRF GT')
        # plt.legend(['CRF red', 'CRF green', 'CRF blue', 'CRF GT'])

    elif crf_x_domain == 'log':

        ax.set_xlabel("Log Real Radiance")
        ax.set_ylabel("Low-Light Pixel")
        x_log = np.log(x*255 + 1e-6)

        ax.plot(x_log, y[:, 0:1], color='r', label='red')
        ax.plot(x_log, y[:, 1:2], color='g', label='green')
        ax.plot(x_log, y[:, 2:3], color='b', label='blue')

        if x_lim is not None:
            x_lim_log = np.log(x_lim*255 + 1e-6)
            ax.set_xlim(-6, x_lim_log)
        if y_lim is not None:
            ax.set_ylim(0, y_lim)



    ax.legend(['red', 'green', 'blue']) # , loc='upper left' , 'GT'
    ax.grid()

    plt.savefig(os.path.join(basedir, f'degrade_curve_{iteration:0>5d}_xlim{x_lim}_ylim{y_lim}.png'))
    # plt.close()
    #! add 写入 TensorBoard
    tb_writer.add_figure("curve/crf_lowImg", fig, global_step=iteration)
    plt.close(fig)


    CRF = np.concatenate([x, y], -1)
    CRF = {'crf_ours': CRF}
    scio.savemat(os.path.join(basedir, 'degrade_curve.mat'), CRF)

    # return xLinear,x,y,x_lim,y_lim,basedir



if __name__ == "__main__":
    # 测试代码
    # basedir = './test'
    # input_exps = 0.2
    # iteration = 1
    sce_list="fern flower fortress horns leaves orchids room trex"
    sce_list=sce_list.split()

    for sce in sce_list:

        root=f'../outputs/329d_lowImg_ev_linear_crf/nerf_llff_data_evllgs_v2e_linear/{sce}'
        print(f"==>> root: {root}")

        checkpoint_path=f'{root}/chkpnt_hdr_nerf_crf40000.pth'
        checkpoint = torch.load(checkpoint_path)


        hdr_nerf_crf = HDR_NeRF_CRF()
        hdr_nerf_crf.load_state_dict(checkpoint['hdr_nerf_crf'])

        
        
        # draw_CRF(basedir=f'{root}/crf', input_exps=0.2, iteration=40000, model=hdr_nerf_crf, x_lim=1.0,y_lim=None)
        ## draw_CRF_before_exp331c1(basedir, iteration, model)