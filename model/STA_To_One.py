import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


######################### 时空角解耦块 #########################
class DisVFIBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # branches
        self.st_branch = STBranch(channels)    # 时空
        self.at_branch = ATBranch(channels)    # 时角
        self.sa_branch = SABranch(channels)    # 空角
        self.pointwise_fuse = nn.Conv3d(in_channels=channels*3, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, _, t, u, v, _, _ = x.shape
        f_st = self.st_branch(x)
        f_at = self.at_branch(x)
        f_sa = self.sa_branch(x)

        f_st = rearrange(f_st, 'b c t an1 an2 h w -> (b t) c (an1 an2) h w')
        f_at = rearrange(f_at, 'b c t an1 an2 h w -> (b t) c (an1 an2) h w')
        f_sa = rearrange(f_sa, 'b c t an1 an2 h w -> (b t) c (an1 an2) h w')
        y = self.pointwise_fuse(torch.cat([f_st, f_at, f_sa], dim=1))
        y = rearrange(y, '(b t) c (an1 an2) h w -> b c t an1 an2 h w', b=b, t=t, an1=u, an2=v)
        return y


# ---------- 三个并行分支（都使用3D conv） ----------
class STBranch(nn.Module):
    """时空分支：fold角度到batch，做 (T, H', W') 的 3D 卷积"""
    def __init__(self, channels):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),   # 空间
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),   # 空间
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False))   # 时空

    def forward(self, x):
        # [b,c,t,u,v,h,w]
        b, _, _, u, v, _, _ = x.shape
        x_st = rearrange(x, 'b c t an1 an2 h w -> (b an1 an2) c t h w')
        y = self.feature_extraction(x_st)     #  [buv,c,t,h,w]
        y = rearrange(y, '(b an1 an2) c t h w -> b c t an1 an2 h w', b=b, an1=u, an2=v)
        return x + y


class ATBranch(nn.Module):
    """时角分支：fold空间到batch，做 (T, U, V) 的 3D 卷积"""
    def __init__(self, channels):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),   # 角度
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),   # 角度
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False))   # 时角

    def forward(self, x):
        # [b,c,t,u,v,h,w]
        b, _, _, _, _, h, w = x.shape
        x_at = rearrange(x, 'b c t an1 an2 h w -> (b h w) c t an1 an2')
        y = self.feature_extraction(x_at)     # [bhw,c,t,u,v]
        y = rearrange(y, '(b h w) c t an1 an2 -> b c t an1 an2 h w', b=b, h=h, w=w)
        return x + y


class SABranch(nn.Module):
    """空角分支：fold time into batch，depth=U*V，做 (U*V, H', W') 的 3D 卷积"""
    def __init__(self,  channels):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),  # 空间
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),  # 空间
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False))  # 角度

    def forward(self, x):
        # [b,c,t,u,v,h,w]
        b, _, t, u, v, _, _ = x.shape
        x_sa = rearrange(x, 'b c t an1 an2 h w -> (b t) c (an1 an2) h w')
        y = self.feature_extraction(x_sa)       # [bt,c,uv,h,w]
        y = rearrange(y, '(b t) c (an1 an2) h w -> b c t an1 an2 h w', b=b, t=t, an1=u, an2=v)
        return x + y


class Residual3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return x + self.body(x)


######################### 时域融合块 #########################
class TWO_ONE_Frame(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.t_interp = TemporalInterpHead(channels)
        self.proj_view = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # [b,c,t,u,v,h,w]
        f_mid = self.t_interp(x)    # [b,c,u,v,h,w]
        f_view = rearrange(f_mid, 'b c an1 an2 h w -> (b an1 an2) c h w')
        proj = self.norm(self.proj_view(f_view))    # [buv,c,h,w]
        return proj


class TemporalInterpHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                       nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                       nn.Conv2d(channels, 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, f):
        # [b,c,t,u,v,h,w]
        b, _, _, u, v, _, _ = f.shape
        f_buffer = rearrange(f, 'b c t an1 an2 h w -> (b an1 an2) (c t) h w')
        attn = self.attn_conv(f_buffer)    # [b,2,h,w]
        attn = F.softmax(attn, dim=1)
        attn = rearrange(attn, '(b an1 an2) c h w -> b c an1 an2 h w', b=b, an1=u, an2=v)
        w1 = attn[:, 0:1, :, :, :, :]
        w2 = attn[:, 1:2, :, :, :, :]
        f_mid = w1 * f[:, :, 0, :, :, :, :] + w2 * f[:, :, 1, :, :, :, :]
        return f_mid


######################### 融合块 #########################
class FusionBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.conv1x1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, x, y):
        # [buv,c,h,w]
        feats = torch.cat((x, y), dim=1)
        out = self.conv1x1(feats)
        attn = self.channel_attention(out)
        out = out * attn
        return x + out

