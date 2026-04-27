import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def sub_mean(x):
    mean = x.mean(2, keepdim=True).mean(3, keepdim=True)
    x -= mean
    return x, mean


# Residual Atrous Spatial Pyramid Pooling (RASPP)
class ResASPP(nn.Module):
    def __init__(self, channels):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_d = nn.Conv2d(in_channels=channels*3, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)

    def __call__(self, x):
        # [b,c,h,w]
        buffer_1 = self.conv_1(x)
        buffer_2 = self.conv_2(x)
        buffer_3 = self.conv_3(x)
        buffer = self.conv_d(torch.cat([buffer_1, buffer_2, buffer_3], dim=1))
        return x + buffer


class InitFeatE(nn.Module):
    def __init__(self, channels):
        super(InitFeatE, self).__init__()
        self.conv_res = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False))

    def __call__(self, x):
        # [b,c,h,w]
        buffer = self.conv_res(x)
        return x + buffer


############################# 空角编码层 ########################################
class SAFE_base(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.SFE_start = GLOBAL_FE(channels, 4, 2.66, False, 'BiasFree')
        self.AFE = GLOBAL_FE(channels, 4, 2.66, False, 'BiasFree')
        self.SFE_end = GLOBAL_FE(channels, 4, 2.66, False, 'BiasFree')

    def forward(self, x):
        ang_res = 5
        B, C, H, W = x.shape
        b = B // (ang_res**2)
        out = self.SFE_start(x)

        out = rearrange(out, '(b an1 an2) c h w -> (b h w) c an1 an2', b=b, an1=ang_res, an2=ang_res)
        out = self.AFE(out)

        out = rearrange(out, '(b h w) c an1 an2 -> (b an1 an2) c h w', b=b, h=H, w=W)
        out = self.SFE_end(out)
        return out


############################# 全局特征提取 ########################################
class GLOBAL_FE(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(GLOBAL_FE, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
# Layer Normalization
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

