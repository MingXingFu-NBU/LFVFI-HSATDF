import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from model.CA_self import SAFE_base, InitFeatE
from model.STA_To_One import DisVFIBlock, TWO_ONE_Frame, FusionBlock


class LFFInet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        channels = opt.channel
        scale = opt.down_sample
        scale2 = scale * scale
        self.pixel_down1 = nn.PixelUnshuffle(scale)
        self.pixel_down2 = nn.PixelUnshuffle(scale)
        self.headConv = nn.Sequential(nn.Conv2d(in_channels=3*scale2*2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       InitFeatE(channels=channels))

        self.initConv1 = nn.Sequential(nn.Conv2d(in_channels=3*scale2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       InitFeatE(channels=channels))
        self.initConv2 = nn.Sequential(nn.Conv2d(in_channels=3*scale2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
                                       InitFeatE(channels=channels))

        self.encoder_level1_SAFE = SAFE_base(channels)
        self.encoder_level2_SAFE = SAFE_base(channels)
        self.encoder_level3_SAFE = SAFE_base(channels)
        self.encoder_level4_SAFE = SAFE_base(channels)
        self.encoder_level5_SAFE = SAFE_base(channels)

        self.LF_VFI_group_1 = nn.Sequential(DisVFIBlock(channels),
                                            DisVFIBlock(channels))
        self.LF7D_2_TO_ONE_1 = TWO_ONE_Frame(channels)
        self.Fusion_1 = FusionBlock(channels)

        self.LF_VFI_group_2 = nn.Sequential(DisVFIBlock(channels),
                                            DisVFIBlock(channels))
        self.LF7D_2_TO_ONE_2 = TWO_ONE_Frame(channels)
        self.Fusion_2 = FusionBlock(channels)

        self.LF_VFI_group_3 = nn.Sequential(DisVFIBlock(channels),
                                            DisVFIBlock(channels))
        self.LF7D_2_TO_ONE_3 = TWO_ONE_Frame(channels)
        self.Fusion_3 = FusionBlock(channels)

        self.LF_VFI_group_4 = nn.Sequential(DisVFIBlock(channels),
                                            DisVFIBlock(channels))
        self.LF7D_2_TO_ONE_4 = TWO_ONE_Frame(channels)
        self.Fusion_4 = FusionBlock(channels)

        self.LF_VFI_group_5 = nn.Sequential(DisVFIBlock(channels),
                                            DisVFIBlock(channels))
        self.LF7D_2_TO_ONE_5 = TWO_ONE_Frame(channels)
        self.Fusion_5 = FusionBlock(channels)

        self.tailConv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * scale2, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.PixelShuffle(scale),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, in_x1, in_x2):
        b, _, u, v, _, _ = in_x1.shape
        x1_2d = rearrange(in_x1, 'b c an1 an2 h w -> (b an1 an2) c h w')     # [buv,c,h,w]
        x2_2d = rearrange(in_x2, 'b c an1 an2 h w -> (b an1 an2) c h w')
        ###################### Downsampling ######################
        x1_2d = self.pixel_down1(x1_2d)
        x2_2d = self.pixel_down2(x2_2d)

        ###################### Initial feature extraction ######################
        x_2d_feats = self.headConv(torch.cat([x1_2d, x2_2d], dim=1))     # [buv,c,h,w]
        x_feats1 = self.initConv1(x1_2d)      # [buv,c,h,w]
        x_feats2 = self.initConv2(x2_2d)

        x_feats1 = rearrange(x_feats1, '(b an1 an2) c h w -> b c an1 an2 h w', b=b, an1=u, an2=v)    # [b,c,u,v,h,w]
        x_feats2 = rearrange(x_feats2, '(b an1 an2) c h w -> b c an1 an2 h w', b=b, an1=u, an2=v)
        x_7d_feats = torch.stack([x_feats1, x_feats2], dim=2)       # [b,c,2,u,v,h,w]

        ############################# Stage 1 #############################
        res_1 = self.encoder_level1_SAFE(x_2d_feats)
        x_7d_1 = self.LF_VFI_group_1(x_7d_feats)
        x_2d_1 = self.LF7D_2_TO_ONE_1(x_7d_1)
        res_1 = self.Fusion_1(res_1, x_2d_1)

        ############################# Stage 2 #############################
        res_2 = self.encoder_level2_SAFE(res_1)
        x_7d_2 = self.LF_VFI_group_2(x_7d_1)
        x_2d_2 = self.LF7D_2_TO_ONE_2(x_7d_2)
        res_2 = self.Fusion_2(res_2, x_2d_2)

        ############################# Stage 3 #############################
        res_3 = self.encoder_level3_SAFE(res_2)
        x_7d_3 = self.LF_VFI_group_3(x_7d_2)
        x_2d_3 = self.LF7D_2_TO_ONE_3(x_7d_3)
        res_3 = self.Fusion_3(res_3, x_2d_3)

        ############################# Stage 4 #############################
        res_4 = self.encoder_level4_SAFE(res_3)
        x_7d_4 = self.LF_VFI_group_4(x_7d_3)
        x_2d_4 = self.LF7D_2_TO_ONE_4(x_7d_4)
        res_4 = self.Fusion_4(res_4, x_2d_4)

        ############################# Stage 5 #############################
        res_5 = self.encoder_level5_SAFE(res_4)
        x_7d_5 = self.LF_VFI_group_5(x_7d_4)
        x_2d_5 = self.LF7D_2_TO_ONE_5(x_7d_5)
        res_5 = self.Fusion_5(res_5, x_2d_5)

        ############################ Reconstruction ############################
        res_feats = res_5 + x_2d_feats
        out = self.tailConv(res_feats)
        out = rearrange(out, '(b an1 an2) c h w -> b c an1 an2 h w', b=b, an1=u, an2=v)
        return out


if __name__ == "__main__":
    from thop import profile
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #########################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=64, help="Number of feature channels")
    parser.add_argument("--down_sample", type=int, default=4, help="Downsampling factor")
    opt = parser.parse_args()
    B = 1; C = 3; T = 2; U = 5; V = 5; H = 128; W = 128
    x0 = torch.randn(B, C, U, V, H, W).to(device)
    x1 = torch.randn(B, C, U, V, H, W).to(device)
    x = torch.stack([x0, x1], dim=2)  # (B,3,2,5,5,128,128)

    net = LFFInet(opt).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=((x0), (x1), ))

    print('Number of parameters: %.3fM' % (params / 1e6))
    print('Number of FLOPs: %.3fG' % (flops / 1e9))

    x = x.to(device)
    with torch.no_grad():
        y = net(x0, x1)
        print("Input shape:", x.shape)
        print("Output shape:", y.shape)  # expect (B,3,5,5,128,128)

# Number of parameters: 7.656M
# Number of FLOPs: 340.714G









