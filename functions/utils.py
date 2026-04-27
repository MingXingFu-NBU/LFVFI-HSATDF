import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from scipy.signal import convolve2d
from einops import rearrange
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips


def mk_dir(required_dir):
    if not os.path.exists(required_dir):
        os.makedirs(required_dir)


def save_checkpoint(state, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)


def to_2d(in_data):
    # [b,c,ah,aw,h,w]
    out_data = rearrange(in_data, 'b c ah aw h w -> (b ah aw) c h w')
    return out_data


def lfi2mlia(in_lfi):
    # [3,ah,aw,h,w] to [h*ah,w*aw,c]
    out_mlia = rearrange(in_lfi, 'c ah aw h w -> (h ah) (w aw) c')
    return out_mlia


def lfi2csai(in_lfi):
    # [3,ah,aw,h,w] to [h,w,c]
    _, u, v, _, w = in_lfi.shape
    out_sai = in_lfi[:, u//2, v//2, :, :]
    out_sai = rearrange(out_sai, 'c h w -> h w c')
    return out_sai


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    mat = torch.tensor([[ 0.257,  0.504,  0.098],
                        [-0.148, -0.291,  0.439],
                        [ 0.439, -0.368, -0.071]],
                       dtype=rgb.dtype, device=rgb.device).T        # 注意转置
    bias = torch.tensor([0.0625, 0.5, 0.5], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)

    ycbcr = torch.matmul(rgb.permute(0, 2, 3, 1), mat)      # (B, H, W, 3)
    ycbcr = ycbcr.permute(0, 3, 1, 2) + bias                # (B, 3, H, W)
    return ycbcr


def cal_rgb_metrics(out, label):
    C, U, V, H, W = label.shape

    # 准备存储指标
    PSNR = np.zeros(shape=(U, V, C), dtype='float32')   # 存储每个通道的PSNR
    SSIM = np.zeros(shape=(U, V, C), dtype='float32')   # 存储每个通道的SSIM

    for channel in range(C):
        out_channel = out[channel, :, :, :, :].data.cpu()
        label_channel = label[channel, :, :, :, :].data.cpu()

        for u in range(U):
            for v in range(V):
                PSNR[u, v, channel] = psnr(label_channel[u, v, :, :].numpy(),
                                           out_channel[u, v, :, :].numpy(), data_range=1.0)
                SSIM[u, v, channel] = ssim(label_channel[u, v, :, :].numpy(),
                                           out_channel[u, v, :, :].numpy(),
                                           gaussian_weights=True, data_range=1.0)

    # 计算通道的平均PSNR和SSIM
    PSNR_avg = PSNR.mean(axis=-1)  # 在RGB通道上求平均
    SSIM_avg = SSIM.mean(axis=-1)  # 在RGB通道上求平均
    psnr_mean = PSNR_avg.sum() / np.sum(PSNR_avg > 0)
    ssim_mean = SSIM_avg.sum() / np.sum(SSIM_avg > 0)
    return psnr_mean.item(), ssim_mean.item()


def cal_y_metrics(out, label):
    U, V, H, W = label.shape

    # 准备存储指标
    PSNR = np.zeros(shape=(U, V), dtype='float32')   # 存储每个通道的PSNR
    SSIM = np.zeros(shape=(U, V), dtype='float32')   # 存储每个通道的SSIM

    for u in range(U):
        for v in range(V):
            PSNR[u, v] = psnr(label[u, v, :, :].cpu().numpy(),
                              out[u, v, :, :].cpu().numpy(), data_range=1.0)
            SSIM[u, v] = ssim(label[u, v, :, :].cpu().numpy(),
                              out[u, v, :, :].cpu().numpy(), gaussian_weights=True, data_range=1.0)

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)
    return psnr_mean.item(), ssim_mean.item()


def cal_lpips_metrics(out, label, loss_fn):
    C, U, V, H, W = label.shape

    # To [-1,1]
    out = out / 0.5 - 1.
    label = label / 0.5 - 1.

    # 准备存储指标
    LPIPS_value = np.zeros(shape=(U, V), dtype='float32')
    for u in range(U):
        for v in range(V):
            dis_img = out[:, u, v, :, :].cuda()
            ref_img = label[:, u, v, :, :].cuda()
            LPIPS_value[u, v] = loss_fn.forward(dis_img.unsqueeze(dim=0), ref_img.unsqueeze(dim=0)).item()

    lpips_mean = LPIPS_value.sum() / (U * V)
    return lpips_mean.item()


def crop_lf_patch(LF_data, spa_length, spa_bound):
    # LF_data: [b,c,ah,aw,h,w]
    test_b, test_c, test_ah, test_aw, test_h, test_w = LF_data.shape

    if test_h % spa_length == 0:
        row_num = test_h // spa_length - 1
    else:
        row_num = test_h // spa_length

    if test_w % spa_length == 0:
        col_num = test_w // spa_length - 1
    else:
        col_num = test_w // spa_length

    # [1,b,c,ah,aw,h,w]
    LF_patch_volume = torch.zeros((1, test_b, test_c, test_ah, test_aw, spa_length + spa_bound, spa_length + spa_bound)).to(LF_data.device)

    # left top
    for row_cp in range(row_num):
        for col_cp in range(col_num):
            crop_LF_patch = LF_data[:, :, :, :, row_cp * spa_length:(row_cp + 1) * spa_length + spa_bound, col_cp * spa_length:(col_cp + 1) * spa_length + spa_bound]
            crop_LF_patch = crop_LF_patch.unsqueeze(0)
            LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    h_bound_start = test_h - spa_length - spa_bound
    w_bound_start = test_w - spa_length - spa_bound

    # right
    for row_cp in range(row_num):
        crop_LF_patch = LF_data[:, :, :, :, row_cp * spa_length:(row_cp + 1) * spa_length + spa_bound, w_bound_start:]
        crop_LF_patch = crop_LF_patch.unsqueeze(0)
        LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # bottom
    for col_cp in range(col_num):
        crop_LF_patch = LF_data[:, :, :, :, h_bound_start:, col_cp * spa_length:(col_cp + 1) * spa_length + spa_bound]
        crop_LF_patch = crop_LF_patch.unsqueeze(0)
        LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # right bottom
    crop_LF_patch = LF_data[:, :, :, :, h_bound_start:, w_bound_start:]
    crop_LF_patch = crop_LF_patch.unsqueeze(0)
    LF_patch_volume = torch.cat([LF_patch_volume, crop_LF_patch], dim=0)

    # [num,b,c,ah,aw,h,w]
    LF_patch_volume = LF_patch_volume[1:, :, :, :, :, :, :]
    return LF_patch_volume, row_num, col_num


def merge_lf_patch(in_lf_patch_volume, rnum, cnum, sr_h, sr_w, spa_length, spa_bound):
    # in_lf_patch_volume: [num,b,c,ah,aw,h,w]
    rec_n, rec_b, rec_c, rec_ah, rec_aw, rec_h, rec_w = in_lf_patch_volume.shape

    h_bound = sr_h - spa_length * rnum
    w_bound = sr_w - spa_length * cnum
    spa_bound_sub = spa_bound // 2

    # left top
    rec_lf_data = torch.zeros(rec_b, rec_c, rec_ah, rec_aw, sr_h, sr_w).to(in_lf_patch_volume.device)
    pvx = 0
    for pvi in range(rnum):
        for pvj in range(cnum):
            if (pvi==0 and pvj==0):
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]

            elif (pvi == 0 and pvj > 0):
                pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]

            elif (pvi>0 and pvj==0):
                pre_lf_patch = in_lf_patch_volume[pvx - cnum, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]

            else:
                pre_lf_patch1 = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
                pre_lf_patch2 = in_lf_patch_volume[pvx - cnum, :, :, :, :, :, :]
                tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
                tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch1[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]
                tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch2[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]
            rec_lf_data[:, :, :, :, pvi*spa_length:(pvi+1)*spa_length, pvj*spa_length:(pvj+1)*spa_length] = tmp_lf_patch[:, :, :, :, :spa_length, :spa_length]
            pvx = pvx + 1

    # right
    for pvk in range(rnum):
        if (pvk==0):
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
        else:
            pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
            tmp_lf_patch[:, :, :, :, :spa_bound_sub, :] = pre_lf_patch[:, :, :, :, spa_length:spa_length + spa_bound_sub, :]

        rec_lf_data[:, :, :, :, pvk*spa_length:(pvk+1)*spa_length, -w_bound:] = tmp_lf_patch[:, :, :, :, :spa_length, -w_bound:]
        pvx = pvx + 1

    # bottom
    for pvl in range(cnum):
        if (pvl == 0):
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
        else:
            pre_lf_patch = in_lf_patch_volume[pvx - 1, :, :, :, :, :, :]
            tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
            tmp_lf_patch[:, :, :, :, :, :spa_bound_sub] = pre_lf_patch[:, :, :, :, :, spa_length:spa_length + spa_bound_sub]

        rec_lf_data[:, :, :, :, -h_bound:, pvl*spa_length:(pvl+1)*spa_length] = tmp_lf_patch[:, :, :, :, -h_bound:, :spa_length]
        pvx = pvx + 1

    # right bottom
    tmp_lf_patch = in_lf_patch_volume[pvx, :, :, :, :, :, :]
    rec_lf_data[:, :, :, :, -h_bound:, -w_bound:] = tmp_lf_patch[:, :, :, :, -h_bound:, -w_bound:]
    return rec_lf_data

def crop_patch(in_lf_image, an, spa_length, spa_bound):

    h,w,c=in_lf_image.shape

    test_h = int(in_lf_image.shape[0])#计算输入图像的高度
    test_w = int(in_lf_image.shape[1])#计算输入图像的宽度

    crop_lengh = spa_length * an      # Height/Width of MLIA an用来调节块的大小的参数
    crop_bound = spa_bound * an


    row_num = test_h // crop_lengh #计算可以裁剪出的行数
    col_num = test_w // crop_lengh #计算可以裁剪出的列数

    if test_h%crop_lengh==0:
        row_num = row_num-1
    if test_w%crop_lengh==0:
        col_num = col_num-1

    crop_patch_volume = np.zeros((crop_lengh + crop_bound, crop_lengh + crop_bound, c,1), dtype=np.float32)  # [H,W,3,N]

    # left top
    for row_cp in range(row_num):
        for col_cp in range(col_num):
            crop_patch = in_lf_image[row_cp * crop_lengh:(row_cp + 1) * crop_lengh + crop_bound, col_cp * crop_lengh:(col_cp + 1) * crop_lengh + crop_bound]
            crop_patch = np.expand_dims(crop_patch, axis=-1)
            crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    h_bound_start = test_h - crop_lengh - crop_bound
    w_bound_start = test_w - crop_lengh - crop_bound

    # right
    for row_cp in range(row_num):
        crop_patch = in_lf_image[row_cp * crop_lengh:(row_cp + 1) * crop_lengh + crop_bound, w_bound_start:]
        crop_patch = np.expand_dims(crop_patch, axis=-1)
        crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    # bottom
    for col_cp in range(col_num):
        crop_patch = in_lf_image[h_bound_start:, col_cp * crop_lengh:(col_cp + 1) * crop_lengh + crop_bound]
        crop_patch = np.expand_dims(crop_patch, axis=-1)
        crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    # right bottom
    crop_patch = in_lf_image[h_bound_start:, w_bound_start:]
    crop_patch = np.expand_dims(crop_patch, axis=-1)
    crop_patch_volume = np.concatenate([crop_patch_volume, crop_patch], axis=-1)

    crop_patch_volume = crop_patch_volume[:, :,:,1:]
    return crop_patch_volume, row_num, col_num


def merge_patch(in_patch_volume_, rnum, cnum, overall_h, overall_w, an, spa_length, spa_bound, chan):

    h_bound = overall_h - spa_length * an * rnum
    w_bound = overall_w - spa_length * an * cnum

    spa_length = spa_length * an
    spa_bound = spa_bound * an
    spa_bound_sub = spa_bound // 2

    # left top
    rec_lf_img = np.zeros((overall_h, overall_w, chan)).astype(np.float32)
    pvx = 0
    for pvi in range(rnum):
        for pvj in range(cnum):
            if (pvi==0 and pvj==0):
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]

            elif (pvi == 0 and pvj > 0):
                in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch[:, spa_length:spa_length + spa_bound_sub, :]

            elif (pvi>0 and pvj==0):
                in_pre_patch = in_patch_volume_[:, :, :, pvx - cnum]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch[spa_length:spa_length + spa_bound_sub, :, :]

            else:
                in_pre_patch1 = in_patch_volume_[:, :, :, pvx - 1]
                in_pre_patch2 = in_patch_volume_[:, :, :, pvx - cnum]
                in_tmp_patch = in_patch_volume_[:, :, :, pvx]
                in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch1[:, spa_length:spa_length + spa_bound_sub, :]
                in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch2[spa_length:spa_length + spa_bound_sub, :, :]

            rec_lf_img[pvi*spa_length:(pvi+1)*spa_length, pvj*spa_length:(pvj+1)*spa_length, :] = in_tmp_patch[:spa_length, :spa_length, :]
            pvx = pvx + 1

    # right
    for pvk in range(rnum):
        if (pvk==0):
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
        else:
            in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
            in_tmp_patch[:spa_bound_sub, :, :] = in_pre_patch[spa_length:spa_length + spa_bound_sub, :, :]

        rec_lf_img[pvk*spa_length:(pvk+1)*spa_length, -w_bound:, :] = in_tmp_patch[:spa_length, -w_bound:, :]
        pvx = pvx + 1

    # bottom
    for pvl in range(cnum):
        if (pvl == 0):
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
        else:
            in_pre_patch = in_patch_volume_[:, :, :, pvx - 1]
            in_tmp_patch = in_patch_volume_[:, :, :, pvx]
            in_tmp_patch[:, :spa_bound_sub, :] = in_pre_patch[:, spa_length:spa_length + spa_bound_sub, :]

        rec_lf_img[-h_bound:, pvl*spa_length:(pvl+1)*spa_length, :] = in_tmp_patch[-h_bound:, :spa_length, :]
        pvx = pvx + 1

    # right bottom
    in_tmp_patch = in_patch_volume_[:, :, :, pvx]
    rec_lf_img[-h_bound:, -w_bound:, :] = in_tmp_patch[-h_bound:, -w_bound:, :]
    return rec_lf_img


def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]
    return Im_out


def LFdivide(lf, patch_size, stride):
    # [c,ah,aw,h,w]
    _, ah, aw, sai_h, sai_w = lf.shape
    data = rearrange(lf, 'c ah aw h w -> (ah aw) c h w')

    bdr = (patch_size - stride) // 2
    numU = (sai_h + bdr * 2 - 1) // stride
    numV = (sai_w + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(ah aw) (c h w) (n1 n2) -> n1 n2 ah aw c h w',
                      n1=numU, n2=numV, ah=ah, aw=aw, h=patch_size, w=patch_size)
    return subLF


def LFintegrate(subLFs, patch_size, stride, sai_h, sai_w):
    # [n1 n2,ah,aw,c,h,w]
    bdr = (patch_size - stride) // 2
    outLF = subLFs[:, :, :, :, :, bdr:bdr+stride, bdr:bdr+stride]
    outLF = rearrange(outLF, 'n1 n2 u v c h w -> c u v (n1 h) (n2 w)')
    outLF = outLF[:, :, :, 0:sai_h, 0:sai_w]
    return outLF


def ColorAugmentation(lfi, index):
    # [ah,aw,c,h,w]
    if index == 1:
        order = [0, 1, 2]
    elif index == 2:
        order = [0, 2, 1]
    elif index == 3:
        order = [1, 0, 2]
    elif index == 4:
        order = [1, 2, 0]
    elif index == 5:
        order = [2, 1, 0]
    else:
        order = [2, 0, 1]
    aug_lfi = lfi[:, :, order, :, :]
    return aug_lfi
