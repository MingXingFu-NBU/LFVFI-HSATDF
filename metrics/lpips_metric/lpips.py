# 计算LPIPS
import argparse
import os
import numpy as np
import torch

import lpips

def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img1 = bgr2ycbcr(img, y_only=False)
        y_channel = img1[:, :, 0]  # 获取 Y 通道的值
        img = np.stack((y_channel, y_channel, y_channel), axis=-1)
        # img = img[..., None]
    return img


def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, ' f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, ' f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)



# SRDIFF
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-d0','--dir0', type=str, default=r'F:\database\lau\odisr\testing\HR/') #  odi
parser.add_argument('-d0','--dir0', type=str, default=r'F:\database\lau\sun_test\HR/') #sun

# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\0-bicubic\ODI\X4')  # lau-net
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\1-EDSR\ODI\X4/')  # my
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_r esolution or restoration\compara_data\1-RCAN\ODI\X4/')  # bicubic
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\2-SWIN\SUN\X16/')
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\3-ESRGAN\ODI\X4/')
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\3-SRDIFF\ODI\X8/')
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\4-360SS\ODI\X4/')
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\4-LAU-Net\ODI\X4/')
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\2super_resolution or restoration\compara_data\4-OSRT\ODI\X2')  # my
# parser.add_argument('-d1','--dir1', type=str, default=r'F:\database\lau\odisr\testing\LR_bicubic\X2')  # my 0.4443 0.4361
parser.add_argument('-d1','--dir1', type=str, default=r'F:\Postgraduate\paper_code\1OI_super_resolution\paper4\base\results\x4')  # my 0.4443 0.4361
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# Initializing the model
loss_fn = lpips.LPIPS(net='vgg', version=opt.version)
if (opt.use_gpu):
    loss_fn.cuda()

# crawl directories
files = os.listdir(opt.dir0)
cumulative_lpips = 0

# RGB
# i=1
# for file in files:
#     if (os.path.exists(os.path.join(opt.dir0, file))):
#         print(i)
#         # Load images
#         file1 = file.split('.')[0]
#         file_png = f"{file1}.png"
#         img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))  # RGB image from [-1,1]
#         img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file_png)))
#
#         if (opt.use_gpu):
#             img0 = img0.cuda()
#             img1 = img1.cuda()
#         # Compute distance
#         dist01 = loss_fn.forward(img0, img1)
#         print('%.4f' % (dist01))
#         dist = dist01
#         cumulative_lpips += dist01.item()
#         i = i + 1
#     #     # 手动释放内存
#     #     del dist01
#     #
#     # 清空GPU内存
#     if (opt.use_gpu):
#         torch.cuda.empty_cache()
# print('Testing set, LPIPS is %.4f' % (cumulative_lpips / len(files)))


# Y
i=1
for file in files:
    if (os.path.exists(os.path.join(opt.dir0, file))):
        print(i)
        # Load images
        file1 = file.split('.')[0]
        file_png = f"{file1}.png"
        img0 = lpips.load_image(os.path.join(opt.dir0, file))
        img0Y = to_y_channel(img0) * 255
        img0 = lpips.im2tensor(img0Y)  # RGB image from [-1,1]

        img1 = lpips.load_image(os.path.join(opt.dir1, file_png))
        img1Y = to_y_channel(img1) * 255
        img1 = lpips.im2tensor(img1Y)

        if (opt.use_gpu):
            img0 = img0.cuda()
            img1 = img1.cuda()
        # Compute distance
        dist01 = loss_fn.forward(img0, img1)
        print('%.4f' % (dist01))
        dist = dist01
        cumulative_lpips += dist01.item()
        i = i + 1
    #     # 手动释放内存
    #     del dist01
    #
    # 清空GPU内存
    if (opt.use_gpu):
        torch.cuda.empty_cache()
print('Testing set, LPIPS is %.4f' % (cumulative_lpips / len(files)))


