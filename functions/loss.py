import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from math import exp
from functions import Vgg19


# ssim functions
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = functional.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = functional.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = functional.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = functional.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = functional.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



#################### Loss functions ####################
class cal_pixel_loss(nn.Module):
    def __init__(self):
        super(cal_pixel_loss, self).__init__()
        self.loss = torch.nn.L1Loss()

    def forward(self, infer, gt):
        pixel_loss = self.loss(infer, gt)    # [b,c,h,w]
        return pixel_loss


class cal_ssim_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(cal_ssim_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channels = 3
        self.window = create_window(window_size, self.channels)

    def forward(self, infer, gt):
        # [b,3,h,w]
        channels = infer.shape[1]
        if channels == self.channels and self.window.data.type() == infer.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channels)

            if infer.is_cuda:
                window = window.cuda(infer.get_device())
            window = window.type_as(infer)

            self.window = window
            self.channels = channels

        ssim_value = _ssim(infer, gt, window, self.window_size, channels, self.size_average)
        ssim_loss = 1.0 - ssim_value
        return ssim_loss


class cal_perceptual_loss(nn.Module):
    def __init__(self, device):
        super(cal_perceptual_loss, self).__init__()
        self.loss = torch.nn.L1Loss()
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(device)

    def forward(self, infer, gt):
        # [b,3,h,w]
        infer_feats = self.vgg19(infer)
        gt_feats = self.vgg19(gt)
        perceptual_loss = (self.loss(infer_feats[0], gt_feats[0])/2.6 +
                           self.loss(infer_feats[1], gt_feats[1])/4.8 +
                           self.loss(infer_feats[2], gt_feats[2])/3.7 +
                           self.loss(infer_feats[3], gt_feats[3])/5.6 +
                           self.loss(infer_feats[4], gt_feats[4])*10/1.5)
        return perceptual_loss


def get_loss(device):
    losses = {}
    losses['pixel_loss'] = cal_pixel_loss()
    losses['ssim_loss'] = cal_ssim_loss()
    losses['perceptual_loss'] = cal_perceptual_loss(device)
    return losses