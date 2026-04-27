import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from einops import rearrange
from datetime import datetime
import imageio
import cv2
from model.LFVFI_Model import LFFInet
from functions.dataset import TestSetLoader
from functions.utils import LFdivide, LFintegrate, cal_rgb_metrics, cal_y_metrics, rgb_to_ycbcr, cal_lpips_metrics, lfi2mlia, mk_dir
import lpips


#########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--model_dir", type=str, default="F:/work2/Ablation/LFVFI_master/1/checkpoints/", help="Checkpoints path")
parser.add_argument("--testset_path", type=str, default="E:/work2/dataset_for2D/validation", help="Test data path")
parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")
parser.add_argument("--patch_size", type=int, default=128, help="Cropped LF patch size, i.e., spatial resolution")
parser.add_argument("--output_path", type=str, default="F:/work2/Ablation/LFVFI_master/outputs/", help="Testing result path")
parser.add_argument("--test_crop", type=int, default=0, help="Cropping image patches during testing")
parser.add_argument("--mini_batch", type=int, default=8, help="Mini batch during testing (crop)")
parser.add_argument("--channel", type=int, default=64, help="Number of feature channels")
parser.add_argument("--down_sample", type=int, default=4, help="Downsampling factor")

args = parser.parse_args()
print(args)

#########################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 设置设备为单卡
torch.cuda.set_device(device)

################################################################################################
def test(opt, test_loader):

    print('==>testing')
    # Pretrained model path
    pretrained_model_path = opt.model_dir + '/LFVFINet_epoch_54.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path, map_location='cuda:0', weights_only=False)
    ckp_dict = checkpoints['model']

    ###########################################################################################
    # Build model
    print("Building LFFInet")
    model_test = LFFInet(opt).to(opt.device)

    print('loaded model from ' + pretrained_model_path)
    model_test_dict = model_test.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_test_dict}
    model_test_dict.update(ckp_dict_refine)
    model_test.load_state_dict(model_test_dict)

    # output folder
    mk_dir(opt.output_path)

    #######################################################################################
    # Test
    model_test.eval()
    with torch.no_grad():

        ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3 = 0., 0.
        ave_y_psnr_dataset3, ave_y_ssim_dataset3 = 0., 0.

        ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4 = 0., 0.
        ave_y_psnr_dataset4, ave_y_ssim_dataset4 = 0., 0.

        ave_lpips_dataset3, ave_lpips_dataset4 = 0., 0.

        loss_fn = lpips.LPIPS(net='vgg', version='0.1').cuda()

        for idx_iter, (img0, img1, label) in tqdm(enumerate(test_loader), total=len(test_loader)):

            # torch.cuda.empty_cache()
            start_time = datetime.now()
            #######################################  Input data  #######################################
            # [b,c,u,v,h,w]
            sai_h = img0.shape[-2]
            sai_w = img0.shape[-1]

            if (opt.test_crop):
                ###################################  Forward inference  ###################################
                img0_sub_lfs = LFdivide(lf=img0[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
                img1_sub_lfs = LFdivide(lf=img1[0], patch_size=opt.patch_size, stride=opt.patch_size//2)

                n1, n2, u, v, c, h, w = img0_sub_lfs.shape
                img0_sub_lfs = rearrange(img0_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')
                img1_sub_lfs = rearrange(img1_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')

                infer_lf = []
                num_infer = (n1 * n2) // opt.mini_batch
                for idx_infer in range(num_infer):
                    img0_lf_patch = img0_sub_lfs[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    img1_lf_patch = img1_sub_lfs[idx_infer * opt.mini_batch: (idx_infer + 1) * opt.mini_batch, :, :, :, :, :]
                    infer_lf_patch = model_test(img0_lf_patch.to(opt.device), img1_lf_patch.to(opt.device))
                    infer_lf.append(infer_lf_patch)

                if (n1 * n2) % opt.mini_batch:
                    img0_lf_patch = img0_sub_lfs[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                    img1_lf_patch = img1_sub_lfs[(idx_infer + 1) * opt.mini_batch:, :, :, :, :, :]
                    infer_lf_patch = model_test(img0_lf_patch.to(opt.device), img1_lf_patch.to(opt.device))
                    infer_lf.append(infer_lf_patch)

                infer_lf = torch.cat(infer_lf, dim=0)
                infer_lf = rearrange(infer_lf, '(n1 n2) c u v h w -> n1 n2 u v c h w', n1=n1, n2=n2)
                infer_lf = LFintegrate(infer_lf,  patch_size=opt.patch_size, stride=opt.patch_size//2, sai_h=sai_h, sai_w=sai_w)

            else:
                ph = ((sai_h - 1) // opt.down_sample + 1) * opt.down_sample
                pw = ((sai_w - 1) // opt.down_sample + 1) * opt.down_sample
                padding = (0, pw - sai_w, 0, ph - sai_h)
                img0 = F.pad(img0, padding)
                img1 = F.pad(img1, padding)
                with torch.set_grad_enabled(False):
                    infer_lf = model_test(img0.to(opt.device), img1.to(opt.device))
                infer_lf = infer_lf[0, :, :, :, :sai_h, :sai_w]

            elapsed_time = datetime.now() - start_time

            ####################################  Calculate metrics  ####################################
            # [c,u,v,h,w]
            infer_lf = infer_lf.squeeze(0)
            label_lf = label.squeeze(0).to(opt.device)

            infer_lf_y = rgb_to_ycbcr(rearrange(infer_lf, 'c u v h w -> c (u h) (v w)').unsqueeze(0))  # [b,c,uh,vw]
            label_lf_y = rgb_to_ycbcr(rearrange(label_lf, 'c u v h w -> c (u h) (v w)').unsqueeze(0))
            infer_lf_y = rearrange(infer_lf_y, '1 c (u h) (v w) -> 1 c u v h w', h=sai_h, w=sai_w).squeeze(0)
            label_lf_y = rearrange(label_lf_y, '1 c (u h) (v w) -> 1 c u v h w', h=sai_h, w=sai_w).squeeze(0)

            rgb_psnr, rgb_ssim = cal_rgb_metrics(infer_lf, label_lf)
            y_psnr, y_ssim = cal_y_metrics(infer_lf_y[0], label_lf_y[0])
            rgb_lpips = cal_lpips_metrics(infer_lf, label_lf, loss_fn)

            print('Test image.%d,  RGB PSNR: %s,  RGB SSIM: %s,  Y PSNR: %s,  Y SSIM: %s,  LPIPS: %s, Elapsed time: %s'
                  % (idx_iter + 1, rgb_psnr, rgb_ssim, y_psnr, y_ssim, rgb_lpips, elapsed_time))

            if (idx_iter < 6):
                ave_rgb_psnr_dataset3 += rgb_psnr / 6.
                ave_rgb_ssim_dataset3 += rgb_ssim / 6.
                ave_y_psnr_dataset3 += y_psnr / 6.
                ave_y_ssim_dataset3 += y_ssim / 6.
                ave_lpips_dataset3 += rgb_lpips / 6
            else:
                ave_rgb_psnr_dataset4 += rgb_psnr / 60.
                ave_rgb_ssim_dataset4 += rgb_ssim / 60.
                ave_y_psnr_dataset4 += y_psnr / 60.
                ave_y_ssim_dataset4 += y_ssim / 60.
                ave_lpips_dataset4 += rgb_lpips / 60

            ###################################  Save inference results  ###################################
            infer_lf_mlia = lfi2mlia(infer_lf).cpu().numpy()
            save_name = '{}/Infer_scene{}.png'.format(opt.output_path, idx_iter + 1)
            imageio.imwrite(save_name, (infer_lf_mlia.clip(0, 1) * 255.0).astype(np.uint8))

            test_sai_path = opt.output_path + str(idx_iter + 1).zfill(3)
            mk_dir(test_sai_path)

            for an_u in range(opt.ang_res):
                for an_v in range(opt.ang_res):
                    infer_sai = rearrange(infer_lf[:, an_u, an_v, :, :], 'c h w -> h w c').cpu().numpy()
                    save_name_sai = '{}/0{}_0{}.png'.format(test_sai_path, an_u, an_v)
                    imageio.imwrite(save_name_sai, (infer_sai.clip(0, 1) * 255.0).astype(np.uint8))

        ###################################  Save evaluation results  ###################################
        print('Validate end!  Average metric: Dataset3 RGB PSNR: %s,  Dataset3 RGB SSIM: %s,  Dataset4 RGB PSNR: %s,  Dataset4 RGB SSIM: %s,'
              % (ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3, ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4))
        print('Average metric: Dataset3 Y PSNR : %s,  Dataset3 Y SSIM : %s,  Dataset4 Y PSNR: %s,  Dataset4 Y SSIM : %s,'
              % (ave_y_psnr_dataset3, ave_y_ssim_dataset3, ave_y_psnr_dataset4, ave_y_ssim_dataset4))
        print('Average metric: Dataset3 LPIPS : %s,  Dataset4 LPIPS : %s,' % (ave_lpips_dataset3, ave_lpips_dataset4))

        file_handle = open(opt.output_path + 'quality_score.txt', mode='a')
        file_handle.write('Average,  Dataset3 RGB PSNR: %s,  Dataset3 RGB SSIM: %s, Dataset4 RGB PSNR: %s,  Dataset4 RGB SSIM: %s\n'
                          'Dataset3 Y PSNR: %s,  Dataset3 Y SSIM: %s, Dataset4 Y PSNR: %s,  Dataset4 Y SSIM: %s\n'
                          'Dataset3 LPIPS: %s,  Dataset4 LPIPS: %s\n'
                          % (ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3, ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4,
                             ave_y_psnr_dataset3, ave_y_ssim_dataset3, ave_y_psnr_dataset4, ave_y_ssim_dataset4,
                             ave_lpips_dataset3, ave_lpips_dataset4))
        file_handle.close()


######################################## main ########################################
def main(opt):
    print('\nload testing Dataset ...')
    test_set = TestSetLoader(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} test LF image from {}'.format(len(test_loader), opt.testset_path))

    test(opt, test_loader)


########################################################
if __name__ == "__main__":
    main(args)










