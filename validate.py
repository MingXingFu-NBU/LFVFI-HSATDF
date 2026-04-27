import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import imageio
from einops import rearrange
from datetime import datetime
from model.LFVFI_Model import LFFInet
from functions.utils import mk_dir, LFdivide, LFintegrate, cal_rgb_metrics, cal_y_metrics, cal_lpips_metrics, rgb_to_ycbcr, lfi2mlia
import lpips


################################################################################################
def run_validate(opt, validate_loader, epoch):

    print('==>validate')

    # output folder
    val_mlia_path = opt.val_path + 'MLIA/'
    mk_dir(val_mlia_path)

    # Pretrained model path
    pretrained_model_path = opt.model_dir + '/LFVFINet_epoch_' + str(epoch) + '.pth'
    if not os.path.exists(pretrained_model_path):
        print('Pretrained model folder is not found ')

    # Load pretrained weight
    checkpoints = torch.load(pretrained_model_path, map_location='cuda:0', weights_only=False)
    ckp_dict = checkpoints['model']

    ###########################################################################################
    # Build model
    print("Building LFFInet")
    model_validate = LFFInet(opt).to(opt.device)

    print('loaded model from ' + pretrained_model_path)
    model_validate_dict = model_validate.state_dict()
    ckp_dict_refine = {k: v for k, v in ckp_dict.items() if k in model_validate_dict}
    model_validate_dict.update(ckp_dict_refine)
    model_validate.load_state_dict(model_validate_dict)

    #######################################################################################
    # Test
    model_validate.eval()
    with torch.no_grad():

        ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3 = 0., 0.
        ave_y_psnr_dataset3, ave_y_ssim_dataset3 = 0., 0.

        ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4 = 0., 0.
        ave_y_psnr_dataset4, ave_y_ssim_dataset4 = 0., 0.

        ave_lpips_dataset3, ave_lpips_dataset4 = 0., 0.

        loss_fn = lpips.LPIPS(net='vgg', version='0.1').cuda()

        for idx_iter, (img0, img1, label) in tqdm(enumerate(validate_loader), total=len(validate_loader)):

            # torch.cuda.empty_cache()
            start_time = datetime.now()

            #######################################  Input data  #######################################
            # [b,c,u,v,h,w]
            sai_h = img0.shape[-2]
            sai_w = img0.shape[-1]

            if (opt.val_crop):
                ###################################  Forward inference  ###################################
                img0_sub_lfs = LFdivide(lf=img0[0], patch_size=opt.patch_size, stride=opt.patch_size//2)
                img1_sub_lfs = LFdivide(lf=img1[0], patch_size=opt.patch_size, stride=opt.patch_size//2)

                n1, n2, u, v, c, h, w = img0_sub_lfs.shape
                img0_sub_lfs = rearrange(img0_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')
                img1_sub_lfs = rearrange(img1_sub_lfs, 'n1 n2 u v c h w -> (n1 n2) c u v h w')

                infer_lf = []
                mini_batch = 8
                num_infer = (n1 * n2) // mini_batch
                for idx_infer in range(num_infer):
                    img0_lf_patch = img0_sub_lfs[idx_infer * mini_batch: (idx_infer + 1) * mini_batch, :, :, :, :, :]
                    img1_lf_patch = img1_sub_lfs[idx_infer * mini_batch: (idx_infer + 1) * mini_batch, :, :, :, :, :]
                    infer_lf_patch = model_validate(img0_lf_patch.to(opt.device), img1_lf_patch.to(opt.device))
                    infer_lf.append(infer_lf_patch)

                if (n1 * n2) % mini_batch:
                    img0_lf_patch = img0_sub_lfs[(idx_infer + 1) * mini_batch:, :, :, :, :, :]
                    img1_lf_patch = img1_sub_lfs[(idx_infer + 1) * mini_batch:, :, :, :, :, :]
                    infer_lf_patch = model_validate(img0_lf_patch.to(opt.device), img1_lf_patch.to(opt.device))
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
                    infer_lf = model_validate(img0.to(opt.device), img1.to(opt.device))
                infer_lf = infer_lf[0, :, :, :, :sai_h, :sai_w]

            elapsed_time = datetime.now() - start_time

            ####################################  Calculate metrics  ####################################
            # [c,u,v,h,w]
            infer_lf = infer_lf.squeeze(0)
            label_lf = label.squeeze(0).to(opt.device)

            infer_lf_y = rgb_to_ycbcr(rearrange(infer_lf, 'c u v h w -> c (u h) (v w)').unsqueeze(0))   # [b,c,uh,vw]
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
            save_name1 = '{}/epoch{}_scene{}_LF.png'.format(val_mlia_path, epoch, idx_iter + 1)
            imageio.imwrite(save_name1, (infer_lf_mlia.clip(0, 1) * 255.0).astype(np.uint8))

            val_sai_path = opt.val_path + 'epoch_' + str(epoch) + '/' + str(idx_iter + 1).zfill(3)
            mk_dir(val_sai_path)

            for an_u in range(opt.ang_res):
                for an_v in range(opt.ang_res):
                    infer_sai = rearrange(infer_lf[:, an_u, an_v, :, :], 'c h w -> h w c').cpu().numpy()
                    save_name = '{}/0{}_0{}.png'.format(val_sai_path, an_u, an_v)
                    imageio.imwrite(save_name, (infer_sai.clip(0, 1) * 255.0).astype(np.uint8))

        ###################################  Save evaluation results  ###################################
        print('Epoch: %d, Validate end!  Average metric: Dataset3 RGB PSNR: %s,  Dataset3 RGB SSIM: %s,  Dataset4 RGB PSNR: %s,  Dataset4 RGB SSIM: %s,'
              % (epoch, ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3, ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4))
        print('Average metric: Dataset3 Y PSNR : %s,  Dataset3 Y SSIM : %s,  Dataset4 Y PSNR: %s,  Dataset4 Y SSIM : %s,'
              % (ave_y_psnr_dataset3, ave_y_ssim_dataset3, ave_y_psnr_dataset4, ave_y_ssim_dataset4))
        print('Average metric: Dataset3 LPIPS : %s,  Dataset4 LPIPS : %s,' % (ave_lpips_dataset3, ave_lpips_dataset4))

        file_handle = open(opt.val_path + 'quality_score.txt', mode='a')
        file_handle.write('Epoch: %d, Average,  Dataset3 RGB PSNR: %s,  Dataset3 RGB SSIM: %s, Dataset4 RGB PSNR: %s,  Dataset4 RGB SSIM: %s\n'
                          'Dataset3 Y PSNR: %s,  Dataset3 Y SSIM: %s, Dataset4 Y PSNR: %s,  Dataset4 Y SSIM: %s\n'
                          'Dataset3 LPIPS: %s,  Dataset4 LPIPS: %s\n'
                          % (epoch, ave_rgb_psnr_dataset3, ave_rgb_ssim_dataset3, ave_rgb_psnr_dataset4, ave_rgb_ssim_dataset4,
                             ave_y_psnr_dataset3, ave_y_ssim_dataset3, ave_y_psnr_dataset4, ave_y_ssim_dataset4,
                             ave_lpips_dataset3, ave_lpips_dataset4))
        file_handle.close()










