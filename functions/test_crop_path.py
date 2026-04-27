import argparse
import logging
import os
import sys
import time
import copy
import shutil
import random

from lpips import lpips
from torch import optim

from model.LFVFI_Model import LFVFI_3DCNN_CA
import cv2
import torch
import numpy as np
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import config
from torch.optim import AdamW
import utils
from dataset import trainLightFieldDataset_h5_LFVFI,testLightFieldDataset_LFVFI
from torch.utils.data import DataLoader
from model.utils import crop_lf_patch,merge_lf_patch,cal_metrics,save_checkpoint,mk_dir,rgb_to_ycbcr
from loss import *
##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)
logging.basicConfig(
    filename='LFVFI_test_crop_patch.log',  # 日志文件名
    level=logging.INFO,      # 日志级别
    format='%(asctime)s-\n%(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S'          # 时间格式
)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=300, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--max_epoch', default=200, type=int)
parser.add_argument('--mode', type=str,default='train')
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")

parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--step_per_epoch', default=3382, type=int, help='local rank')
parser.add_argument("--trainset_path", type=str,
                    default="E:\\work2\\LF_Vidio_code\\LFVFI\\dataset\\train\\",
                    help="Training data path")
parser.add_argument("--testset_path", type=str, default="E:\\work2\dataset_for2D\\validation\\",
                    help="Test data path")
parser.add_argument("--model_dir", type=str, default="checkpoints/LFVFI_CA_3DCNN/", help="Checkpoints path")
parser.add_argument("--path_pre_pth", type=str, default='checkpoints/LFVFI_CA_3DCNN/LFVFI_CA50.pth.tar', help="pre path")
args = parser.parse_args()
# 设置设备为单卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
log_path = 'train_log'



def test(args, test_loader,model,eval_alpha=0.5):
    # print('Evaluating for epoch = %d' % epoch)
    # losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    lpips_model = lpips.LPIPS(net="alex")
    lpips_model.eval()
    LPIPS_dataset3 = []
    LPIPS_iter_test_dataset3 = {i: 0 for i in range(1, 7)}
    LPIPS_dataset4 = []
    LPIPS_iter_test_dataset4 = {i: 0 for i in range(1, 61)}

    lfname = ['dataset3_3frame', 'dataset4_3frame']

    Y_dataset3_psnr_name=[]
    Y_dataset3_ssim_name=[]
    Y_psnr_dataset3 = {i: 0 for i in range(1, 7)}
    Y_ssim_dataset3 = {i: 0 for i in range(1, 7)}

    Y_dataset4_psnr_name=[]
    Y_dataset4_ssim_name=[]
    Y_psnr_dataset4 = {i: 0 for i in range(1, 61)}
    Y_ssim_dataset4 = {i: 0 for i in range(1, 61)}

    dataset3_psnr_name=[]
    dataset3_ssim_name=[]
    psnr_dataset3 = {i: 0 for i in range(1, 7)}
    ssim_dataset3 = {i: 0 for i in range(1, 7)}

    dataset4_psnr_name=[]
    dataset4_ssim_name=[]
    psnr_dataset4 = {i: 0 for i in range(1, 61)}
    ssim_dataset4 = {i: 0 for i in range(1, 61)}
    s=1
    j=1
    # dataset4_psnr_name=[]
    # dataset4_ssim_name=[]
    # psnr_dataset4 = {i: 0 for i in range(1, 61)}
    # ssim_dataset4 = {i: 0 for i in range(1, 61)}
    # j=1
    t = time.time()
    with torch.no_grad():
        with tqdm(test_loader, dynamic_ncols=True) as tqdmDataLoader:
            for data,timestep, dataset_name, h, w in tqdmDataLoader:
                if dataset_name[0] == lfname[0]:
                    mk_dir('testing_results/%s/%d/' % (dataset_name[0], s))
                elif dataset_name[0] == lfname[1]:
                    mk_dir('testing_results/%s/%d/' % (dataset_name[0], j))
                LPIPS_view_test = {i: None for i in range(0, 26)}
                lf = np.zeros((1, h, w, 5, 5, 3))  #
                subinput, row_nows, col_nums = crop_lf_patch(data[0],124,4)
                subinput = torch.tensor(subinput, dtype=torch.float)
                subinput = subinput.cpu()
                patch_num = subinput.shape[0]  # [256,256,3,num]
                subout_fusion = torch.zeros([1,3,5,5,128, 128]).to(args.device)
                with torch.set_grad_enabled(False):
                    for i in range(0, patch_num):
                        inputs_patch = subinput[i]
                        out,feat = model(inputs_patch[:,:3].to(args.device),inputs_patch[:,3:6].to(args.device))
                        subout_fusion = torch.cat((subout_fusion, out), dim=0)
                        del inputs_patch, out
                        if i % 10 == 0:  # 每 10 个 patch 回收一次，可调
                            torch.cuda.empty_cache()
                sr_view = subout_fusion[1:].unsqueeze(1)
                output_SAI_patch = merge_lf_patch(sr_view, row_nows, col_nums, h.item(), w.item(), 124, 4)
                for u in range(0, 5):
                    for v in range(0, 5):
                        output = output_SAI_patch[:, :, u, v, :, :]
                        target = data[0,:,6:9,u, v, :, :].to(args.device)
                        distance = lpips_model((output.detach().cpu()),
                                               (target.detach().cpu()))
                        LPIPS_view_test[v * 5 + u] = distance.item()
                        if dataset_name[0] == lfname[0]:
                            out_img = cv2.cvtColor(output[0].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                            cv2.imwrite('testing_results/%s/%d/input_%d_%d.png' % (dataset_name[0], s, u, v),
                                        ((out_img) * 255).astype(np.uint8))
                            gt_img = cv2.cvtColor(target[0].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                            cv2.imwrite('testing_results/%s/%d/target_%d_%d.png' % (dataset_name[0], s, u, v),
                                        ((gt_img) * 255).astype(np.uint8))
                        elif dataset_name[0] == lfname[1]:
                            out_img = cv2.cvtColor(output[0].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                            cv2.imwrite('testing_results/%s/%d/input_%d_%d.png' % (dataset_name[0], j, u, v),
                                        ((out_img) * 255).astype(np.uint8))
                            gt_img = cv2.cvtColor(target[0].permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                            cv2.imwrite('testing_results/%s/%d/target_%d_%d.png' % (dataset_name[0], j, u, v),
                                        ((gt_img) * 255).astype(np.uint8))

                output_lf = rearrange(output_SAI_patch, '1 c an1 an2 h w  ->1 c  (an1 h) (an2 w)', h=h.item(),w=w.item(), an1=5, an2=5, c=3)
                gt_lf = data.squeeze(1)
                gt_lf = rearrange(gt_lf[:,6:9], '1 c an1 an2 h w->1 c  (an1 h) (an2 w)', h=h.item(),w=w.item(), an1=5, an2=5, c=3)
                Y_output_lf=rgb_to_ycbcr(output_lf)
                Y_gt_lf = rgb_to_ycbcr(gt_lf)
                y_pred = Y_output_lf[:, 0:1, :, :]  # (B,1,H,W)
                y_gt = Y_gt_lf[:, 0:1, :, :]
                Y_psnr, Y_ssim = cal_metrics(5, y_pred, y_gt)
                sr_psnr, sr_ssim = cal_metrics(5, output_lf, gt_lf)

                if dataset_name[0] == lfname[0]:
                    psnr_dataset3[s] = sr_psnr
                    ssim_dataset3[s] = sr_ssim
                    Y_psnr_dataset3[s] = Y_psnr
                    Y_ssim_dataset3[s] = Y_ssim
                    LPIPS_iter_test_dataset3[s] = sum(LPIPS_view_test[i] for i in range(0, 25)) / 25.0
                    s += 1
                elif dataset_name[0] == lfname[1]:
                    psnr_dataset4[j] = sr_psnr
                    ssim_dataset4[j] = sr_ssim
                    Y_psnr_dataset4[j] = Y_psnr
                    Y_ssim_dataset4[j] = Y_ssim
                    LPIPS_iter_test_dataset4[j] = sum(LPIPS_view_test[i] for i in range(0, 25)) / 25.0
                    j += 1
                tqdmDataLoader.set_postfix(ordered_dict={
                    "indx:": j,
                    "PSNR: ": sr_psnr,
                    "SSIM: ": sr_ssim,
                    "Y_PSNR: ": Y_psnr,
                    "Y_SSIM: ": Y_ssim,
                    "LPIPS": sum(LPIPS_view_test[i] for i in range(0, 25)) / 25.0,
                })
            Y_dataset3_psnr_name.append(sum(Y_psnr_dataset3[i] for i in range(1, 7)) / 6.0)
            Y_dataset4_psnr_name.append(sum(Y_psnr_dataset4[i] for i in range(1, 61)) / 60.0)

            Y_dataset3_ssim_name.append(sum(Y_ssim_dataset3[i] for i in range(1, 7)) / 6.0)
            Y_dataset4_ssim_name.append(sum(Y_ssim_dataset4[i] for i in range(1, 61)) / 60.0)

            dataset3_psnr_name.append(sum(psnr_dataset3[i] for i in range(1, 7)) / 6.0)
            dataset4_psnr_name.append(sum(psnr_dataset4[i] for i in range(1, 61)) / 60.0)

            dataset3_ssim_name.append(sum(ssim_dataset3[i] for i in range(1, 7)) / 6.0)
            dataset4_ssim_name.append(sum(ssim_dataset4[i] for i in range(1, 61)) / 60.0)

            LPIPS_dataset3.append(sum(LPIPS_iter_test_dataset3[i] for i in range(1, 7)) / 6.0)
            LPIPS_dataset4.append(sum(LPIPS_iter_test_dataset4[i] for i in range(1, 61)) / 60.0)
            return (lfname,
                    dataset3_psnr_name, dataset3_ssim_name,
                    dataset4_psnr_name, dataset4_ssim_name,
                    Y_dataset3_psnr_name, Y_dataset3_ssim_name,
                    Y_dataset4_psnr_name, Y_dataset4_ssim_name,
                    LPIPS_dataset3, LPIPS_dataset4,
                    )


""" Entry Point """
def main(args):


    '''DATA  testing LOADING'''
    print('\nload testing Dataset ...')
    test_set = testLightFieldDataset_LFVFI(args)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=True)
    print('Loaded {} training image from {}'.format(len(test_loader), args.testset_path))


    model = LFVFI_3DCNN_CA().to(args.device)
    pre_path = args.path_pre_pth
    checkpoint = torch.load(pre_path)
    model.load_state_dict(checkpoint['state_dict'])



    best_psnr = 0



    lfname, dataset3_psnr_name, dataset3_ssim_name, dataset4_psnr_name, dataset4_ssim_name, Y_dataset3_psnr_name, Y_dataset3_ssim_name, Y_dataset4_psnr_name, Y_dataset4_ssim_name, LPIPS_dataset3, LPIPS_dataset4= test(args,test_loader,model,eval_alpha=0.5)
    log_fusion_print = f"""
    {lfname[0]:<10}: dataset3_PSNR:{dataset3_psnr_name[0]:.2f} dB  dataset3_SSIM:{dataset3_ssim_name[0]:.3f}
    {lfname[1]:<10}: dataset4_PSNR:{dataset4_psnr_name[0]:.2f} dB  dataset4_SSIM:{dataset4_ssim_name[0]:.3f}
    {lfname[0]:<10}: dataset3_LPIPS:{LPIPS_dataset3[0]:.3f}  

    {lfname[1]:<10}: dataset4_Y_PSNR:{Y_dataset4_psnr_name[0]:.2f} dB  dataset4_Y_SSIM:{Y_dataset4_ssim_name[0]:.3f}
    {lfname[0]:<10}: dataset3_Y_PSNR:{Y_dataset3_psnr_name[0]:.2f} dB  dataset3_Y_SSIM:{Y_dataset3_ssim_name[0]:.3f}
    {lfname[1]:<10}: dataset4_LPIPS:{LPIPS_dataset4[0]:.3f} 
    """
    print(log_fusion_print)
    epoch_info = (
                 f"""
    {lfname[0]:<10}: dataset3_PSNR:{dataset3_psnr_name[0]:.2f} dB  dataset3_SSIM:{dataset3_ssim_name[0]:.3f}
    {lfname[1]:<10}: dataset4_PSNR:{dataset4_psnr_name[0]:.2f} dB  dataset4_SSIM:{dataset4_ssim_name[0]:.3f}
    {lfname[0]:<10}: dataset3_LPIPS:{LPIPS_dataset3[0]:.3f}  

    {lfname[1]:<10}: dataset4_Y_PSNR:{Y_dataset4_psnr_name[0]:.2f} dB  dataset4_Y_SSIM:{Y_dataset4_ssim_name[0]:.3f}
    {lfname[0]:<10}: dataset3_Y_PSNR:{Y_dataset3_psnr_name[0]:.2f} dB  dataset3_Y_SSIM:{Y_dataset3_ssim_name[0]:.3f}
    {lfname[1]:<10}: dataset4_LPIPS:{LPIPS_dataset4[0]:.3f} 
                """)
    logging.info(epoch_info)
if __name__ == "__main__":
    main(args)
