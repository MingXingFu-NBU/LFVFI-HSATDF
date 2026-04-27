import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import argparse
import logging
import os
from os.path import join
from einops import rearrange
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import imageio
from functions.dataset import TrainSetLoader, TestSetLoader
from functions.utils import mk_dir, to_2d, lfi2mlia
from functions.loss import get_loss
from validate import run_validate
from model.LFVFI_Model import LFFInet
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


#########################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda:0', help="GPU setting")
parser.add_argument('--train_epoch', type=int, default=60, help="Number of epochs to train")
parser.add_argument('--batch_size', type=int, default=1, help="Training batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--n_steps", type=int, default=15, help="Number of epochs to update learning rate")
parser.add_argument("--decay_value", type=float, default=0.5, help="Learning rate decaying factor")
parser.add_argument("--model_dir", type=str, default="checkpoints/", help="Checkpoints path")
parser.add_argument("--val_path", type=str, default="results_val/", help="Validation result path")
parser.add_argument("--trainset_path", type=str, default="F:/Final_version/dataset/train", help="Training data path")
parser.add_argument("--testset_path", type=str, default="F:/Final_version/dataset/validation", help="Test data path")
parser.add_argument("--ang_res", type=int, default=5, help="Angular resolution of light field")
parser.add_argument("--patch_size", type=int, default=128, help="Cropped LF patch size, i.e., spatial resolution")
parser.add_argument("--resume_epoch", type=int, default=0, help="Resume from checkpoint epoch")
parser.add_argument("--train_save", type=int, default=0, help="Save the image in training")
parser.add_argument("--val_crop", type=int, default=0, help="Cropping image patches during validation")
parser.add_argument("--channel", type=int, default=64, help="Number of feature channels")
parser.add_argument("--down_sample", type=int, default=4, help="Downsampling factor")

args = parser.parse_args()
print(args)

#########################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # 设置设备为单卡
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True

#####################################################################################################
# SEED = 1234
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# np.random.seed(SEED)

# Loss functions
cal_loss = get_loss(device)

######################################### Train #########################################
def train(opt, train_loader, test_loader):

    print('==>training')
    start_time = datetime.now()

    train_results_dir = 'training_results/'
    if opt.train_save:
        mk_dir(train_results_dir)

    # model save folder
    mk_dir(opt.model_dir)

    #######################################################################################
    # Build model
    print("Building LFFInet")
    model_train = LFFInet(opt).to(opt.device)

    total = sum([param.nelement() for param in model_train.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))      # 打印模型参数量

    #######################################################################################
    # Optimizer and loss logger
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.n_steps, gamma=opt.decay_value)
    losslogger = defaultdict(list)

    #######################################################################################
    if opt.resume_epoch:
        resume_path = join(opt.model_dir, 'LFVFINet_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>Loading model parameters '{}'".format(resume_path))
            checkpoints = torch.load(resume_path, map_location={'cuda:0': opt.device}, weights_only=False)
            model_train.load_state_dict(checkpoints['model'])
            optimizer.load_state_dict(checkpoints['optimizer'])
            scheduler.load_state_dict(checkpoints['scheduler'])
            losslogger = checkpoints['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    #######################################################################################
    # start training
    epoch_state = opt.resume_epoch + 1
    for idx_epoch in range(epoch_state, opt.train_epoch + 1):   # epochs
        model_train.train()
        print('Current epoch: %d, learning rate: %e' % (idx_epoch, optimizer.state_dict()['param_groups'][0]['lr']))

        loss_epoch = 0.   # Total loss per epoch
        for idx_iter, (img0, img1, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # [b,c,u,v,h,w]
            img0, img1, label = img0.to(opt.device), img1.to(opt.device), label.to(opt.device)

            ############################  Forward inference  ############################
            pred = model_train(img0, img1)

            #############################  Calculate loss  #############################
            l1_loss = cal_loss['pixel_loss'](infer=to_2d(pred), gt=to_2d(label))
            ssim_loss = cal_loss['ssim_loss'](infer=to_2d(pred), gt=to_2d(label)) * 0.15
            vgg_loss = cal_loss['perceptual_loss'](infer=to_2d(pred), gt=to_2d(label)) * 0.03
            loss = l1_loss + ssim_loss + vgg_loss

            # print(l1_loss.item(), ssim_loss.item(), vgg_loss.item())

            # Cumulative loss
            loss_epoch += loss.item()

            ###########################  Backward and optimize  ###########################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###########################  Save training results  ###########################
            if opt.train_save:
                if idx_iter % 1000 == 0 or idx_iter == len(train_loader) - 1:
                    in_name1 = '{}/epoch{}_iter{}_input1.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    in_name2 = '{}/epoch{}_iter{}_input2.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    infer_name = '{}/epoch{}_iter{}_infer.jpg'.format(train_results_dir, idx_epoch, idx_iter)
                    label_name = '{}/epoch{}_iter{}_label.jpg'.format(train_results_dir, idx_epoch, idx_iter)

                    # [c,ah,aw,h,w]
                    save_in1 = (img0[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_in2 = (img1[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_infer = (pred[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)
                    save_label = (label[0, :, :, :, :, :].detach().cpu().numpy().clip(0, 1) * 255.0)

                    # [3,ah,aw,h,w] --> [h*ah,w*aw,3]
                    imageio.imwrite(in_name1, lfi2mlia(save_in1).astype(np.uint8))
                    imageio.imwrite(in_name2, lfi2mlia(save_in2).astype(np.uint8))
                    imageio.imwrite(infer_name, lfi2mlia(save_infer).astype(np.uint8))
                    imageio.imwrite(label_name, lfi2mlia(save_label).astype(np.uint8))

        ################################ Update learning rate ###############################
        scheduler.step()

        ####################################  Print loss  ####################################
        losslogger['epoch'].append(idx_epoch)
        losslogger['loss'].append(loss_epoch / len(train_loader))
        elapsed_time = datetime.now() - start_time
        print('Training==>>Epoch: %d,  loss: %s,  elapsed time: %s'
              % (idx_epoch, loss_epoch / len(train_loader), elapsed_time))

        # write loss
        file_handle = open('loss.txt', mode='a')
        file_handle.write('epoch: %d,  loss: %s,  elapsed time: %s\n'
                          % (idx_epoch, loss_epoch / len(train_loader), elapsed_time))
        file_handle.close()

        # save trained model's parameters
        if idx_epoch % 1 == 0:
            model_save_path = join(opt.model_dir, "LFVFINet_epoch_{}.pth".format(idx_epoch))
            state = {'epoch': idx_epoch, 'model': model_train.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger}
            torch.save(state, model_save_path)
            print("checkpoints saved to {}".format(model_save_path))

        # save loss figure
        if idx_epoch % 1 == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(opt.model_dir + "loss.png")
            plt.close('all')

        ################################ Validate ################################
        if idx_epoch % 1 == 0 or idx_epoch == 1:
            run_validate(opt, test_loader, idx_epoch)


######################################## main ########################################
def main(opt):
    print('\nload Training Dataset ...')
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('Loaded {} training LF image from {}'.format(len(train_loader), opt.trainset_path))

    print('\nload testing Dataset ...')
    test_set = TestSetLoader(opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('Loaded {} test LF image from {}'.format(len(test_loader), opt.testset_path))

    train(opt, train_loader, test_loader)


########################################################
if __name__ == "__main__":
    main(args)
