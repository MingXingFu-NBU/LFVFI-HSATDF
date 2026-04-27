import argparse
import os
import numpy as np
import torch
import scipy.io as scio
import lpips

# Initializing the model
use_gpu = True

loss_fn = lpips.LPIPS(net='vgg', version='0.1')
if (use_gpu):
    loss_fn.cuda()

# RGB
ang_res = 5
compare_path = 'E:/FMX/Quantitative_comparison/work2_test_result_final/'
compare_name = ['VFIformer','RIFE','EMA_VFI','MSEConv','FIRMA','UPR_Net','LF_Slomo','Proposed']
GT_path = 'E:/FMX/Quantitative_comparison/work2_test_result_final/GT/'

output_path = 'lpips_results/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

for mi in range(len(compare_name) - 1):         # comparison method
    scene_path = compare_path + compare_name[mi]
    scene_name = os.listdir(scene_path)

    metric_path = output_path + compare_name[mi] + '.mat'

    lpips_score = []
    for si in range(len(scene_name)):       # scene
        lpips_lf = 0.
        for an_u in range(0, ang_res):         # angular resolution
            for an_v in range(0, ang_res):
                dis_file = scene_path + '/' + scene_name[si] + '/0' + str(an_u) + '_0' + str(an_v) + '.png'
                ref_file = GT_path + scene_name[si] + '/0' + str(an_u) + '_0' + str(an_v) + '.png'

                dis_img = lpips.im2tensor(lpips.load_image(dis_file))    # RGB image from [-1,1]
                ref_img = lpips.im2tensor(lpips.load_image(ref_file))

                if (use_gpu):
                    dis_img = dis_img.cuda()
                    ref_img = ref_img.cuda()
                # Compute distance
                lpips_sai = loss_fn.forward(dis_img, ref_img)
                lpips_lf += lpips_sai.item()
        lpips_lf = lpips_lf / (ang_res**2)
        # print('%.4f' % (lpips_lf))
        lpips_score.append(lpips_lf)

    scio.savemat(metric_path, {'lpips_scores': lpips_score})

    value1 = lpips_score[0:6]
    value2 = lpips_score[6:66]

    ave_lpips1 = np.mean(value1)
    ave_lpips2 = np.mean(value2)

    print("%s:  %s：%.4f， %s：%.4f" % (compare_name[mi], "dataset3", ave_lpips1, "dataset4", ave_lpips2))




