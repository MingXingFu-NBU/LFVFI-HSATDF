clc
clear all;
   
addpath('traditional_metrics');    % linear psnr/ssim 
addpath('LightField_Measure');     % LFIQA

ang = 5;
gt_path = 'GT/';
com_path = '../outputs/';
scene_folder = dir(gt_path); scene_folder = scene_folder(3:end);

%% metrics
test_num = length(scene_folder);
y_psnr_sai = zeros(test_num,ang,ang);   y_ssim_sai = zeros(test_num,ang,ang);

%% Start evaluate
tic
for scene_ind = 1:test_num
    % per scene
    scene_name = scene_folder(scene_ind).name;
    for an_u = 1:ang
        for an_v = 1:ang
            gt_sai = imread([gt_path,scene_name,'/',sprintf('%02d',an_u-1),'_',sprintf('%02d',an_v-1),'.png']);            
            com_sai = imread([com_path,'/',scene_name,'/',sprintf('%02d',an_u-1),'_',sprintf('%02d',an_v-1),'.png']);
            gt_sai = rgb2ycbcr(gt_sai);    gt_sai = gt_sai(:,:,1);
            com_sai = rgb2ycbcr(com_sai);  com_sai = com_sai(:,:,1);
            y_psnr_sai(scene_ind,an_u,an_v) = cal_psnr(im2double(com_sai), im2double(gt_sai));  
            y_ssim_sai(scene_ind,an_u,an_v) = cal_ssim(im2double(com_sai), im2double(gt_sai));  
        end
    end
   fprintf('Evaluating on the Scene "%s"\n', num2str(scene_ind));
end
toc

metric_save_path = 'Quantitative_results/y_psnr_ssim/';
if (~exist(metric_save_path,'dir'))
    mkdir(metric_save_path);
end

save([metric_save_path,'Proposed_y_psnr_ssim_metrics.mat'], 'y_psnr_sai','y_ssim_sai');
