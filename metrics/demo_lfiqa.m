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
lfiqa = zeros(test_num,1);

%% Start evaluate
tic
for scene_ind = 1:test_num
    % per scene
    scene_name = scene_folder(scene_ind).name;
    ref_im = imread([gt_path,scene_name,'/02_02.png']);
    h = size(ref_im,1);  w = size(ref_im,2);

    gt_lf = zeros(ang,ang,h,w,3); 
    com_lf = zeros(ang,ang,h,w,3);   
    % 5D LFI
    for an_u = 1:ang
        for an_v = 1:ang
            gt_sai = imread([gt_path,scene_name,'/',sprintf('%02d',an_u-1),'_',sprintf('%02d',an_v-1),'.png']);            
            com_sai = imread([com_path,'/',scene_name,'/',sprintf('%02d',an_u-1),'_',sprintf('%02d',an_v-1),'.png']);    
            gt_lf(an_u,an_v,:,:,:) = gt_sai;
            com_lf(an_u,an_v,:,:,:) = com_sai;
        end
    end
    lfiqa(scene_ind,1) = LightField_Measure(gt_lf, com_lf, ang); 
    fprintf('Evaluating on the Scene "%s"\n', num2str(scene_ind));
end
toc
    
metric_save_path = 'Quantitative_results/lfiqa/';
if (~exist(metric_save_path,'dir'))
    mkdir(metric_save_path);
end

save([metric_save_path,'Proposed_lfiqa.mat'], 'lfiqa');



