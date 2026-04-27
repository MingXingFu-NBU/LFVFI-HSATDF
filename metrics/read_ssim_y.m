clc
clear all;

%% PU21
path = 'Quantitative_results/y_psnr_ssim/';

alg_Proposed = load([path,'Proposed_y_psnr_ssim_metrics']);
alg_Proposed = alg_Proposed.y_ssim_sai;    alg_VFIformer = squeeze(mean(mean(alg_Proposed,2),3)); 
ave_alg_Proposed1 = round(mean(alg_Proposed(1:6)),3);
ave_alg_Proposed2 = round(mean(alg_Proposed(7:66)),3);