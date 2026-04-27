clc
clear all;

%% LFIQA
path = 'Quantitative_results/lfiqa/';

alg_Proposed = load([path,'Proposed_lfiqa.mat']);
alg_Proposed = alg_Proposed.lfiqa;    
ave_alg_Proposed1 = round(mean(alg_Proposed(1:6)),3);
ave_alg_Proposed2 = round(mean(alg_Proposed(7:66)),3);

