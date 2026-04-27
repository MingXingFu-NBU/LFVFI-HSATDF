clc
clear all;

clc
clear all;

%% PU21
path = 'lpips_results/';
com_method = {'VFIformer','RIFE','EMA_VFI','MSEConv','FIRMA','UPR_Net','LF_Slomo','Proposed'};

alg_VFIformer = load([path,com_method{1},'.mat']);
alg_VFIformer = alg_VFIformer.lpips_scores;
alg_VFIformer1 = mean(alg_VFIformer(1:6));  alg_VFIformer2 = mean(alg_VFIformer(7:66));

alg_RIFE = load([path,com_method{2},'.mat']);
alg_RIFE = alg_RIFE.lpips_scores;
alg_RIFE1 = mean(alg_RIFE(1:6));  alg_RIFE2 = mean(alg_RIFE(7:66));

alg_EMA_VFI = load([path,com_method{3},'.mat']);
alg_EMA_VFI = alg_EMA_VFI.lpips_scores;
alg_EMA_VFI1 = mean(alg_EMA_VFI(1:6));  alg_EMA_VFI2 = mean(alg_EMA_VFI(7:66));

alg_MSEConv = load([path,com_method{4},'.mat']);
alg_MSEConv = alg_MSEConv.lpips_scores;
alg_MSEConv1 = mean(alg_MSEConv(1:6));  alg_MSEConv2 = mean(alg_MSEConv(7:66));

alg_FIRMA = load([path,com_method{5},'.mat']);
alg_FIRMA = alg_FIRMA.lpips_scores;
alg_FIRMA1 = mean(alg_FIRMA(1:6));  alg_FIRMA2 = mean(alg_FIRMA(7:66));

alg_UPR_Net = load([path,com_method{6},'.mat']);
alg_UPR_Net = alg_UPR_Net.lpips_scores;
alg_UPR_Net1 = mean(alg_UPR_Net(1:6));  alg_UPR_Net2 = mean(alg_UPR_Net(7:66));

alg_LF_Slomo = load([path,com_method{7},'.mat']);
alg_LF_Slomo = alg_LF_Slomo.lpips_scores;
alg_LF_Slomo1 = mean(alg_LF_Slomo(1:6));  alg_LF_Slomo2 = mean(alg_LF_Slomo(7:66));
