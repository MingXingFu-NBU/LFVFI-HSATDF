function psnr_value = cal_psnr(dis, ref)

% Input image: [0,1]
dis = double(dis);
ref = double(ref);
element_num = length(dis(:));
diff = (dis-ref).^2;
mse = sum(diff(:))/element_num;
psnr_value = 10*log10(1^2/mse);

end