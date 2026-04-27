function ssim_value = cal_ssim(dis, ref)

% Input image: [0,1]
K = [0.01,0.03];
L = 1;         
C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
window =  fspecial('gaussian', 11, 1.5);	
window = window/sum(window(:));
padding = 'same';

[~,~,col] = size(dis);
dis = double(dis);
ref = double(ref);
SSIM = [];

if col == 1    % gray
    mu1   = filter2(window, dis, padding);
    mu2   = filter2(window, ref, padding);

    mu1_sq = mu1.*mu1;      % E(x)^2
    mu2_sq = mu2.*mu2;      % E(y)^2
    mu1_mu2 = mu1.*mu2;     % E(x)*E(y)
    sigma1_sq = filter2(window, dis.*dis, padding) - mu1_sq;      % D(x)=E(x.^2)- E(x)^2
    sigma2_sq = filter2(window, ref.*ref, padding) - mu2_sq;      % D(y)=E(y.^2)- E(y)^2
    sigma12 = filter2(window, dis.*ref, padding) - mu1_mu2;       % cov(x,y)=E(x*y)- E(x)*E(y)

    SSIM_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2)); 
    SSIM = mean(SSIM_map(:));
    
else             % color
    for c = 1:col
        mu1   = filter2(window, dis(:,:,c), padding);
        mu2   = filter2(window, ref(:,:,c), padding);

        mu1_sq = mu1.*mu1;     % E(x)^2
        mu2_sq = mu2.*mu2;      % E(y)^2
        mu1_mu2 = mu1.*mu2;     % E(x)*E(y)
        sigma1_sq = filter2(window, dis(:,:,c).*dis(:,:,c), padding) - mu1_sq;      % D(x)=E(x.^2)- E(x)^2
        sigma2_sq = filter2(window, ref(:,:,c).*ref(:,:,c), padding) - mu2_sq;      % D(y)=E(y.^2)- E(y)^2
        sigma12 = filter2(window, dis(:,:,c).*ref(:,:,c), padding) - mu1_mu2;       % cov(x,y)=E(x*y)- E(x)*E(y)

        SSIM_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2)); 
        SSIM(c) = mean(SSIM_map(:));
    end
end

ssim_value = mean(SSIM);

end
