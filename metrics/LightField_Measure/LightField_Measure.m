function score =  LightField_Measure(View_RefName,View_DisName, Ang_Res)
% Input:  (1) View_RefName: file names of the reference views
%         (2) View_DisName: file names of the distorted views
% Output: (1) score: quality score
% Usage:  Given the file names of the reference and distorted views
%         score =  LightField_Measure(View_RefName,View_DisName)
        
% for i = 1:length(View_RefName)
ik = 0;
for ia = 1:Ang_Res
    for ib = 1:Ang_Res
        ik = ik + 1;
    %     viewRef = imread(View_RefName{1,i});
    %     viewDis = imread(View_DisName{1,i});
        viewRef = rgb2gray(squeeze(double(View_RefName(ia,ib,:,:,:))));
        viewDis = rgb2gray(squeeze(double(View_DisName(ia,ib,:,:,:))));

        cornerScore(1,ik) = cornerSIM(viewRef,viewDis);
        edgeScore(1,ik) = edgeMSE(viewRef,viewDis);
    end
end

angularScore = angularAnalysis(edgeScore);

score = log(mean(cornerScore./(edgeScore.^0.5+(0.01*255)^2)*angularScore));

end