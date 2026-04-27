clc
clear
base_path = 'E:\FMX\Quantitative_comparison\2';
target_path = 'E:\FMX\Quantitative_comparison\3';
subFolder = dir(base_path); subFolder = subFolder(3:end);
class(base_path)  
class(subFolder)
ang=5;

for method_ind = 1:length(subFolder)
        method_folder=fullfile(base_path, subFolder(method_ind).name);
        dataset_folder = dir(method_folder); dataset_folder = dataset_folder(3:end);
        c=1;
        for dataset_id = 1:length(dataset_folder)
            sub_dataset_folder=fullfile(method_folder,dataset_folder(dataset_id).name);
            sub_scene_folder = dir(sub_dataset_folder); sub_scene_folder = sub_scene_folder(3:end);
            data = [];
            
            for scene_id= 1:length(sub_scene_folder)
                target_folder=fullfile(target_path,subFolder(method_ind).name,num2str(c,'%03d'));
    
                if ~exist(target_folder,'dir')
                    mkdir(target_folder);
                end
                scene_folder=fullfile(sub_dataset_folder,sub_scene_folder(scene_id).name);
                i=1;
                for an_u = 1:ang
                    for an_v = 1:ang
                        u   = an_u - 1;
                        v   = an_v - 1;
                        stem= sprintf('%02d_%02d.png',u,v);
                        mew_syem=sprintf('%02d_%02d.png',u,v);
                        % 公共部分  "0_1.png" 等
                        newFile = fullfile(target_folder, mew_syem);
                        % 两种可能
                        candIn  = fullfile(scene_folder,stem);
                        candOut = fullfile(scene_folder,['output' stem]);
                        cand_Out = fullfile(scene_folder,['output_' stem]);
                        % candin=fullfile(scene_folder,['target_' stem]);
                        
                        if exist(candIn,'file')                  % 优先 input
                            out_name= candIn;
                        elseif exist(candOut,'file')             % 退而求 output
                            out_name = candOut;
                        elseif exist(cand_Out,'file')
                            out_name = cand_Out;
                        else
                            error('找不到对应视图文件：%s 或 %s',candIn,candOut);
                        end
                        copyfile(out_name, newFile);

                        % View_DisName{1,i} = fullfile(scene_folder,['input_' num2str(an_u-1,'%d') '_' num2str(an_v-1,'%d') '.png']);

                    end
                end
                c=c+1;
            end
            

        end
end
% disp(C)

