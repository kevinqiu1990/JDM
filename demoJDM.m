% init program
clc();
clear();
addpath (genpath('.'))

% final result file
result_file_name=['./output/result.csv'];
result_file=fopen(result_file_name,'w');
% result file header
headerStr = 'repository,target,source,f1,AUC';
fprintf(result_file,'%s\n',headerStr);

% JDM process file
process_f1_file_name = ['./output/process_f1_result.csv'];
process_f1_file=fopen(process_f1_file_name,'w');
process_AUC_file_name = ['./output/process_AUC_result.csv'];
process_AUC_file=fopen(process_AUC_file_name,'w');

% Set parameter
sigma = 0.1; % Function width of gaussian kernel
initMethodId = 1; %1-TCA, 2-KMM, 3-DG, 4-NNFilter, 5-LR only
percent = 1; % End condition of the pseudo-label refinment procedure. Here, 1 means 100%.
max_iter = 20; % Max iteration of JDM process

% Choose repository
% use AEEEM
repositoryName = 'AEEEM';
load ./data/AEEEM.mat
fileList={'EQ','JDT','LC','ML', 'PDE'};
attributeNum=61; % the projects in AEEEM own 61 attributes
labelIndex=62;
% use PROMISE
% repositoryName = 'PROMISE';
% load ./data/PROMISE.mat
% fileList={'ant','arc','camel','elearning','ivy','prop','synapse','systemdata','tomcat','velocity'};
% attributeNum=20; % the projects in PROMISE own 20 attributes
% labelIndex=21;


% ---------Start JDM---------

% Traverse target projects
for i = 1:length(fileList)
    targetName=fileList{i};
    targetData=eval(targetName);
    targetData(targetData(:,labelIndex)==-1,labelIndex)=0;
    targetX = targetData(:,1:attributeNum);
    targetX = zscore(targetX);
    targetY = targetData(:,labelIndex);
    
    % Traverse source projects
    for j = 1:length(fileList)
        if(i~=j) % Skip this loop if target and souce projects are same
            sourceName=fileList{j};
            sourceData=eval(sourceName);
            sourceData(sourceData(:,labelIndex)==-1,labelIndex)=0;
            sourceX = sourceData(:,1:attributeNum);
            sourceX = zscore(sourceX);
            sourceY = sourceData(:,labelIndex);
            
            % call initMethod to generate init Cls
            [Cls, initMethod] = generateInitCls(initMethodId, sourceX, sourceY, targetX, targetY);
            [~,~,~,~,~,f_measure,~,~,AUC] = evaluate(Cls, targetY);
            fprintf(process_f1_file,'%f,(init)',f_measure);
            fprintf(process_AUC_file,'%f,(init)',AUC);
            
            % save the Cls into ClsArray to compare later
            ClsArray = [Cls];
            
            for t = 2:max_iter
                [betaW, Xs, Ys] = JDM('rbf',sourceX,targetX,sourceY,Cls,sigma);
                betaW = normalizeAlpha(betaW, 1);
                
                model = train(betaW, Ys, sparse(Xs), '-s 0 -c 1');
                Cls = predict(targetY, sparse(targetX), model);
                [~,~,~,~,~,f_measure,~,~,AUC] = evaluate(Cls, targetY);
                
                % Calculate the percentage of number that same prediction as the previous round
                size_same = size(Cls(Cls==ClsArray(:,t-1)),1);
                size_y = size(targetY,1);
                currentPercent = size_same/size_y;
                fprintf(process_f1_file,'%f,(%0.3f),',f_measure, currentPercent);
                fprintf(process_AUC_file,'%f,(%0.3f),',AUC, currentPercent);
                
                % By comparing last iteration, if all pseudo-labels are not changes, break the loop
                if currentPercent >= percent
                    break;
                end
                
                ClsArray = [ClsArray,Cls];
            end
            
            fprintf(process_f1_file,'\n');
            fprintf(process_AUC_file,'\n');
            
            %parameter string
            resultStr = [repositoryName,',',targetName,',',sourceName,',',num2str(f_measure),',',num2str(AUC)]
            fprintf(result_file,'%s\n',resultStr);
        end
    end
end

% close files
close(result_file_name);
close(process_f1_file_name);
close(process_AUC_file_name);