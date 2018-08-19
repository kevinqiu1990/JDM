function [Cls, initMethod] = generateInitCls(initMethodId, sourceX, sourceY, targetX, targetY)

if initMethodId == 1
    initMethod = 'TCA';
    options = tca_options('Kernel', 'linear', 'KernelParam', 1, 'Mu', 1, 'lambda', 1, 'Dim', 10);
    [newtrainX, ~, newtestX] = tca(sourceX, targetX, targetX, options);
    model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(newtestX), model);
elseif initMethodId == 2
    initMethod = 'KMM';
    betaW = KMM('rbf', sourceX, targetX, 0.01);
    betaW = NormalizeAlpha(betaW, 1)
    model = train(betaW, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 3
    initMethod = 'DG';
    gravitation = cal_data_gravitation(targetX, sourceX);
    model = train(gravitation, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 4
    initMethod = 'NNFilter';
    [sourceX, sourceY] = NNFilter(15, sourceX, sourceY, targetX);
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 5
    initMethod = 'LR';
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
end

end

function gravitation = cal_data_gravitation(allWC , CCdata)
disp('calculating data gravitation');
graMat=zeros(size(CCdata,1),1);

minAttr=min(allWC,[],1);
maxAttr=max(allWC,[],1);

for i=1:size(CCdata,1)
    for j=1:size(CCdata,2)
        if (CCdata(i,j)>=minAttr(:,j) && CCdata(i,j)<=maxAttr(:,j))
            graMat(i,:)=graMat(i,:)+1;
        end
    end
end

gravitation=graMat ./ (size(CCdata,2) - graMat +1).^2;
gravitation = gravitation./sum(gravitation);
end

function [filteredTrainX, filteredTrainY] = NNFilter(k, trainX, trainY, testX)
%获取实例
testSize =size(testX,1);

%distance行数等于trainX实例数，列数等于testX实例数
[distance,index]=pdist2(trainX, testX, 'Euclidean', 'Smallest', k);

%去index重复的
finalIndex = [];
filteredTrainX = [];
filteredTrainY = [];
for i=1:k
    for j=1:testSize
        if (~ismember(index(i,j),finalIndex))
            finalIndex = [finalIndex;index(i,j)];
            filteredTrainX = [filteredTrainX; trainX(index(i,j),:)];
            filteredTrainY = [filteredTrainY; trainY(index(i,j),:)];
        end
    end
end

end