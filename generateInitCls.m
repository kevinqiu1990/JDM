function [Cls, initMethod] = generateInitCls(initMethodId, sourceX, sourceY, targetX, targetY)

if initMethodId == 1 % use TCA as the initial Pseudo-Label predictor
    initMethod = 'TCA';
    options = tca_options('Kernel', 'linear', 'KernelParam', 1, 'Mu', 1, 'lambda', 1, 'Dim', 10);
    [newtrainX, ~, newtestX] = tca(sourceX, targetX, targetX, options);
    model = train([], sourceY, sparse(newtrainX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(newtestX), model);
elseif initMethodId == 2 % use KMM
    initMethod = 'KMM';
    betaW = KMM('rbf', sourceX, targetX, 0.01);
    betaW = normalizeAlpha(betaW, 1);
    model = train(betaW, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 3 % use DG
    initMethod = 'DG';
    gravitation = cal_data_gravitation(targetX, sourceX);
    model = train(gravitation, sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 4 % use NNFilter
    initMethod = 'NNFilter';
    [sourceX, sourceY] = NNFilter(15, sourceX, sourceY, targetX);
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
elseif initMethodId == 5 % use LR only
    initMethod = 'LR';
    model = train([], sourceY, sparse(sourceX), '-s 0 -c 1');
    Cls = predict(targetY, sparse(targetX), model);
end

end