clear;

%%%%%%%%%%%%%%% Load Data %%%%%%%%%%%%%%%
[X_src, Y_src, X_tar_u, Y_tar_u, X_tar_o, Y_tar_o] = loadData;
%%% X_src (n1 x d) -- souce domain labeled data (Y_src are corresponding labels), d is the number of features.
%%% X_tar_u (n2 x d) -- target domain unlabeled data used for training (Y_tar_u are corresponding labels)
%%% X_tar_o (n3 x d) -- target domain out-of-sample test data (Y_tar_o are corresponding labels)

%%%%%%%%%%%%%%% Set Parameters %%%%%%%%%%%%%%%
dim = 10;            %%% Reduced dimensions. Default value is 5
kerneltype = 'rbf';  %%% Kernel type. Default value is 'linear', for wireless sensor data, 'lap' may be better,
kernelparam = 1;     %%% Kernel parameter. Default value is 1. If the kerneltype is 'rbf_auto' or 'lap_auto', then no need to set kernel parameter.
mu = 1;              %%% Parameters. Default value is 1
lambda = 1;          %%% Optional. Default value is 0. If lambda != 0 then a data-dependent kernel for both labeled and unlabeled data is performed. Refer to
                     %%% "Beyond the Point Cloud: from Transductive to Semi-supervised" for details. For wireless sensor data, this manifold term is important.

%%%%%%%%%%%%%%% TCA %%%%%%%%%%%%%%%
fprintf('TCA based Feature Extraction \n');
options = tca_options('Kernel', kerneltype, 'KernelParam', kernelparam, 'Mu', mu, 'lambda', lambda, 'Dim', dim);
[X_src_tca, X_tar_u_tca, X_tar_o_tca] = tca(X_src, X_tar_u, X_tar_o, options);
%%% X_src_tca --- (n1 x dim), new feature representations of X_src
%%% X_tar_u_tca --- (n2 x dim), new feature representations of X_tar_u
%%% X_tar_o_tca --- (n3 x dim), new feature representations of X_tar_o










