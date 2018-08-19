function [E_src, E_tar, E_tar_o] = tca(X_src, X_tar, X_tar_o, options)

n_tar = size(X_tar,1);
n_src = size(X_src,1);

mu = options.Mu;
lambda = options.lambda;
dim = options.Dim;

X = [X_src; X_tar];
L = [(1/(n_src*n_src))*ones(n_src, n_src) (-1/(n_src*n_tar))*ones(n_src, n_tar); (-1/(n_src*n_tar))*ones(n_tar, n_src) (1/(n_tar*n_tar))*ones(n_tar, n_tar)];
K = calckernel(options.Kernel, options.KernelParam, X);
K_tar_o = calckernel(options.Kernel, options.KernelParam, X, X_tar_o);

if lambda ~= 0 % Using Deform Kernel
    lap_options = ml_options('NNType','nn','NN',10,'Degree', 1);
    L1 = laplacian(X,'nn',lap_options);
    [K, K_tar_o] = Deform(lambda, K, L1, K_tar_o);
end

% TCA Kernel Matrix Construction
H = eye(n_src+n_tar) - 1/(n_src+n_tar) *ones(n_src+n_tar,1)*ones(n_src+n_tar,1)';
Kc = pinv(mu*eye(n_src+n_tar) + K*L*K)*K*H*K;

[V D] = eig(Kc);
eig_values = diag(D);
[eig_values_sorted index_sorted] = sort(eig_values, 'descend');
V = V(:, index_sorted);
E_src = K(1:n_src,:) * V;
E_tar = K(n_src+1:end,:) * V;
E_tar_o = K_tar_o * V;

E_src = E_src(:, 1:dim);
E_tar = E_tar(:, 1:dim);
E_tar_o = E_tar_o(:, 1:dim);


