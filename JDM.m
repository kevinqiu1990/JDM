function [betaW, Xs, Ys] = JDM(ker, Xs, Xt, Ys, Yt, sigma)

% ----------------------------------------------------------------------------------------%
% JDM - joint distribution matching (JDM) model for cross-project defect prediction
%
% Parameters
%    ker - kernel_type = 'linear' | 'poly' | 'rbf' | ...
%    Xs - the features of source project (Ms x Ns)
%    Xt - the features of target project (Mt x Nt)
%    Ys - the labels of source project (1 x Ns)
%    Yt - the pseudo-labels of target project (1 x Nt)
%    sigma - function width of gaussian kernel
% 
% Returns:
%    betaW - an adaptive weight vector for the instances of the source project (1 x Ns)
%    Xs - reordered features of source project (Ms x Ns)
%    Ys - reordered features of source project (1 x Ns)
%
% Acknowledgement: Modified from the method of Kerneal Mean Matching (KMM)
% Huang, J., Smola, A. J., Gretton, A., et al.: ¡®Correcting sample selection bias by unlabeled data¡¯, 
% We thank Qian Sun for providing us the source code of KMM
%
% Example : 
%    [betaW, Xs, Ys] = JDM('rbf', sourceX, targetX, sourceY, predictY, sigma)
% ----------------------------------------------------------------------------------------%

% extract data based on labels
X0s = Xs(find(Ys==0),:);
X1s = Xs(find(Ys==1),:);
Xs  = [X0s;X1s];
Ys  = [Ys(Ys==0);Ys(Ys==1)];

X0t = Xt(find(Yt==0),:);
X1t = Xt(find(Yt==1),:);
Xt  = [X0t;X1t];

ns = size(Xs,1);  % number of source samples
nt = size(Xt,1);  % number of target samples
n0s = size(X0s,1); % number of source samples with label 0
n0t = size(X0t,1); % number of target samples with label 0
n1s = size(X1s,1); % number of source samples with label 1
n1t = size(X1t,1); % number of target samples with label 1

% variables: (here in the program / in (12) in the paper)
% H is K
% kappa

% minimize...
% 'calculating H=K...'
H   = calckernel(ker, sigma, Xs, Xs);
H0  = calckernel(ker, sigma, X0s, X0s);
H1  = calckernel(ker, sigma, X1s, X1s);

% 'calculating kappa...'
R = calckernel(ker, sigma, Xs, Xt);
kappa=(R'*ones(nt, 1));
kappa=-ns/nt*kappa;
R0 = calckernel(ker, sigma, X0s, X0t);
kappa0=(R0'*ones(n0t, 1));
kappa0=-n0s/n0t*kappa0;
R1 = calckernel(ker, sigma, X1s, X1t);
kappa1=(R1'*ones(n1t, 1));
kappa1=-n1s/n1t*kappa1;

kappa  = kappa';
kappa0 = kappa0';
kappa1 = kappa1';

% subject to...
% abs(sum(beta_i) - m) <= m*eps
% which is equivalent to A*beta <= b where A=[1,...1;-1,...,-1] and b=[m*(eps+1);m*(eps-1)]
eps = (sqrt(ns)-1)/sqrt(ns);
A = ones(1,ns);
A(2,:) = -ones(1,ns);
b = [ns*(eps+1); ns*(eps-1)];

% X=solve the cvx programming problem:
%              min 0.5*x'*H*x + kappa'*x
% subject to:  A*x <= b
%              Aeq*x = beq
%              LB <= x <= UB

% 'solving cvx for betaW...'
cvx_begin
	variables betaW(ns,1)
	minimize 0.5*quad_form(betaW,H) + kappa*betaW + ...
             0.5*quad_form(betaW(1:n0s,:),H0) + kappa0*betaW(1:n0s,:) + ...
             0.5*quad_form(betaW(n0s+1:n0s+n1s,:),H1) + kappa1*betaW(n0s+1:n0s+n1s,:)
    subject to %condition
        betaW >= 0;
        betaW <= 1000;
        A * betaW <= b;
cvx_end
    
% guarantee that all betaW greater than 0
threshold=0.01*abs(median(betaW));
betaW(find(betaW<threshold)) = 0;
sprintf('number of betaW < %f: %d (0 is good)', threshold, length(find(betaW<0.1)));

end
