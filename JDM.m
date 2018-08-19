function [betaW, Xs, Ys] = JDM(ker,Xs,Xt,Ys,Yt,sigma)

% init data
X0s = Xs(find(Ys==0),:);
X1s = Xs(find(Ys==1),:);
Xs  = [X0s;X1s];
Ys  = [Ys(Ys==0);Ys(Ys==1)];

X0t = Xt(find(Yt==0),:);
X1t = Xt(find(Yt==1),:);
Xt  = [X0t;X1t];

ns = size(Xs,1);  % number of train samples
nt = size(Xt,1);  % number of test samples
n0s = size(X0s,1);
n0t = size(X0t,1);
n1s = size(X1s,1);
n1t = size(X1t,1);

% variables: (here in the program / in (12) in the paper)
% H is K
% kappa

% minimize...
% 'calculating H=K...'
H   = calckernel(ker, sigma, Xs, Xs);
H0  = calckernel(ker, sigma, X0s, X0s);
H1  = calckernel(ker, sigma, X1s, X1s);
% H   = (H+H')/2; %make the matrix symmetric (it isn't symmetric before because of bad precision)
% H0  = (H0+H0')/2; 
% H1  = (H1+H1')/2; 

% 'calculating kappa...'
R3 = calckernel(ker, sigma, Xs, Xt);
kappa=(R3'*ones(nt, 1));
kappa=-ns/nt*kappa;
R30 = calckernel(ker, sigma, X0s, X0t);
kappa0=(R30'*ones(n0t, 1));
kappa0=-n0s/n0t*kappa0;
R31 = calckernel(ker, sigma, X1s, X1t);
kappa1=(R31'*ones(n1t, 1));
kappa1=-n1s/n1t*kappa1;

kappa  = kappa';
kappa0 = kappa0';
kappa1 = kappa1';

% subject to...
% abs(sum(beta_i) - m) <= m*eps
% which is equivalent to A*beta <= b where A=[1,...1;-1,...,-1] and b=[m*(eps+1);m*(eps-1)]
eps = (sqrt(ns)-1)/sqrt(ns);
%eps=1000/sqrt(nsamples);
A = ones(1,ns);
A(2,:) = -ones(1,ns);
b = [ns*(eps+1); ns*(eps-1)];

% X=solve the cvx programming problem:
%              min 0.5*x'*H*x + kappa'*x
% subject to:  A*x <= b
%              Aeq*x = beq
%              LB <= x <= UB

% 'solving cvx for betas...'
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
