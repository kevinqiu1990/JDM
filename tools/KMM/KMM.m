function [beta] = KMM(ker, X, Xtst, sigma)

nsamples = size(X,1);  % number of train samples
ntestsamples = size(Xtst,1);  % number of test samples

% variables: (here in the program / in (12) in the paper)
% H is K
% f is kappa

% minimize...
% 'calculating H=K...'

H = calckernel(ker, sigma, X, X);
H=(H+H')/2; %make the matrix symmetric (it isn't symmetric before because of bad precision)

% 'calculating f=kappa...'
R3 = calckernel(ker, sigma, X, Xtst);
R3 = R3';
f=(R3*ones(ntestsamples, 1));
f=-nsamples/ntestsamples*f;

% did the same, but slowlier:
% f=-nsamples/ntestsamples*ones(nsamples,1);
% for i=1:nsamples
%     fi=0;
%     for j=1:ntestsamples
%         fi = fi + mykernel(X(i,:),Xtst(j,:),sigma);
%     end
%     f(i,1) = f(i,1)*fi;
% end
%
% do they really the same?
%'different f?'
%[f1 f]

% subject to...
% abs(sum(beta_i) - m) <= m*eps
% which is equivalent to A*beta <= b where A=[1,...1;-1,...,-1] and b=[m*(eps+1);m*(eps-1)]
eps = (sqrt(nsamples)-1)/sqrt(nsamples);
%eps=1000/sqrt(nsamples);
A=ones(1,nsamples);
A(2,:)=-ones(1,nsamples);
b=[nsamples*(eps+1); nsamples*(eps-1)];

Aeq = [];
beq = [];

% 0 <= beta_i <= 1000 for all i
LB = ones(nsamples,1).*0;
UB = ones(nsamples,1).*1;

% X=QUADPROG(H,f,A,b,Aeq,beq,LB,UB) attempts to solve the quadratic programming problem:
%              min 0.5*x'*H*x + f'*x
% subject to:  A*x <= b
%              Aeq*x = beq
%              LB <= x <= UB

% 'solving quadprog for betas...'
[beta] = quadprog(H,f,A,b,Aeq,beq,LB,UB);

% guarantee that all beta greater than 0
threshold=0.01*abs(median(beta));
beta(find(beta<threshold)) = 0;
sprintf('number of beta < %f: %d (0 is good)', threshold, length(find(beta<0.1)));

end
