function [pcaA V] = fastPCA( A, k )
% fastPCA
% input£ºA --- Sample matrix, one sample per row
%        k --- reduce to k dimension
% output£ºpcaA --- A matrix consisting of k-dimensional sample eigenvectors, one sample per row
%         V --- Principal component vector

[r c] = size(A);
% Sample mean
meanVec = mean(A);
% Calculate the transpose of the covariance matrix covMatT
Z = (A-repmat(meanVec, r, 1));
covMatT = Z * Z';
% Calculate the first k eigenvalues and eigenvectors of covMatT
[V D] = eigs(covMatT, k);
% Get the eigenvector of the covariance matrix (covMatT)'
V = Z' * V;
% Norm eigenvectors
for i=1:k
    V(:,i)=V(:,i)/norm(V(:,i));
end
% Linear transformation (projection) dimension reduction
pcaA = Z * V;