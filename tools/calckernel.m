function K=calckernel(kernel_type,kernel_param,X1,X2)

% CALCKERNEL Computes Gram matrix of a specified kernel
% -------------------------------------------------------%
% Usage:
% K=calckernel(kernel_type,kernel_param,X1);
% K=calckernel(kernel_type,kernel_param,X1,X2);
%
% kernel_type = 'linear' | 'poly' | 'rbf' | ...
% kernel_param = -- | degree | sigma
%
% Given a single data matrix X (n x d where d is dimensionality)
% returns Gram matrix K (n x n)
%
% Given two data matrices X1 (n1 x d), X2 (n2 x d)
% returns Gram matrix K (n2 x n1)
%
% Acknowledgement: Modified from Vikas Sindhwani's software:
%               http://manifold.cs.uchicago.edu/manifold_regularization/software.html
%
% Author: Sinno Jialin Pan (Dept. of CSE, HKUST)
% June 2008
% -------------------------------------------------------%



dim=size(X1,2);
n1=size(X1,1);

if nargin==4
    n2=size(X2,1);
end

switch kernel_type

    case 'linear'

        if nargin==4
            K=X2*X1';
        else
            K=X1*X1';
        end

    case 'poly'

        if nargin==4
            K=(X2*X1').^kernel_param;
        else
            K=(X1*X1').^kernel_param;
        end

    case 'rbf'

        if nargin==4
            %         disp('rbf kernel exp(-|x1-x2|^2/(2*dim*param))');
            K = exp(-(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
                - 2*X2*X1')/(dim*2*kernel_param));     
        else
            %         disp('rbf kernel exp(-|x1-x2|^2/(2*dim*param))');
            P=sum(X1.*X1,2);
            K = exp(-(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1')/(dim*2*kernel_param));    
        end
        
         %K
        
    case 'rbf_auto'

        if nargin==4
            A = (repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) - 2*X2*X1');
            kernel_param = mean(mean(A));
            K = exp(-A/(kernel_param));
        else
            %         disp('rbf kernel exp(-|x1-x2|^2/(2*dim*param))');
            P = sum(X1.*X1,2);
            A = (repmat(P',n1,1) + repmat(P,1,n1) - 2*X1*X1');
            kernel_param = mean(mean(A));
            K = exp(-A/(kernel_param));
        end

    case 'invsquare'

        if nargin==4
            %         disp('1/(gamma*|u-v|^2+1)');
            K = 1./(kernel_param*(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
                - 2*X2*X1') + 1);
        else
            %         disp('1/(gamma*|u-v|^2+1)');
            P=sum(X1.*X1,2);
            K = 1./(kernel_param*(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1') + 1);
        end

    case 'inv'

        if nargin==4
            %         disp('1/(sqrt(gamma)*|u-v|+1)');
            K = 1./(sqrt(kernel_param)*sqrt(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
                - 2*X2*X1') + 1);
        else
            %         disp('1/(sqrt(gamma)*|u-v|+1)');
            P=sum(X1.*X1,2);
            K = 1./(sqrt(kernel_param)*sqrt(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1') + 1);
        end


    case 'lap'

        if nargin==4
            %         disp('rbf kernel exp(-|x1-x2|^2/(2*dim*param))');
            K = exp(-sqrt(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
                - 2*X2*X1')/(kernel_param));
        else
            %         disp('rbf kernel exp(-|x1-x2|^2/(2*dim*param))');
            P=sum(X1.*X1,2);
            K = exp(-sqrt(repmat(P',n1,1) + repmat(P,1,n1) ...
                - 2*X1*X1')/(kernel_param));
        end
        
    case 'lap_auto'

        if nargin==4
            A = sqrt(repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) - 2*X2*X1');
            kernel_param = mean(mean(A));
            K = exp(-A/kernel_param);
        else
            P=sum(X1.*X1,2);
            A = sqrt(repmat(P',n1,1) + repmat(P,1,n1) - 2*X1*X1');
            kernel_param = mean(mean(A));
            K = exp(-A/kernel_param);
        end

    case 'exp'

        if nargin==4
            K = exp(kernel_param*X2*X1');
        else
            K = exp(kernel_param*X1*X1');
        end
        
    otherwise
        error(['Unsupported kernel ' ker])
end
