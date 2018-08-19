function A = adjacency(DATA, TYPE, PARAM, DISTANCEFUNCTION);

% Compute the adjacency graph of the data set DATA
%
% A = adjacency(DATA, TYPE, PARAM, DISTANCEFUNCTION);
% 
% DATA - NxK matrix. Data points are rows. 
% TYPE - string 'nn' or string 'epsballs'.
% PARAM - integer if TYPE='nn', real number if TYPE='epsballs'.
% DISTANCEFUNCTION - function mapping a (DxM) and a (D x N) matrix
%                    to an M x N distance matrix (D:dimensionality)
% Returns: A, sparse symmetric NxN matrix of distances between the
% adjacent points. 
%
% Example: 
% 
% A = adjacency(X,'nn',6) 
%   A contains the adjacency matrix for the data
%   set X. For each point, the distances to 6 adjacent points are
%   stored. N
%
% Note: the adjacency relation is symmetrized, i.e. if
% point a is adjacent to point b, then point b is also considered to be
% adjacent to point a.
%
%
% Author: 
%
% Mikhail Belkin 
% misha@math.uchicago.edu
%
% Modified by: Vikas Sindhwani
% June 2004

disp('Computing Adjacency Graph');

if (nargin < 3) | (strcmp(TYPE,'nn') & strcmp(TYPE,'epsballs')) | ~isreal(PARAM)  
  disp(sprintf('ERROR: Too few arguments given or incorrect arguments.\n'));
  disp(sprintf('USAGE:\n A = laplacian(DATA, TYPE, PARAM)'));
  disp(sprintf('DATA - the data matrix. Data points are rows.'));
  disp(sprintf('Nearest neigbors: TYPE =''nn''    PARAM = number of nearest neigbors')); 
  disp(sprintf('Epsilon balls: TYPE =''epsballs''    PARAM = redius of the ball\n'));
  return;
end

n = size(DATA,1);
disp (sprintf ('DATA: %d points in %d dimensional space.',n,size (DATA,2)));

switch TYPE
 case {'nn'}
  disp(sprintf('Creating the adjacency matrix. Nearest neighbors, N=%d.', PARAM)); 
 case{'eps', 'epsballs'} 
  disp(sprintf('Creating the adjacency matrix. Epsilon balls, eps=%f.', PARAM));
end;
  
A = sparse(n,n);

step = 100;  

% nn
if (strcmp(TYPE,'nn'))   
  for i1=1:step:n    
    i2 = i1+step-1;
    if (i2> n) 
      i2=n;
    end;
    XX= DATA(i1:i2,:);  
    dt = feval(DISTANCEFUNCTION, XX',DATA');
    [Z,I] = sort ( dt,2);    
    for i=i1:i2
      if ( mod(i, 500) ==0) 
    	%disp(sprintf('%d points processed.', i));
      end;
      % do not link duplicated sample points,  modified by panjf
      skipcopy=1;
      while Z(i-i1+1,skipcopy)==0
        skipcopy=skipcopy+1;
      end
      copyend=PARAM+skipcopy-1;
      if copyend>n
         copyend=n;
      end
      for j=skipcopy:copyend
	A(i,I(i-i1+1,j))= Z(i-i1+1,j); 
	A(I(i-i1+1,j),i)= Z(i-i1+1,j); 
      end; % for j
    end % for i
  end % for i1
% epsilon balls
else
  for i1=1:step:n
    i2 = i1+step-1;
    if (i2> n) 
        i2=n;
    end
    XX= DATA(i1:i2,:);  
    dt = feval(DISTANCEFUNCTION, XX',DATA');
    [Z,I] = sort ( dt,2 );  
    for i=i1:i2
    %  if ( mod(i, 500) ==0) disp(sprintf('%d points processed.', i)); end;
      j=2;
      while (j<=size(Z,2) && (Z(i-i1+1,j) < PARAM)) 
        %% j = j+1;  ???? don't understand this line, panjf
        jj = I(i-i1+1,j);
        A(i,jj)= Z(i-i1+1,j);
        A(jj,i)= Z(i-i1+1,j);
        j = j+1;
      end  % while j
    end % for i
  end % for i1

end;

 

