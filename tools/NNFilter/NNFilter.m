% NNFilter method
function [filteredTrainX, filteredTrainY] = NNFilter(k, trainX, trainY, testX)

% This method is the implementation of the JDM authors
% Reference:?
% Turhan, B: 'On the relative value of cross-company and within-company data for defect prediction'

% Find k?nearest neighbors for testX
[distance,index]=pdist2(trainX, testX, 'Euclidean', 'Smallest', k);

% Filtering
testSize =size(testX,1);
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