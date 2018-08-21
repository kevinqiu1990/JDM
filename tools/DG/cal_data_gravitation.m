% DG method
function gravitation = cal_data_gravitation(WC_data , CC_data)

% This method is the implementation of the JDM authors
% Reference:?
% Peng, L: 'Data gravitation based classification',

disp('calculating data gravitation');
graMat=zeros(size(CC_data,1),1);

minAttr=min(WC_data,[],1);
maxAttr=max(WC_data,[],1);

for i=1:size(CC_data,1)
    for j=1:size(CC_data,2)
        if (CC_data(i,j)>=minAttr(:,j) && CC_data(i,j)<=maxAttr(:,j))
            graMat(i,:)=graMat(i,:)+1;
        end
    end
end

gravitation=graMat ./ (size(CC_data,2) - graMat +1).^2;
gravitation = gravitation./sum(gravitation);

end