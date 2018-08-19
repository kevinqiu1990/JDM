function [accuracy,sensitivity,specificity,precision,recall,f_measure,gmean,MCC, AUC] = evaluate(PREDICTED, ACTUAL)

idx = (ACTUAL()==1);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp; 

tp_rate = tp/p;
tn_rate = tn/p;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
MCC = (tp*tn - fp*fn) /  sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

%º∆À„AUC÷µ
[A,I]=sort(PREDICTED);
M=0;
N=0;
for i=1:length(PREDICTED)
    if(ACTUAL(i)==1)
        M=M+1;
    else
        N=N+1;
    end
end
sigma=0;
for i=M+N:-1:1
    if(ACTUAL(I(i))==1)
        sigma=sigma+i;
    end
end
AUC=(sigma-(M+1)*M/2)/(M*N);

if isnan(f_measure)
    f_measure = 0;
end
    
if isnan(MCC)
    MCC = 0;
end

if isnan(AUC)
    AUC = 0;
end
    
end