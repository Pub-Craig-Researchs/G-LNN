%**************************************************************************
%   "Testing for Multiple Bubbles" by Phillips, Shi and Yu (2015)
    
%   In this program, we calculate critical values for the generalized sup 
%   ADF statistic.
% *************************************************************************

%% important note: GSADF is a test for the entire sequence and BSADF is the test to select sequence in this paper.

function [cv_gsadf,cv_bsadf]=CV_GSADF(T,swindow0,qe)


m=1000;
dim=T-swindow0+1;

%% %%%% DATA GENERATING PROCESS %%%%%%
e=randn(T,m); 
a=T^(-1);
y=cumsum(e+a);

%% THE GENERALIZED SUP ADF TEST %%%%%%

gsadf=ones(m,1);  
sadfs=zeros(m,dim); 
for j=1:m
    disp(j)
    for r2=swindow0:1:T
        dim0=r2-swindow0+1;
        rwadft=zeros(dim0,1);
        for r1=1:1:dim0
            rwadft(r1)= ADF_FL(y(r1:r2,j),0,1);  % two tail 5% significant level
        end
        sadfs(j,r2-swindow0+1)=max(rwadft);
    end
    gsadf(j)=max(sadfs(j,:));
end

cv_gsadf=quantile(gsadf,qe);
cv_bsadf=quantile(sadfs,qe);
end
