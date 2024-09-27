%% THE GENERALIZED SUP ADF TEST && Backward SUP ADF TEST %%
dateS=datenum(date(swindow0:end));

bsadfs=zeros(dim,1); 
for r2=swindow0:1:T
    dim0=r2-swindow0+1;
    rwadft=zeros(dim0,1);
    for r1=1:1:dim0
       rwadft(r1)= ADF_FL(y(r1:r2,1),0,1);   % two tail 5% significant level
    end
    bsadfs(r2-swindow0+1)=max(rwadft);
end
  
gsadf=max(bsadfs);

disp('The GSADF statistic');disp(gsadf);
disp('The critical values');disp(cv_gsadf); %% NOTE: cv_gsadf can be obtained through the function in this folder

figure()
plot(dateS,bsadfs,'--b','Linewidth',2);
hold on
plot(dateS,cv_bsadf(2,:)',':r','Linewidth',2);
xlim([min(dateS) max(dateS)]);
