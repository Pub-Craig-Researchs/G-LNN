% -----------------------------------------------------------------------
%  "Testing for Multiple Bubbles" by Phillips, Shi and Yu (2011)
%-------------------------------------------------------------------------
function estm = ADF_FL(y,adflag,mflag)

  t1=size(y,1)-1;
  const=ones(t1,1);
  trend=1:1:t1;trend=trend';

  y1  = y(size(y,1)-t1:size(y,1)-1);
  dy  = y(2:size(y,1)) - y(1:size(y,1)-1);
  dy0 = dy(size(dy,1)-t1+1:size(dy,1));
  x=y1;

  if mflag==1;  x=[x const]; end
  if mflag==2;  x=[x const trend]; end
 % if mflag==3;  x=x; end;

  x1=x; 

  t2=t1-adflag;
  x2=x1(size(x1,1)-t2+1:size(x1,1),:);         % from k+1 to the end (including y1 and x)-@
  dy01=dy0(size(dy0,1)-t2+1:size(dy0,1));      % from k+1 to the end (including dy0)-@
   
 if adflag>0
    j=1; 
    while j<=adflag
      x2=[x2 dy(size(dy,1)-t2+1-j:size(dy,1)-j)];     %including k lag variables of dy in x2-@
      j=j+1; 
    end
 end

  beta = (x2'*x2)^(-1)*(x2'*dy01);                       % model A-@
  eps  = dy01 - x2*beta;
 
  if mflag==1; sig = sqrt(diag(eps'*eps/(t2-adflag-2)*(x2'*x2)^(-1))); end
  if mflag==2; sig = sqrt(diag(eps'*eps/(t2-adflag-3)*(x2'*x2)^(-1))); end
  if mflag==3; sig = sqrt(diag(eps'*eps/(t2-adflag-1)*(x2'*x2)^(-1))); end

  tvalue = beta./sig;
end
 
  estm = tvalue(1);

end
