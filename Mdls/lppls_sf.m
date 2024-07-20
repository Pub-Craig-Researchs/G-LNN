function Loss = lppl_sf(x,lpplData)
%% TODO: Leave the name of my paper here
time = lpplData(:,1);
price = lpplData(:,2);
N = size(time,1);
Loss = NaN(size(x,1),9);

ln_ap = log(abs(x(:,3)'-price));
f = log(abs(x(:,1)'-time+eps));
g = cos(x(:,2)'.*log(abs(x(:,1)'-time+eps)));
h = sin(x(:,2)'.*log(abs(x(:,1)'-time+eps)));

for idx = 1:size(x,1)
    K = [ones(N,1),f(:,idx),g(:,idx),h(:,idx)];
    quantilerror = createArray(length(price),9);
    for q = 1:9
        mdl_liner = quantreg(K,ln_ap(:,idx),q/10);
        quantilerror(:,q) = (exp(K*mdl_liner)-abs(price-x(idx,3))).^2;
    end
    Loss(idx,:) = mean(quantilerror);
end
end
