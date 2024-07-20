function Loss = lppls_rf(x,lpplData)

%% Why Stock Markets Crash: Critical Events in Complex Financial Systems %%

Time = lpplData(:,1);
Price = lpplData(:,2);
N = size(Time,1);

for idx = 1:size(x,1)
    f = abs(x(idx,1)-Time+eps).^x(idx,2);
    g = f .* cos(x(idx,3).*log(abs(x(idx,1)-Time+eps)));
    h = f .* sin(x(idx,3).*log(abs(x(idx,1)-Time+eps)));

    f = reshape(f,[],1);
    g = reshape(g,[],1);
    h = reshape(h,[],1);
    K = [ones(N,1),f,g,h];

    mdl_liner = regress(Price,K);
    Loss(idx,1) = mean((K*mdl_liner-Price).^2);
end

end
