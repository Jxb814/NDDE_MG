function [grads_theta,grads_tau,loss] = grad_ddesimu_gap(tt,t_simu,dltheta,dltau,dlhist_vec,hist_t,targets,lm,st,w)
loss = 0;
for i = 1:length(dlhist_vec)
    dlhist = @(t)interp1(hist_t,dlhist_vec{i}',t)';
    dlx = dl_mdde_ab4(@(t,x,xdelay,theta)ddeModel(t,x,xdelay,theta),dltheta,dltau,dlhist,t_simu);
    gap = round((tt(2)-tt(1))/(t_simu(2)-t_simu(1)));
    dlx = dlx(:,1:gap:end);
    if strcmp(lm,'L1')
        newloss = l1loss(dlx(:,2:end),targets{i}(:,2:end),'NormalizationFactor','all-elements','DataFormat','CT');
    elseif strcmp(lm,'L2')
        newloss = l2loss(dlx(:,2:end),targets{i}(:,2:end),'NormalizationFactor','all-elements','DataFormat','CT');
    end
    if strcmp(w,'weighted')
        newloss = 1/sqrt(st(mod(i-1,length(st))+1))*newloss;
    end
    
    loss = loss + newloss;
end
loss = loss/length(dlhist_vec);
grads_theta = dlgradient(loss,dltheta);
grads_tau = dlgradient(loss,dltau);
end