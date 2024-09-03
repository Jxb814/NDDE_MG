function [grads_theta,grads_tau,loss] = grad_ddederiv(tt_all,tt_int,dltheta,dltau,input_x,targets_dx,lm,st,w)
loss = 0;
for i = 1:length(targets_dx)
    x = reshape(interp1(tt_all,input_x{i}',tt_int)',size(input_x{1},1),[]);
    xdelay = [];
    for k = 1:length(dltau)
        xdelay = [xdelay; reshape(interp1(tt_all,input_x{i}',tt_int-dltau(k))',size(input_x{1},1),[])];
    end
    dldx = ddeModel(tt_int,x,xdelay,dltheta);
    if strcmp(lm,'L1')
        if strcmp(w,'average')
            newloss = l1loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'weighted')
            newloss = l1loss(dldx,targets_dx{i},repmat(1./st,size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        end
    elseif strcmp(lm,'L2')
        if strcmp(w,'average')
            newloss = l2loss(dldx,targets_dx{i},'NormalizationFactor','all-elements','DataFormat','CT');
        elseif strcmp(w,'weighted')
            newloss = l2loss(dldx,targets_dx{i},repmat(1./st,size(dldx,1),1),'NormalizationFactor','all-elements','DataFormat','CT');
        end
    end
    loss = loss + newloss;
end
loss = loss/length(targets_dx);
grads_theta = dlgradient(loss,dltheta);
grads_tau = dlgradient(loss,dltau); 
end