%% Mackey-Glass NDDE
clear
clc
close all
addpath('utils');
plottingPreferences()
%%
lossfun='simu'; 
% lossfun='deriv';
lm = 'L2';  % loss measure, which norm to use
% w = 'weighted';
w = 'average';
if strcmp(lossfun,'deriv')
    numIter = 2000;
    plotFrequency = 50;
    learnRate = 0.01;
    learnRate_tau = 0.001;
    batchSize = 20; 
    steps = 1;
elseif strcmp(lossfun,'simu')
    numIter = 1000;
    plotFrequency = 10;
    learnRate = 0.1;
    learnRate_tau = 0.1;
    batchSize = 10;
    steps = 10;
end

sys = 'MG_chao10_addhist_clean'; % chao/LC % addhist/nodrop/drop10s

if contains(sys,'chao10')
    truetau = 1;
    T_tr = 15;    % this includes the initial history!
elseif contains(sys,'chao15')
    truetau = 1.5;
    T_tr = 30;    % this includes the initial history!
elseif contains(sys,'LC05')
    truetau = 0.5;
    T_tr = 15;    % this includes the initial history!
elseif contains(sys,'EQ02')
    truetau = 0.2;
    T_tr = 15;    % this includes the initial history!
end

beta = 4;
n = 9.65;
gamma = 2;
trueModel = @(t,x,xdelay) beta*xdelay/(1+xdelay^n)-gamma*x;
tr_c = 0.3;
dt =0.1;
% adjust the simulation accuracy/computation
dt_simu = 0.1;
gap = round(dt/dt_simu);     % simulation time step = dt/gap
tau_max = 2*truetau;


T_ts = 5;
if contains(sys,'drop10')
    Tst = 10; % drop first 10 second  
elseif contains(sys,'nodrop')
    Tst = 0;
elseif contains(sys,'addhist')
    Tst = -tau_max;  
end
tend = Tst+T_tr+T_ts; 

clean_data = cell(1,length(tr_c));
xdata = cell(1,length(tr_c));
xdata_all= cell(1,length(tr_c));
targets_all = cell(1,length(tr_c));
targets_dx =cell(1,length(tr_c));

hist_time = Tst:dt:Tst+tau_max;
tt_all = Tst:dt:tend;      % matching xdata_all
tt_target = Tst+tau_max:dt:tend;
t_deriv = Tst+tau_max:dt:Tst+T_tr;
for i = 1:length(tr_c)
    histvec_int = tr_c(i);
    hist_int = @(t)histvec_int;
    sol = dde23(trueModel,truetau,hist_int,0:dt:tend);
    clean_data_t = 0:0.01:tend;
    clean_data{i}  = deval(sol,clean_data_t);
    if contains(sys,'addhist')
        clean_data{i}  = [repmat(histvec_int,1,round(tau_max/0.01)) clean_data{i}];
        
    else
        clean_data{i} = clean_data{i}(round(Tst/0.01)+1:end);
    end
    clean_data_t = Tst:0.01:tend;

    xdata{i}  = deval(sol,0:dt:tend);
    if contains(sys,'noise')
        sig = 0.05;
        rng(1)
        xdata{i} = xdata{i}+sig*randn(1,length(0:dt:tend));
    end
    if contains(sys,'addhist')
        xdata_all{i} = [repmat(histvec_int,1,round(tau_max/dt)) xdata{i}];
        targets_all{i} = xdata{i};
        targets_dx{i} = ctrDiff(targets_all{i},tt_target);
    else
        % drop data
        xdata_all{i} = xdata{i}(round(Tst/dt)+1:end);
        targets_all{i} = xdata_all{i}(round(tau_max/dt)+1:end);
        targets_dx{i} = ctrDiff(targets_all{i},tt_target);
    end
    
end

figure(1)
set(gcf,'Position',[100 100 400 300])
hold on;
for i = 1:length(xdata)
    plot(clean_data_t,clean_data{i}(:))
    scatter(tt_all,xdata_all{i}(:),5,'filled')
end
plot([Tst+tau_max Tst+tau_max],[0 1.5],'k--')
plot([Tst Tst],[0 1.5],'k-')
plot([Tst+T_tr Tst+T_tr],[0 1.5],'k-')
box on;
title(['MG equation, ',num2str(length(xdata)),' datasets'])
xlim([Tst tend])
ylabel('$x(t)$')
xlabel('$t$')

%%
figure(10)
set(gcf,'Position',[100 100 400 300])
hold on;
for i = 1:length(xdata)
    plot(clean_data{i}(1:round((T_tr-truetau)/0.01)),clean_data{i}(round(truetau/0.01)+1:round((T_tr)/0.01)))
    scatter(xdata_all{i}(1:round((T_tr-truetau)/dt)),xdata_all{i}(round(truetau/dt)+1:round((T_tr)/dt)),10,'filled')
end
box on;
axis equal
xlabel('$x(t-\tau)$')
ylabel('$x(t)$')
xlim([0.2 1.4])
ylim([0.2 1.4])
yticks([0.2 0.8 1.4])
xticks([0.2 0.8 1.4])
%% learning 
hiddenSize = 5;
nx = 1; % 1 state
% tau = dlarray(rand(1));
tau = dlarray(1.5);
nd = length(tau);

NDDE = struct;
NDDE.fc1 = struct;
sz = [hiddenSize nx*(1+nd)];
NDDE.fc1.Weights = initializeGlorot(sz);
NDDE.fc1.Bias    = initializeZeros([sz(1) 1]);

NDDE.fc2 = struct;
sz = [hiddenSize hiddenSize];
NDDE.fc2.Weights = initializeGlorot(sz);
NDDE.fc2.Bias    = initializeZeros([sz(1) 1]);

NDDE.fc3 = struct;
sz = [nx hiddenSize];
NDDE.fc3.Weights = initializeGlorot(sz);

%%
start = tic;
Loss = zeros(1,numIter);
Tau_tr = zeros(length(tau),numIter);

gradDecay = 0.9;
sqGradDecay = 0.999;
aveGrad   = [];
aveSqGrad = [];
aveGrad_tau = [];
aveSqGrad_tau = [];
nsets = length(xdata);
nn = 0;
for iter = 1:numIter
    if strcmp(lossfun,'simu')
        hist_v = cell(1,nsets*batchSize);
        dlhist_v = cell(1,nsets*batchSize);
        targets_tr= cell(1,nsets*batchSize);
        tt = 0:dt:dt*steps;       % the sampling time
        t_simu = 0:(dt/gap):dt*steps; % simulation time step
        hist_t = -tau_max:dt:0;
        st = randperm(length(t_deriv)-steps,batchSize);
        for kk = 1:nsets
            for batch = 1:batchSize
                hist_v{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch):st(batch)+length(hist_t)-1);
                dlhist_v{(kk-1)*batchSize+batch} = dlarray(hist_v{(kk-1)*batchSize+batch});
                targets_tr{(kk-1)*batchSize+batch} = xdata_all{kk}(:,st(batch)...
                        +length(hist_t)-1:st(batch)+length(hist_t)-1+steps);
            end
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddesimu,tt,t_simu,NDDE,tau,dlhist_v,hist_t,targets_tr,lm,st,w);
    elseif strcmp(lossfun,'deriv')
        targets_dx_tr = cell(1,nsets);
        st = randperm(length(t_deriv),batchSize);
        t_batch = t_deriv(st);
        for kk = 1:nsets
            targets_dx_tr{kk} = targets_dx{kk}(st);
        end
        [grads_NDDE,grads_tau,loss] = dlfeval(@grad_ddederiv,tt_all,t_batch,NDDE,tau,xdata_all,targets_dx_tr,lm,st,w);
    end
    
    Tau_tr(:,iter)=extractdata(tau');
    currentLoss = double(extractdata(loss));
    Loss(iter) = currentLoss;
    if Loss(iter)<=min(Loss(1:iter))
        NDDE_best = NDDE;
        tau_best = tau;
        nn = 0;
    elseif iter>10 && Loss(iter)>mean(Loss(iter-10:iter))
        nn = nn+1;
    end

    if nn == 0 %mod(iter,plotFrequency) == 0  || iter == 1
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
      
        figure(2)
        set(gcf,'Position',[50 50 1000 300])
        clf
        subplot(2,3,1)
        semilogy(1:iter, Loss(1:iter),'b')
        xlabel('Iteration')
        xlim([1,iter+1])
        ylabel('Loss')
        title("Training loss")

        subplot(2,3,4)
        hold on
        for k = 1:nd
            plot(1:iter, Tau_tr(k,1:iter),'b','LineWidth',2)
        end
        plot(1:iter+1, truetau*ones(1,iter+1), 'k--',LineWidth=2)
        ylim([0,tau_max])
        xlim([1,iter+1])
        xlabel('Iteration')
        ylabel('$\tau$')
        hold off
        box on
        title("Learned delay")
       
        subplot(2,3,2)
        % simulation
        W1 = double(extractdata(NDDE.fc1.Weights));
        W2 = double(extractdata(NDDE.fc2.Weights));
        W3 = double(extractdata(NDDE.fc3.Weights));
        b1 = double(extractdata(NDDE.fc1.Bias));
        b2 = double(extractdata(NDDE.fc2.Bias));
        NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[x;reshape(xdelay,[],1)]+b1)+b2);

        % predict the whole traj
        sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target);
        tt_target_fine=Tst+tau_max:0.01:tend;
        y = deval(sol1,tt_target_fine);
        hold on
        plot(tt_target_fine,y,'b','LineWidth',1)
        plot(hist_time,xdata_all{1}(:,1:length(hist_time)),'b')
        scatter(tt_target,targets_all{1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time,xdata_all{1}(:,1:length(hist_time)),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[0 1.5],'k--', 'LineWidth',1)
        xlim([hist_time(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        ylim([0 1.5])
        title('Network simulation')

        subplot(2,3,5)
        hold on
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_fine-tau(k))',nx,[])];
        end
        dx = ddeModel(tt_target_fine,x,xdelay,NDDE);
        plot(tt_target_fine,extractdata(dx),'b')
        scatter(tt_target,targets_dx{1},5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[-2 2],'k--', 'LineWidth',1)
        xlim([hist_time(1) tend])
        ylim([-2 2])
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on

        subplot(2,3,3)
        % predict only for testing
        hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
        tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
        sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
            @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);
        tt_target_ts_fine = Tst+T_tr:0.01:Tst+T_tr+T_ts;
        y = deval(sol1,tt_target_ts_fine );
        hold on
        plot(tt_target_ts_fine ,y,'b','LineWidth',1)
        plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
        scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,'filled','MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[0 1.5],'k--', 'LineWidth',1)
        xlim([hist_time_ts(1) tend])
        xlabel('$t$')
        ylabel('$x$')
        hold off
        box on
        ylim([0 1.5])
        title('Network simulation')

        subplot(2,3,6)
        hold on
        % only testing
        x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
        xdelay = [];
        for k = 1:length(tau)
            xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
        end
        dx = ddeModel(tt_target_ts_fine ,x,xdelay,NDDE);
        plot(tt_target_ts_fine,extractdata(dx),'b')
        scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,"filled",'MarkerEdgeColor',"#D95319",'MarkerFaceColor',"#D95319")
        plot([Tst+T_tr Tst+T_tr],[-2 2],'k--', 'LineWidth',1)
        xlim([hist_time_ts(1) tend])
        ylim([-2 2])
        xlabel('$t$')
        ylabel('$\dot{x}$')
        title('Network direct prediction')
        hold off
        box on


    end
     
    [NDDE,aveGrad,aveSqGrad] = adamupdate(NDDE,grads_NDDE,aveGrad,aveSqGrad,iter,...
        learnRate,gradDecay,sqGradDecay);
    [tau,aveGrad_tau,aveSqGrad_tau] = adamupdate(tau,grads_tau,aveGrad_tau,aveSqGrad_tau,iter,...
        learnRate_tau,gradDecay,sqGradDecay);
    tau = min(max(0.00001,tau),tau_max-0.00001);
end

%% plot loss and delay path
if numIter == 0
    iter=0;
end

NDDE = NDDE_best;
tau = tau_best;
W1 = double(extractdata(NDDE.fc1.Weights));
W2 = double(extractdata(NDDE.fc2.Weights));
W3 = double(extractdata(NDDE.fc3.Weights));
b1 = double(extractdata(NDDE.fc1.Bias));
b2 = double(extractdata(NDDE.fc2.Bias));
NNmodel = @(t,x,xdelay) W3*tanh(W2*tanh(W1*[x;reshape(xdelay,[],1)]+b1)+b2);
%%
figure(5)
set(gcf,'Position',[100 100 300 200])
semilogy(1:iter, Loss(1:iter))
hold on
[~,min_id] = min(Loss(1:iter));
semilogy([min_id min_id],[min(Loss(1:iter)) max(Loss(1:iter))],'k:')
hold off
xlim([1,iter+1])
xlabel('Iteration')
ylabel('Loss')
ylim([min(Loss(1:iter)) max(Loss(1:iter))])
yticks([0.00001 0.0001 0.001 0.01 0.1])

figure(6)
set(gcf,'Position',[100 100 300 200])
hold on
for k = 1:nd
    plot(1:iter, Tau_tr(k,1:iter),LineWidth=2)
    
end
plot(1:iter+1, truetau*ones(1,iter+1), 'k--',LineWidth=2)
plot(1:iter+1, zeros(1,iter+1),'k--',LineWidth=2)
plot([min_id min_id],[0 tau_max],'k:')
ylim([0,tau_max])
xlim([1,iter+1])
xlabel('Iteration')
ylabel('Delays')
hold off
box on

figure(9)
set(gcf,'Position',[100 100 300 200])
% predict only for testing
hist_time_ts = Tst+T_tr-tau_max:dt:Tst+T_tr;
tt_target_ts = Tst+T_tr:dt:Tst+T_tr+T_ts;
sol1 = dde23(@(t,x,xdelay,theta)NNmodel(t,x,xdelay),double(extractdata(tau)),...
    @(t)interp1(tt_all,xdata_all{1}',t)',tt_target_ts);
tt_target_ts_fine = Tst+T_tr:0.01:Tst+T_tr+T_ts;
y = deval(sol1,tt_target_ts_fine );
hold on
plot(tt_target_ts_fine ,y,'b','LineWidth',1)
plot(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),'b')
scatter(tt_target_ts,targets_all{1}(:,end-length(tt_target_ts)+1:end),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
scatter(hist_time_ts,interp1(0:dt:tend,xdata{1}',hist_time_ts),5,...
    'filled','MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[0 1.5],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
xlabel('$t$')
ylabel('$x$')
hold off
box on
ylim([0 1.5])
title('Network simulation')

figure(11)
set(gcf,'Position',[100 100 300 200])
hold on
x = reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine)',nx,[]);
xdelay = [];
for k = 1:length(tau)
    xdelay = [xdelay; reshape(interp1(clean_data_t,clean_data{1}',tt_target_ts_fine-tau(k))',nx,[])];
end
dx = ddeModel(tt_target_ts_fine ,x,xdelay,NDDE);
plot(tt_target_ts_fine,extractdata(dx),'b')
scatter(tt_target_ts,targets_dx{1}(:,end-length(tt_target_ts)+1:end),5,...
    "filled",'MarkerEdgeColor',[0.5 0.5 0.5],'MarkerFaceColor',[0.5 0.5 0.5])
plot([Tst+T_tr Tst+T_tr],[-2 2],'k--', 'LineWidth',1)
xlim([hist_time_ts(1) tend])
ylim([-2 2])
xlabel('$t$')
ylabel('$\dot{x}$')
title('Network direct prediction')
hold off
box on

%% plot the testing
id = find(extractdata(tau)>0.2);
test_tau_NN = 0.00001*ones(size(tau));
test_tau = 0.8;
test_tau_NN(id) = test_tau;

ts_c = 0.9;
hist_ts = @(t)ts_c;
hist_t = -tau_max:dt_simu:0;
sol = dde23(trueModel,test_tau,hist_ts,0:dt_simu:tend);
test_simu1  = deval(sol,0:dt_simu:tend);
test_simu2 = ddeab4(trueModel,test_tau,hist_ts,0:dt_simu:tend);
dlhist_vec = dlarray(ts_c*ones(1,length(hist_t)));
dlhist_1 = @(t)interp1(hist_t,dlhist_vec',t)';

y = dde23(@(t,x,xdelay)NNmodel(t,x,xdelay),test_tau_NN,hist_ts,0:dt:tend);
y = deval(y,0:dt_simu:tend);

x1 = [ts_c*ones(size(hist_t)) test_simu1(2:end)];
x2 = [ts_c*ones(size(hist_t)) y(2:end)];
figure(7)
set(gcf,'Position',[100 100 300 300])
hold on
plot(x2(1:end-round(test_tau/dt_simu)),x2(1+round(test_tau/dt_simu):end),'b')
plot(x1(1:end-round(test_tau/dt_simu)),x1(1+round(test_tau/dt_simu):end),'r--')
ylabel('$x(t)$')
xlabel('$x(t-\tau)$')
xlabel('$x(t-\tau)$')
ylabel('$x(t)$')
axis equal
xlim([0.2 1.4])
ylim([0.2 1.4])
yticks([0.2 0.8 1.4])
xticks([0.2 0.8 1.4])
hold off
box on

legend(['NDDE,' ...
    ' trained on $\tau = 0.5$'],'DDE','location','best')
title('Network simulation on testing data')


figure(8)
set(gcf,'Position',[100 100 400 200])
hold on
plot([hist_t 0:dt_simu:tend],[ts_c*ones(size(hist_t)) y],'b')
plot([hist_t 0:dt_simu:tend],[ts_c*ones(size(hist_t)) test_simu1],'r--')
xlim([-tau_max tend])
box on;
ylim([0 1.6])
xlabel('$t$')
ylabel('$x(t)$')
title('Network simulation on testing data')

%%
Delay = double(extractdata(tau));
save(['net_',sys,'_',lossfun],'NDDE','Delay')
%%
function dx=ddeModel(t,x,xdelay,theta)
dx = theta.fc3.Weights*tanh(theta.fc2.Weights...
    *tanh(theta.fc1.Weights*[x;xdelay]+theta.fc1.Bias)...
    +theta.fc2.Bias);
end
