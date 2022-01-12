RUNS = 3;
EPOCHS = 50;
lambdas = [1e-1 1e-2 1e-3 1e-4];
etas = [ 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 ];
rhos = [ 0 1e3 1e2 1e1 1e0 1e-1 1e-2 1e-3 1e-4];
COLS = [-1 10 50 100 500];
BSS = [32 64 128];
path = 'result_s2qn/';
close all;
datasets = {
    'REALSIM'   %1
    'CIFAR10B'  %2
    'MNISTB'    %3
    'EPSILON'   %4
    'ADULT'     %5
    'W8A'       %6
    'ALLAML'    %7
    };

lw = 2;%RUNS;
ms = 8;
methods = {
    'NEWTON'
    'Structured_QN'
    'Structured_QF'
    'SVRG-LBFGS'
    'OBFGS'
    'SVRG-SQN'
    'SQN'
    'SGD'
    'SVRG'
    'adam'
    'Nystrom_sgd_dpp'
    'Nystrom_sgd'
    'Nystrom_svrg'
    'Nystrom_svrg_dpp'
    };

for j=1:length(rhos)
    Name = strcat(path,datasets{5}, sprintf('/K%d_B%d_%s_%.1e_R_%.1e_rho_%.1e_run_%d.mat', COLS(2), BSS(1), methods{14}, etas(4), lambdas(2), rhos(j), 1));
%     disp(Name);
    if exist(Name, 'file')
        d = load(Name);
        endi=min(length(d.info_s1.iter),EPOCHS+1);
        if ~isfield(d.info_s1,'val_cost') || length(d.info_s1.iter) < EPOCHS+1
            continue;
        else
%                 disp(Name);
        end

        cost = d.info_s1.cost(1:endi);
        vcost = d.info_s1.val_cost(2:endi);
        acc = d.info_s1.acc_tr(1:endi);
        vacc = d.info_s1.acc_val(2:endi);
        time = d.info_s1.time(1:endi);
        fprintf('K%d_B%d_%s_%.1e_R_%.1e_rho_%.1e: %0.6f - %0.6f\n', COLS(2), BSS(1), methods{14}, etas(4), lambdas(2), rhos(j), trapz(1:endi,cost), min(cost));
        hold on;
        plot(1:endi,log(cost),'displayname',Name,'linewidth',2);
    end
end