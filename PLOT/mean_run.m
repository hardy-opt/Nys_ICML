function [mu, sg] = mean_run(dataset, method, lambda, eta, rho, EPOCHS, run, COL, BS)
    i=0;
    ratio=0.1;
    K = containers.Map();
    K('REALSIM')=ceil(20959*ratio);
    K('CIFAR10B')=ceil(3073*ratio);
    K('MNISTB')=ceil(785*ratio);
    K('EPSILON')=ceil(2001*ratio);
    K('W8A')=ceil(301*ratio);
    K('ADULT')=ceil(124*ratio);
    K('ALLAML')=ceil(7130*ratio);
    
    for j = 1:run
        if rho > -1
            if COL>0
                Name = strcat(dataset, sprintf('K%d_B%d_%s_%.1e_R_%.1e_rho_%.1e_run_%d.mat', COL, BS, method, eta, lambda, rho, j));
            else
                Name = strcat(dataset, sprintf('K%d_B%d_%s_%.1e_R_%.1e_rho_%.1e_run_%d.mat', K(dataset), BS, method, eta, lambda, rho, j));
            end
        else
            Name = strcat(dataset, sprintf('B%d_%s_%.1e_R_%.1e_run_%d.mat', BS, method, eta, lambda, j));
        end
%         disp(Name);
        if exist(Name, 'file')
            d = load(Name);
            endi=min(length(d.info_s1.iter),EPOCHS+1);
            if ~isfield(d.info_s1,'val_cost') || length(d.info_s1.iter) < EPOCHS+1
                continue;
            else
%                 disp(Name);
            end
            
            i = i + 1;
            cost(i,:) = d.info_s1.cost(1:endi);
            vcost(i,:) = d.info_s1.val_cost(2:endi);
            acc(i,:) = d.info_s1.acc_tr(1:endi);
            vacc(i,:) = d.info_s1.acc_val(2:endi);
            time(i,:) = d.info_s1.time(1:endi);
        else
%             disp(Name);
            continue;
        end
    end
    mu = containers.Map();
    sg = containers.Map();
    if i>0
        mu('cost') = mean(cost, 1);
        sg('cost') = std(cost, [], 1);
        mu('val_cost') = mean(vcost, 1);
        sg('val_cost') = std(vcost, [], 1);
        mu('acc_tr') = mean(acc, 1);
        sg('acc_tr') = std(acc, [], 1);
        mu('acc_val') = mean(vacc, 1);
        sg('acc_val') = std(vacc, [], 1);
        mu('time') = mean(time, 1);
        mu('epoch') = [0:EPOCHS];
    end
end