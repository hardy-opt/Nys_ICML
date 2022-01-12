function  nys_curve_exp(darg) %4.8446772001339822e-01
%     clc;
%     clear;
    close all;
    addpath(genpath(pwd));
   % addpath('/data/Datasets/'); % Dataset repository
    NUM_RUN = 3;
    NUM_EPOCH = 3;
    P = 10;  %Partition for DP sampling
    K = 50;  % No. of clusters for DP sampling
    dat = strcat('results_Jan22/',darg);  % result path
    method = {'NSVRG-D', 'NSGD-D','NSVRG','NSGD', 'Structured_QN', 'Structured_QF'};
    omethod = {'SVRG-LBFGS', 'SVRG-SQN', 'adam', 'SQN', 'OBFGS', 'SVRG', 'SGD', 'NEWTON'};
    BATCHES = [64];% 64 128];
    COLS = [20];% 100];
    for s=1:NUM_RUN
        for reg=[1]% 0.1 0.01 0.001 0.0001]
            for step = 0.001%[1 0.1 0.01 0.001 0.0001 0.00001]
                data = loaddata(s, reg, step, dat);                   
                %for rho = [10 ]%1000 100 10 1 0.1 0.01 0.001]
                    for m=1:4
                        for COL =  COLS 
                            if COL > size(data.x_train,1)
                                break;
                            end
                            for BATCH_SIZE = BATCHES
                                if BATCH_SIZE > size(data.x_train,2)
                                    break;
                                end
                                %rng(s); % do not remove 
                                fprintf('K%d - B%d - %s - Reg:%f - Step:%f - Run:%d\n', COL, BATCH_SIZE, method{m}, reg, step, s);
                                options.max_epoch=NUM_EPOCH;    
                                %problem = linear_svm(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                                options.w_init = data.w_init;   
                                options.step_alg = 'fix';
                                options.step_init = step; 
                                options.verbose = 2;
                                options.batch_size = BATCH_SIZE;
                                options.column = COL;
                                options.partitions = P;
                                options.clusters = K;

                                Name = sprintf('%s/K%d_B%d_%s_%.1e_R_%.1e_run_%d.mat',dat, COL, BATCH_SIZE, method{m},options.step_init,reg,s);

                                
                                if m==1
                                   [w_s1, info_s1] = Nystrom_svrg(problem, options,reg,1);  % NSVRG-D
                                   save(Name,'info_s1');
                                elseif m==2
                                    options.step_alg = 'decay-2'; %decay
                                    [w_s1, info_s1] = Nystrom_sgd(problem, options,reg,1); % NSGD-D
                                    save(Name,'info_s1');
                                elseif m==3
                                   [w_s1, info_s1] = Nystrom_svrg(problem, options,reg,0);  % NSVRG
                                   save(Name,'info_s1');
                                elseif m==4
                                    options.step_alg = 'decay-2'; %decay
                                    [w_s1, info_s1] = Nystrom_sgd(problem, options,reg,0); % NSGD
                                    save(Name,'info_s1');
                                elseif m==5
                                    %if rho==rho(1)
                                    options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = structured_QN(problem, options,1);  %Nystrom
                                    save(Name,'info_s1');
                                    %end
                                elseif m==6
                                    %if rho==rho(1)
                                    options.step_alg = 'decay-2';
                                    [w_s1, info_s1] = structured_QN(problem, options,0); %Fisher
                                    save(Name,'info_s1');
                                    %end
                                end 
                            end
                            
                        end
                    end
                %end
                 for BATCH_SIZE = BATCHES
                    if BATCH_SIZE > size(data.x_train,2)
                        break;
                    end
                    for m=1:7
                        
                        fprintf('%s - Reg:%f - Step:%f  - Run:%d\n', omethod{m}, reg, step, s);
                        options.max_epoch=NUM_EPOCH;    
                        problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
                        options.w_init = data.w_init;   
                        options.step_alg = 'fix';
                        options.step_init = step; 
                        options.verbose = 2;
                        options.batch_size = BATCH_SIZE;

                        Name = sprintf('%s/B%d_%s_%.1e_R_%.1e_run_%d.mat',dat,BATCH_SIZE,omethod{m},options.step_init,reg,s);

                        if m==1    
                            options.sub_mode='SVRG-LBFGS';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==2
                            options.sub_mode='SVRG-SQN';
                            %options.sub_mode= 'Lim-mem';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==3
                            options.step_alg = 'decay-2'; 
                            options.sub_mode='Adam';
                            [w_s1, info_s1] = adam(problem, options);
                        elseif m==4
                            options.store_grad = 0;
                            options.sub_mode = 'SQN';
                            options.step_alg = 'decay-2';
                            [w_s1, info_s1] = slbfgs(problem, options);
                        elseif m==5
                            options.sub_mode = 'Lim-mem';
                            [w_s1, info_s1] = obfgs(problem, options);
                        elseif m==6
                             %options.step_alg = 'decay-2'; 
                             [w_s1, info_s1] = svrg(problem, options);
                        elseif m==7
                            options.step_alg = 'decay-2'; 
                            [w_s1, info_s1] = sgd(problem, options);
                        elseif m==8
                            options.sub_mode = 'STANDARD';
                            %options.regularized = true;
                            options.step_alg = 'backtracking';
                            %options.max_epoch=5;
                            [w_s1, info_s1] = newton(problem, options);
                        end                    
                        save(Name,'info_s1');
                    end
                end
            end
        end
    end
end

function [data]=loaddata(s,reg,step,dat)
    strs = strsplit(dat,'/');
    if strcmp(strs{end}, 'REALSIM')
        data = REALSIM(s,reg,step);
    elseif strcmp(strs{end}, 'CIFAR10B')
        data = CIFAR10B(s,reg,step);
    elseif strcmp(strs{end}, 'MNISTB')
        data = MNISTB(s,reg,step);
    elseif strcmp(strs{end}, 'EPSILON')
        data = EPSILON(s,reg,step);
    elseif strcmp(strs{end}, 'ADULT')
        data = ADULT(s,reg,step);
    elseif strcmp(strs{end}, 'W8A')
        data = W8A(s,reg,step);
    elseif strcmp(strs{end}, 'ALLAML')
        data = ALLAML(s,reg,step);
    elseif strcmp(strs{end}, 'SMK_CAN')
        data = SMK_CAN(s,reg,step);
    elseif strcmp(strs{end}, 'GISETTE')
        data = GISETTE(s,reg,step);    
    else
        disp('Dataset tho de');
    end
end
