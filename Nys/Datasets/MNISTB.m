function data = MNISTB(seed,reg,ler)
    M = load('mnist-b.mat');     
    %Data normalization (0 mean, unit varianve)
    D=M.x_train;
    [n,d]=size(D);
    D1=M.x_test;
    s = std(D);
    s(s==0)=1;
    m=mean(D);
    D = (D-m)./s;    
    D = [D  ones(size(D,1),1)];
    D1 = (D1-m)./s;    
    D1 = [D1  ones(size(D1,1),1)];
    
    rng('default');
    perm = randperm(n);
    A =  D(perm,:);
    B = M.y_train(perm);

    data.x_test = D1';
    data.y_test = M.y_test';
    data.x_train = A';
    data.y_train = B';
    
    fprintf('This is MNISTB train data with n=%d, d=%d\n',size(data.x_train'));


    fprintf('This is MNISTB test data with n=%d, d=%d\n',size(data.x_test'));

    %Initial point with different random seed
    rng(seed);
    data.w_init = randn(d+1,1);

    
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      tic;
%     options.max_epoch=2;
%     options.batch_size = 64;
%     % define problem definitions
%     problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
%     % For large data set, use sub-sample instead of data.x_train
%     options.w_init = inv(data.x_train*data.x_train'+reg*eye(size(data.x_train,1)))*data.x_train*data.y_train'; 
%     %options.w_init = data.w_init;   
%     options.step_alg = 'fix';
%     options.step_init = ler; 
%     options.verbose = 2;
%    % [w_sgd,~] = sgd(problem,options);
%     data.w_init = options.w_init;
%      toc                   
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
end