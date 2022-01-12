function data = REALSIM(seed,reg,ler)

    M = load('realsim.mat'); 
    %data matrix M loads four files (x_train, y_train, x_test, y_test)
    
    [n,d] = size(M.X);

    D = full(M.X);
    
    %Data normalization (0 mean, unit varianve)
    s = std(D);
    s(s==0)=1;
    m=mean(D);
    D = (D-m)./s;
    
    D = [D  ones(n,1)];
    
    rng(seed);
    perm = randperm(n);
    A =  D(perm,:);
    B = M.y(perm);
    f = floor(n/5);

    idx = f*(seed-1)+1:f*seed;

    idxv = ones(n,1);
    idxv(idx) = 0;
    data.x_test = A(idxv==0,:)';
    data.y_test = B(idxv==0);
    data.x_train = A(idxv==1,:)';
    data.y_train = B(idxv==1);
    
    fprintf('This is Realsim train data with n=%d, d=%d\n',size(data.x_train'));


    fprintf('This is Realsim test data with n=%d, d=%d\n',size(data.x_test'));

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
%     
    
    
end