function data = GISETTE(seed,reg,ler)

    M = load('gisette.mat'); 
    %data matrix M loads four files (x_train, y_train, x_test, y_test)
    
    [n,d] = size(M.x_train);
    
    D = M.x_train;
    
    %Data normalization (0 mean, unit varianve)
    s = std(D);
    s(s==0)=1;
    m=mean(D);
    D = (D-m)./s;
    
    D = [D  ones(n,1)];

    rng(seed);
    perm = randperm(n);
    A =  D(perm,:);
    B = M.y_train(perm);
    data.x_train = A';  
    data.y_train = B';
      
    fprintf('This is Gisette train data with n=%d, d=%d\n',size(data.x_train'));

    P = M.x_test;
    [nn,~] = size(M.x_test);
    %Data normalization (0 mean, unit varianve)
    s = std(P);
    s(s==0)=1;
    m=mean(P);
    P = (P-m)./s;
    
    P = [P  ones(nn,1)];
    
    rng(seed);
    perm = randperm(nn);
    data.x_test =  P(perm,:)';
    data.y_test = M.y_test(perm)';


    fprintf('This is Gisette test data with n=%d, d=%d\n',size(data.x_test'));

    %Initial point with different random seed
    rng(seed);
    data.w_init = randn(d+1,1);


end
