function [dppX,dppy,dpp_idxp_s,dpp_idxn_s] = DPP(X,y,P,K,plot)
    %  [dppX,dppy,dpp_idxp,dpp_idxn] = DPP(X,y,P,K,plot)  %%%% plot 0/1
    %  dppX{i}, dppy{i} data partition. 
    %  dpp_idxp{i} idx of Xp and dpp_idxn{i} idx of Xn
    %  X is [n x d]

    Xp=X(y==+1,:);
    idx=kmeans(Xp,K,'MaxIter',10);
    for j = 1:P
        dpp_idxp{j}=[];
    end
    dpj=0;
    for i=1:K
        idx_c=find(idx==i);
        m = size(idx_c,1);
        idx_c=idx_c(randperm(m,m));
        mc = floor(m/P);
        mr = mod(m,P);
        for j = 1:P
            dpp_idxp{j}=[dpp_idxp{j};idx_c(1:mc)];
            idx_c(1:mc)=[];
        end
        for j=1:mr
            dpp_idxp{dpj+1}=[dpp_idxp{dpj+1};idx_c(1)];
            idx_c(1)=[];
            dpj=mod(dpj+1,P);
        end
    end
    
    for j = 1:P
        dpp_idxn{j}=[];
    end
    Xn=X(y==-1,:);
    idx=kmeans(Xn,K,'MaxIter',10);
    for i=1:K
        idx_c=find(idx==i);
        m = size(idx_c,1);
        idx_c=idx_c(randperm(m,m));
        mc = floor(m/P);
        mr = mod(m,P);
        for j = 1:P
            dpp_idxn{j}=[dpp_idxn{j};idx_c(1:mc)];
            idx_c(1:mc)=[];
        end
        for j=1:mr
            dpp_idxn{dpj+1}=[dpp_idxn{dpj+1};idx_c(1)];
            idx_c(1)=[];
            dpj=mod(dpj+1,P);
        end
    end
    
    
    for j = 1:P
        dppX{j}=[Xp(dpp_idxp{j},:);Xn(dpp_idxn{j},:)];
        dppy{j}=[ones(size(dpp_idxp{j},1),1);-ones(size(dpp_idxn{j},1),1)];
    end
    
    if plot==1
        plot_pc(X,y,dpp_idxp,dpp_idxn)
    end
    spos= find(y==+1);
    sneg= find(y==-1);
    for j = 1:P
        dpp_idxp_s{j}=spos(dpp_idxp{j});
        dpp_idxn_s{j}=sneg(dpp_idxn{j});
    end
end

function plot_pc(X,y,dpp_idxp,dpp_idxn)
    [w,pc]=pca(X);
    pcp=pc(y==1,:);
    pcn=pc(y==-1,:);
%     disp([size(pcp),size(pcn)] );
    figure;
    subplot(1,3,1);  %%% 100% data
    plot(pcn(:,1),pcn(:,2),'r.');hold on;
    plot(pcp(:,1),pcp(:,2),'bx');hold on;
    pos=sprintf('%d',size(pcp,1));
    neg=sprintf('%d',size(pcn,1));
    legend(neg,pos);
    ylabel('Principal Component-2')
    title('Full data')
    subplot(1,3,2); %%% 10% data with 1 partition
    plot(pcn(dpp_idxn{1},1),pcn(dpp_idxn{1},2),'r.');hold on;
    plot(pcp(dpp_idxp{1},1),pcp(dpp_idxp{1},2),'bx');hold on;
    pos=sprintf('%d',size(dpp_idxp{1},1));
    neg=sprintf('%d',size(dpp_idxn{1},1));
    legend(neg,pos);
    xlabel('Principal Component-1')
    title('Partition 1')
    subplot(1,3,3); %%% 10% data with 1 partition
    plot(pcn(dpp_idxn{2},1),pcn(dpp_idxn{2},2),'r.');hold on;
    plot(pcp(dpp_idxp{2},1),pcp(dpp_idxp{2},2),'bx');hold on;
    pos=sprintf('%d',size(dpp_idxp{2},1));
    neg=sprintf('%d',size(dpp_idxn{2},1));
    legend(neg,pos);
    title('Partition 2')
end