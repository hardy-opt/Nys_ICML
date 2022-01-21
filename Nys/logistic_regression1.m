classdef logistic_regression1
    
   
    
    properties
       name;    
        dim;
        samples;
        lambda;
        classes;  
        hessain_w_independent;
        d;
        n_train;
        n_test;
        x_train;
        y_train;
        x_test;
        y_test;
        x_norm;
        x;  
    end
    
    methods
        
        function obj = logistic_regression1(x_train,y_train,x_test,y_test,varargin)
            obj.x_train = x_train;
            obj.y_train = y_train;
            obj.x_test = x_test;
            obj.y_test = y_test;
            
            if nargin < 5
                obj.lambda = 0.001;
            else
                obj.lambda = varargin{1};
            end
            
            obj.d = size(obj.x_train,1);
            obj.n_train = length(y_train);
            obj.n_test = length(y_test);
            obj.name = 'logistic_regression';
            obj.dim = obj.d;
            obj.samples = obj.n_train;
            obj.classes = 2;
            obj.x_norm = sum(obj.x_train.^2,1);
            obj.x = obj.x_train;
        end
        
        function f = cost(obj,w)
            
            %f = sum(log(1+exp(-obj.y_train.*(w'*obj.x_train)))/obj.n_train,2) + obj.lambda*(w'*w)/2;
            
            sigmod_result = sigmoid(obj.y_train.*(w'*obj.x_train));
            sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            f = -sum(log(sigmod_result),2)/obj.n_train + obj.lambda * (w'*w) / 2;
%             
        end
        
         function f = cost_batch(obj, w, indices)

            f = -sum(log(sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))/obj.n_train,2) + obj.lambda * (w'*w) / 2;

        end
        
      %  function g = sgrad(obj,w,ind)  %%% Stochastic gradient for "one" single index
            
            %g = y_train.*x_train * (1-sigmoid(y_train.*(w'*x_train)));
            
       %     g = -sum(ones(obj.d,1) * obj.y_train(ind).*obj.x_train(:,ind) * (ones(1,length(ind))-sigmoid(obj.y_train(ind).*(w'*obj.x_train(:,ind))))',2)/length(ind)+ obj.lambda * w;
       % end
        
        function g = grad(obj,w,ind) %%% sum of stochastic gradient to get full gradient
            
            %g = y_train.*x_train * (1-sigmoid(y_train.*(w'*x_train)));
            
            xy = obj.y_train(ind).*obj.x_train(:,ind); 
            g = -sum(((ones(length(ind),1)-sigmoid(xy'*w)).*xy')',2)/length(ind)+obj.lambda*w;
            %g =-sum(ones(obj.d,1) * xy * (ones(1,length(indices))-sigmoid(w'*xy))',2)/length(indices)+ obj.lambda * w;
            %g = -sum(ones(obj.d,1) * obj.y_train(indices).*obj.x_train(:,indices) * (ones(1,length(indices))-sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices))))',2)/length(indices)+ obj.lambda * w;
        end
        
        function g = full_grad(obj,w)
            
            g = grad(obj, w, 1:obj.n_train);
            
        end
        
        function h = hess(obj, w, indices)

            %org code
            %temp = exp(-1*(y_train(indices)').*(x_train(:,indices)'*w));
            %b = temp ./ (1+temp);
            %h = 1/length(indices)*x_train(:,indices)*(diag(b-b.^2)*(x_train(:,indices)'))+lambda*eye(d); 

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
           
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            h = 1/length(indices)* h1 * obj.x_train(:,indices)'+obj.lambda*eye(obj.d);
        end
        
        function h = full_hess(obj, w)

            h = hess(obj, w, 1:obj.n_train);

        end
       
        
        function hv = hess_vec(obj, w, v, indices)  %%% Hessian - vector multiplication

            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            hv = 1/length(indices)* h1 * (obj.x_train(:,indices)' * v) +obj.lambda*v;

        end
        
        function hv = fullhess_vec(obj, w, v)   %%% Hessian - vector multiplication
            
                
            hv = hess_vec(obj,w,v,1:obj.n_train);

        end
        
        
        function [H,orta] = hessian_time(obj,w)
            tic;
            H = hess(obj,w,1:obj.n_train);
            orta = toc;
            
        end
        
        
        function ph = partial_hess(obj,w,indices)
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
           
            ph = 1/length(indices)* h1;

        end
        
        function phv = partial_hess_vec(obj,v,indices,ph)
            
            phv = ph* (obj.x_train(:,indices)' * v) +obj.lambda*v;
        end
        
        
          
        function dh = diag_hess(obj,w,indices)
            
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            thd2 = sigm_val.* (ones(1,length(indices))-sigm_val);
            %sqtthd = sqrt(thd2);
            %xthd2 = obj.x_train(:,indices).^2*thd2';
            xy = obj.x_train(:,indices).*obj.y_train(indices);
            xythd2 = xy.^2*thd2';
            dh = (1/length(indices))*xythd2 + obj.lambda*ones(obj.dim,1);
            %dh = (1/length(indices))*xthd2 + obj.lambda*ones(obj.dim,1);
            %dh = sum((obj.x_train(:,indices)*sqtthd').^2,2)/length(indices);
            %[n1,d1]= size(dh)
            %if any(isnan(sqtthd)) || any(isinf(sqtthd))
            %   fprintf(' sqt is naninf= \n');
            %end
            
%              h = hess(obj,w,indices);
%              H = diag(h);
%             
%              if all(dh == H)
% %                 
%               fprintf('digaonal elements are the same\n');
%              end
           % dh = dh + obj.lambda*ones(n1,d1);
         
        end
        
        
        function dh = full_diag_hess(obj,w)
           
            dh = diag_hess(obj,w,1:obj.n_train);
        end
        
        
        %%%%%%% Test cost
        function f = test_cost(obj,w)
            
            %f = sum(log(1+exp(-obj.y_train.*(w'*obj.x_train)))/obj.n_train,2) + obj.lambda*(w'*w)/2;
            
            sigmod_result = sigmoid(obj.y_test.*(w'*obj.x_test));
            sigmod_result = sigmod_result + (sigmod_result<eps).*eps;
            f = -sum(log(sigmod_result),2)/obj.n_test + obj.lambda * (w'*w) / 2;
            
        end
        
        %%%%%%%
        function p = prediction(obj, w,D)

            if strcmp(D,'Tr')
                D = obj.x_train;
            elseif strcmp(D,'Vl')
                D = obj.x_test;
            end
            p = sigmoid(w' * D);

            class1_idx = p>0.5;
            class2_idx = p<=0.5;         
            p(class1_idx) = 1;
            p(class2_idx) = -1;         

        end
        
        

        function a = accuracy(obj, y_pred,D)

            
            if strcmp(D,'Tr')
                D = obj.y_train;
                l = length(D);
            elseif strcmp(D,'Vl')
                D = obj.y_test;
                l = length(D);
            end
            a = sum(y_pred == D)/l; 

        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
         function [Z,fn,apptime] =  app_hess(obj,w,indices,set,H)
             
            tic;
           % fprintf('Approximate Hessian Started\n');
            sigm_val = sigmoid(obj.y_train(indices).*(w'*obj.x_train(:,indices)));
            c = sigm_val .* (ones(1,length(indices))-sigm_val); 
            
            h1 = (obj.x_train(:,indices) .* (obj.y_train(indices).^2 .* c));
            
            h2 = obj.x_train(set,indices)';

            a = 1/length(indices)* h1 * h2;
            
            l = length(set);
%             for i=1:l
%                 a(set(i),i) = a(set(i),i)+obj.lambda;
%             end
%removed lambda on 10th Jan 2022 (Nystrom on %just f and not on l2)
            
             aN = a;
             a = aN;

            C = a;
            A = a(set,:);
            if issparse(A)
                A = full(A);
            end
            [U,W,V] = svd(A);
            
%              if l==obj.d
%                 plot(diag(W),'LineWidth',5);
%                 title('Singular Values','FontSize',20);
%                 ax = gca;
%                  ax.XAxis.FontSize = 30;
%                  ax.YAxis.FontSize = 30;
%                 hold on;
%             end
            r = rank(W);
            D = diag(W);
            I = [1./D(1:r);zeros(l-r,1)];
            v = sqrt((I));
            B = U.*v';
            %B = U(:,1:r).*v';
            Z = C*B;
           % G = Z*Z';
            
            apptime = toc;
            %fprintf('Approximate Hessian Has been computed\n');
            %%%%%%%%%%%%%%%%%%%%%%
%             if p==1
%             tic;
%             fprintf('Original Hessian Started\n');
%             H = full_hess(obj,w);
%             fprintf('Original Hessian Has been computed\n');
%             origtime = toc;
%             elseif p~=1
%                 origtime = 0;
%             end
            %%%%%%%%%%%%%%%%%%%%%%
            
            %M = H - G;
            fn = 0;%norm(M,'fro');
          
            
%             if all(H(set,set)==a(set,:))
%                 if all(H(:,set)==a)
%                 fprintf('both Hessian are the same\n');
%                 end
%             end
%             
            
         end
        
        
        
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %%%%%   Structered Quasi-Newton
                  
        function Q = Yang_base(obj, w, indice, r, par)   %%% H : Base matrix approximated  by Nystrom arxiv version 1.

            sam = length(indice);
            dimension = size(w,1);
            
            if par == 1     % For Nystrom
            
                W = hess(obj, w, indice);  
            
            else            % Fisher Matrix
                
                G = zeros(dimension,sam);
                for e = 1:sam
                    G(:,e) =  grad(obj,w,indice(e));
                end
                W = (1/sam)*(G*G');
            end
                      

            O = rand(dimension,r);
            
            Z =  W*O;
            
            A = inv(O'*Z + 1e-5*eye(r));
            [U,Sg,V]=svd(A);
            sv = diag(Sg);
            if all((sv)>0)
                A = A + 1e-5*(eye(r));
                 C = chol(A);
            else
                
                A = A + eye(r);
                
                [U,Sg,V]=svd(A);
                sv = diag(Sg);
                
                if any(sv)==0 || any(sv)<0
                    
                    A = A + diag(A);
                    
                end  
                 C = chol(A);
                
            end
            
            Q = Z*C';
            
            
            
            
            
        end
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        
        function w_opt = calc_solution(obj, maxiter, method)

            if nargin < 3
                method = 'lbfgs';
            end        

            options.max_iter = maxiter;
            options.verbose = true;
            options.tol_optgap = 1.0e-24;
            options.tol_gnorm = 1.0e-16;
            options.step_alg = 'backtracking';

            if strcmp(method, 'sd')
                [w_opt,~] = sd(obj, options);
            elseif strcmp(method, 'cg')
                [w_opt,~] = ncg(obj, options);
            elseif strcmp(method, 'newton')
                options.sub_mode = 'INEXACT';    
                options.step_alg = 'non-backtracking'; 
                [w_opt,~] = newton(obj, options);
            else 
                options.step_alg = 'backtracking';  
                options.mem_size = 5;
                [w_opt,~] = lbfgs(obj, options);              
            end
        end
        
    end
end
