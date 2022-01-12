function [w, infos] = Nystrom_svrg(problem, in_options,reg,dp)
%rho is replaced by reg on 12th Jan 2022.

    % If dp = 1 then NSVRG-DP
    % else NSVRG

    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % set local options 
    local_options.sub_mode = 'Nystrom';  % SQN or SVRG-SQN or SVRG-LBFGS
    local_options.mem_size = 20;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    
    %%%
    RH=[]; rh_old = 0;
    %%%
    col = options.column;
    K = options.clusters;
    P = options.partitions;
    
    
    if dp==1
     %[dppX,dppy,dpp_idxp_s,dpp_idxn_s] = %DPP(X',y',P,K,plott); P=partitions, K=clusters 
        [dppX,dppy,dpp_idxp_s,dpp_idxn_s] = DPP(problem.x_train',problem.y_train',P,K,0);
    end
    

    % set paramters
    if options.batch_size > n
        options.batch_size = n;
    end   
    
    if ~isfield(in_options, 'batch_hess_size')
        options.batch_hess_size = 20 * options.batch_size;
    end    

    if options.batch_hess_size > n
        options.batch_hess_size = n;
    end    
    
      
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;

    num_of_bachces = floor(n / options.batch_size)*2;     


    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', 'Nystrom_svrg', epoch, f_val, optgap);
    end     


    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while (epoch < options.max_epoch)
            perm_idx = [1:n 1:n];

            % compute full gradient
            %full_grad_new = problem.grad(w,1:n);
            full_grad_new = problem.full_grad(w);
            % count gradient evaluations
            grad_calc_count = grad_calc_count + n; 


            % store w for SVRG
            w0 = w;
            full_grad = full_grad_new;

      


        for j = 1 : num_of_bachces

               if j==1 
                  
%                   rng(j);
                    set = randperm(d,col);
                   
                    if any(isnan(full_grad)) || any(isinf(full_grad)) || any(isnan(w0)) || any(isinf(w0))
                    return;
                    end
                    
                    if dp==1
                    dppi=mod(epoch,P)+1;
                    
                   % Compute Z from C and M
                   % fprintf('Size of G = %d\n',length(G)); [dpp_idxp_s{dppi};dpp_idxn_s{dppi}]
                    [Z,fn1,apta] = problem.app_hess(w0,[dpp_idxp_s{dppi};dpp_idxn_s{dppi}],set,0);
                    else
                    [Z,fn1,apta] = problem.app_hess(w0,1:n,set,0);
                    end
                %lam = 1e-3;%norm(full_grad); % Norm of full_gradient
                lk = length(set); % k: colums
               % HI = inv(H); % Hessian Inverse
                
%                 if rho==0 %;%Nystrom regularized
% 
%                         rho1 = 1e-4*norm(Z,'fro');
%                     %    RH = [RH rho1];
%                         nfg = 1/rho1;
%                 else
%                 nfg = 1/rho;
%               %  RH = [RH rho1];
%                 end
                
                nfg = 1/reg;
                Ey = eye(lk);
                Q = Z/(Ey+nfg*(Z'*Z));
                
                
                
                end
            % update step-size
            step = options.stepsizefun(total_iter, options);                
            
            
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            
                     % calculate variance reduced gradient

                     grad_w0 = problem.grad(w0,indice_j);
                     grad = full_grad + grad - grad_w0;    

                     %NI =  nfg*eye(d) - (nfg)*Z/(Ey+nfg*(Z'*Z))*Z';
                     vect = Z'*grad;
                    
                     
                     NI = nfg*(grad - nfg*(Q*vect)); 
                
                     v = step*NI;
  
                     w = w - v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end                                       
            
            total_iter = total_iter + 1;
           
        end
        
             
        %vr = norm(step*v-step*problem.grad(w,1:n))^2;

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * options.batch_size;        
        epoch = epoch + 1;
        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.6e, optgap = %.4e, time=%0.3f\n', 'Nystrom_svrg', epoch, f_val, optgap,elapsed_time);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
       % infos.RHO = RH;
end

