function [w, infos] = structured_QN(problem, in_options, par)

% Structured quasi-Newton method



    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    r1 = 0.1;
    r2 = 10;
    alpha = 10;
    r = in_options.column-5; 
    % set local options 
    local_options.mem_size = 5;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      


    % set paramters
    if options.batch_size > n
        options.batch_size = n;
    end   
    
    %if ~isfield(in_options, 'batch_hess_size')
        options.batch_hess_size = min(2000,ceil(0.01*n));
    %end    
    
        

    if options.batch_hess_size > n
        options.batch_hess_size = n;
    end    
           
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;  
    
    s_array = [];
    y_array = [];    
    u_old = w;
    u_new = zeros(d,1);  


    
    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.8f, optgap = %.4e\n', 'Structured_QN', epoch, f_val, optgap);
    end     

    % main loop
    nofup = 0;
    while (epoch < options.max_epoch)
            
            alpha = total_iter + 1;
            w0 = w;
            
            % update step-size
            step = options.stepsizefun(total_iter, options);                
         
            if epoch > 0              

                % calculate gradient
                options.batch_size = min(floor(options.batch_size*1.1),n);

                indice_h = randperm(n, options.batch_hess_size);     %Hessian batch
                indice_j = randperm(n, options.batch_size);          % gradient batch size
                grad_cur = problem.grad(w, indice_j);  % Gradient current (at k-th w)

                Q = problem.Yang_base(w,indice_h,r,par); % Nystrom Approximation (using random)

                if norm(g_prv) < r1
                    lambda = 2*r1/alpha*(norm(g_prv)+r1);
                elseif norm(g_prv) > r2
                    lambda = 2*norm(g_prv)/alpha*(norm(g_prv)+r2);
                else
                    lambda = 1/alpha;
                end
            
                Struc_qn  = - s4qn(s_array,y_array,gamma1,Q,lambda,grad_cur);

                % update w 
              
                w = w + (step*Struc_qn);  
            else
                indice_j = randperm(n, options.batch_size);          % gradient batch size
                grad_cur = problem.grad(w, indice_j);  % Gradient current (at k-th w)
                w = w - (step*grad_cur); 
                
            end
            
            g_prv = grad_cur; % grad at k-1 th epoch 
            
            grad_new =  problem.grad(w,indice_j);% at k-1-th w
                     
            
%             if epoch > 0            
                % store cavature pair
                s = w - w0;
                y = grad_new - grad_cur;
                s_array = [s_array s];
                y_array = [y_array y]; 

                
                % remove overflowed pair
                if(size(s_array,2)>options.mem_size)
                    s_array(:,1) = [];
                    y_array(:,1) = [];
                end     
              
                 gamma1 = (s'*y)/(y'*y);

%             end
            
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
            
                
            
            total_iter = total_iter + 1;
        
             
        %vr = norm(step*v-step*problem.grad(w,1:n))^2;

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + 2*options.batch_size;
        
        nofup = nofup + options.batch_size;
        
        if floor(nofup/(2*n))>epoch
            epoch = epoch + 1;
            
            % store infos
            [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

            % display infos
            if options.verbose > 0
                fprintf('%s: Epoch = %03d, cost = %.8f, optgap = %.4e,time=%.3f,S=%.3e,Y=%.3e\n', 'Struc_QN', epoch, f_val, optgap,elapsed_time,norm(s),norm(y));
            end
        end
   end
    
    if epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
      
    
end

