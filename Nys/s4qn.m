function Hg = s4qn(U,V,gamma,Q,lambda,g) 


    % (U,V) : m vector pairs
    % U : vectors of w_(k) - w_(k-1)
    % V : vectors of g(w_(k)) - g(w_(k-1))
    % gamma : positive constant
    % d : dimension
    % m : memory
    % Lamb : gamma * Identity 
    % Q = Nystrom Approximation
    
    [d,m] = size(U);
    
    r = size(Q,2);
    
    Lamb_init = gamma*eye(d);
    
    % Construc C
    
    C = [Lamb_init*U,V];
    
    C_tilda = [C,Q];
    
    A = U'*Lamb_init*U;  % d x 2m
    
    D = diag(sum(U.*V)); % m x m
    
    L = zeros(m,m);
    
    for j=1:m
        for i=j+1:m
            %if i > j
            L(i,j) = U(:,i)'*V(:,j);
            %end
        end
    end
    
    P = [A,L;L',-D];
    
    
    P_tilda = zeros(r+2*m);
    
    P_tilda(1:2*m,1:2*m) = inv(P);
    
    P_tilda(2*m+1:end,2*m+1:end) = - eye(r);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    Lamb_tilda = Lamb_init + lambda*eye(d);
    
    T = P_tilda - C_tilda'*inv(Lamb_tilda)*C_tilda;
    
    Struc_QN = inv(Lamb_tilda) + inv(Lamb_tilda)*C_tilda*inv(T)*C_tilda'*inv(Lamb_tilda);
    
    Hg = Struc_QN*g;

end
