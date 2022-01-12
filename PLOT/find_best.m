function [bestmu, bestsg, besteta, bestrho] = find_best(dataset, method, lambda, etas, rhos, EPOCHS, RUNS, COL, BS, sort_param)
    bestmu = containers.Map();
    bestsg = containers.Map();
    besteta = -1;
    bestrho = -1;
    if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost')
        best = inf;
    else
        best = -inf;
    end
    for eta = etas
        if isempty(rhos)
            [mu, sg] = mean_run(dataset, method, lambda, eta, -1, EPOCHS, RUNS, COL, BS);            
            if mu.Count > 0
                av=mu(sort_param);
                at=mu('epoch');
                if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost')
%                     auc=trapz(at,av);
                    auc=min(mu(sort_param));
                    if auc < best
                        best = auc; %min(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestrho = -1;
                    end
                else
%                     auc=trapz(at,av);
                    auc=max(mu(sort_param));
                    if auc > best
                        best = auc; %max(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestrho = -1;
                    end
                end
            end
        else
            for rho = rhos
                [mu, sg] = mean_run(dataset, method, lambda, eta, rho, EPOCHS, RUNS, COL, BS);
                if mu.Count > 0
                    av=mu(sort_param);
                    at=mu('time');
                    auc=trapz(at,av); 
                    if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost')
                        if auc < best   %%min(mu(sort_param))
                            best = auc;
                            bestmu = mu;
                            bestsg = sg;
                            besteta = eta;
                            bestrho = rho;
                        end
                    else
                        if auc > best   %max(mu(sort_param))
                            best = auc;
                            bestmu = mu;
                            bestsg = sg;
                            besteta = eta;
                            bestrho = rho;
                        end
                    end
                end
            end
        end
    end
end
