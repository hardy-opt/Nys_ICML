function [] = final_plot()
%     close all;
    % Markers = {'o','+','*','.','x','_','|','square','diamond','^','v','>','<','pentagram' }
    % To change X and Y axis properties GOTO LINE 44
    %Relativer error LINE 219
    RUNS = 3;
    EPOCHS = 50;
    lambdas = [1e-1 1e-2 1e-3 1e-4];
    etas = [ 1e0 1e-1 1e-2 1e-3 1e-4 1e-5 ];
    rhos = [ 0 1e3 1e2 1e1 1e0 1e-1 1e-2 1e-3 1e-4];
    path = 'result/';
   
    datasets = {
        'REALSIM'   %1 
        'CIFAR10B'  %2
        'MNISTB'    %3
        'EPSILON'   %4
        'ADULT'     %5
        'W8A'       %6
        'ALLAML'    %7
        };

    lw = 2;%RUNS;
    ms = 8;
    params = load_settings(lw, ms, lambdas, etas, rhos, RUNS, EPOCHS);
    sparams = {
       % params('NEWTON')
        params('SVRG-LBFGS')
        params('OBFGS')
        params('SVRG-SQN')
        params('SQN')
        params('Structured_QN')
        params('SGD')
        params('SVRG')
        params('adam')
        params('Nystrom_svrg')
        params('Nystrom_sgd_dpp')
        params('Nystrom_sgd')
        params('Nystrom_svrg_dpp')

%         params('K30_Nystrom_svrg_LM_dpp')
%         params('K30_Nystrom_svrg_LM')
%         params('K30_Nystrom_sgd_LM_dpp')
%         params('K30_Nystrom_sgd_LM')
%          params('K30_Nystrom_GD')
%          params('K30_Nystrom_GD_dpp')
%          params('K30_Nystrom_GDIM')
%          params('K30_Nystrom_GDIM_dpp')

        };
    
    yparams = {'cost', 'val_cost', 'acc_tr', 'acc_val'};
    xparams = {'time', 'epoch'};
    plot_params.sort = yparams{1};
    plot_params.y = yparams{1};
    plot_params.x = xparams{2};
    for dsi=5
            disp(datasets{dsi})
    for l = 1:length(lambdas)
%          sparams{9}.etas=etas(l);
%          sparams{2}.etas=etas(4);
%         sparams{9}.rhos=rhos(l);
        subplot(1,4,l);
        plot_method_lambda(strcat(path, datasets{dsi}, '/'), sparams, lambdas(l), plot_params, eps+05e-4);
%         xlim([0,200]);
%         saveas(gcf,strcat(lower(datasets{dsi}),'_ctc_lam',num2str(l+1),'.eps'),'epsc');
%           legendmarkeradjust(20,20);
        fprintf('\n');
    end
    end
    %legend({'SVRG-LBFGS','SVRG-SQN', 'ADAM', 'SQN', 'OBFGS', 'NSVRG-D', 'NSVRG', 'NSGD-D', 'NSGD'})

end

function plot_method_lambda(dataset, sparams, lambda, plot_params, ref)
%     if length(sparams)>1
%         figure;
%     end
    hold on;
    set(gca, 'FontSize', 16);
    title(strcat('\lambda=',sprintf('10^{%0.0f}', log10(lambda))));
    switch plot_params.x
        case 'time'
            xlabel('Time (seconds)');
        case 'epoch'
            xlabel('Epochs');
    end
    switch plot_params.y
        case 'cost'
            ylabel('Opt. Error (log scale)');
        case 'val_cost'
            ylabel('Test Error (log scale)');
        case 'acc_tr'
            ylabel('Train Accuracy');
        case 'acc_val'
            ylabel('Test Accuracy');
    end
    if ref>=0
        opt_cost=inf;
        for m = 1:length(sparams)
            hold on;
            [bestmu, bestsg, besteta, bestrho] = best_method_lambda(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, plot_params.sort);
            if bestmu.Count > 1
                y = bestmu(plot_params.y);
                if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
                    opt_cost = min(opt_cost,min(y));
                end
            end
        end
    end
    maxy = -inf;
    miny = inf;
    for m = 1:length(sparams)
        hold on;
        [bestmu, bestsg, besteta, bestrho] = best_method_lambda(dataset, sparams{m}.name, lambda, sparams{m}.etas, sparams{m}.rhos, sparams{m}.EPOCHS, sparams{m}.RUNS, plot_params.sort);
        if bestmu.Count > 1
            if strcmp(plot_params.y, 'val_cost') || strcmp(plot_params.y,'acc_val')
                x=bestmu(plot_params.x);
                x=x(2:end);
            else
                x=bestmu(plot_params.x);
            end
            if ref>=0 && (strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost'))
                y = (bestmu(plot_params.y)-opt_cost+ref)/(1+opt_cost);
                s = bestsg(plot_params.y)/(1+opt_cost);
            else
                y = bestmu(plot_params.y);
                s = bestsg(plot_params.y);
            end
            if bestrho == -1
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f})',log10(besteta)));
            elseif bestrho == 0
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=\mid\midZ\mid\mid_{F})');
            else
                displayname=strcat(strrep(sparams{m}.name, '_', '-'), ' (\eta=', sprintf('10^{%0.0f},',log10(besteta)),' \rho=', sprintf('10^{%0.0f})', log10(bestrho)));
            end
            if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
                [~, idx] = min(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,y(1));
                miny = min(miny,min(y));
            else
                [~, idx] = max(y);
                if bestrho == -1
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  %-12s  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, ' ', plot_params.y, y(idx), idx, round(x(idx)));
                else
                    fprintf('%-23s: lambda: %.1e  eta: %.1e  rho: %.1e  %s: %0.6f  @(idx: %d, val: %d)\n', sparams{m}.name, lambda, besteta, bestrho, plot_params.y, y(idx), idx, round(x(idx)));
                end
                maxy = max(maxy,max(y));
                miny = min(miny,min(y));
            end
            idx=length(y);
            if length(sparams)==1
                displayname = strcat(displayname, '@\lambda=', sprintf('10^{%0.0f})', log10(lambda)));
%                 errorbar(x(1:idx), y(1:idx), s(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                plot(x(1:idx), y(1:idx), 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
            else
%                 errorbar(x(1:idx), y(1:idx), s(1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname);
                plot(x(1:1:idx), y(1:1:idx), 'linestyle', sparams{m}.line, 'color', sparams{m}.linecolor, 'Marker', sparams{m}.marker, 'markersize', sparams{m}.markersize, 'linewidth', sparams{m}.linewidth, 'MarkerFaceColor', sparams{m}.facecolor, 'displayname', displayname, 'MarkerIndices', 1:1:idx);
            end
        end
    end
    if maxy > miny 
        ylim([miny, maxy]);
    end
%     legend('FontSize',24);
    if strcmp(plot_params.y, 'cost') || strcmp(plot_params.y, 'val_cost')
        set(gca, 'YScale', 'log');
    end
end

function [bestmu, bestsg, besteta, bestrho] = best_method_lambda(dataset, method, lambda, etas, rhos, EPOCHS, RUNS, sort_param)
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
            [mu, sg] = mean_RUNS(dataset, method, lambda, eta, -1, EPOCHS, RUNS);
            if mu.Count > 0
                if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost')
                    a=mu(sort_param);
                    if (a(EPOCHS)) < best
                        best = min(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestrho = -1;
                    end
                else
                    if max(mu(sort_param)) > best
                        best = max(mu(sort_param));
                        bestmu = mu;
                        bestsg = sg;
                        besteta = eta;
                        bestrho = -1;
                    end
                end
            end
        else
            for rho = rhos
                [mu, sg] = mean_RUNS(dataset, method, lambda, eta, rho, EPOCHS, RUNS);
                if mu.Count > 0
                    if strcmp(sort_param, 'cost') || strcmp(sort_param, 'val_cost')
                        if min(mu(sort_param)) < best
                            best = min(mu(sort_param));
                            bestmu = mu;
                            bestsg = sg;
                            besteta = eta;
                            bestrho = rho;
                        end
                    else
                        if max(mu(sort_param)) > best
                            best = max(mu(sort_param));
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

function [mu, sg] = mean_RUNS(dataset, method, lambda, eta, rho, EPOCHS, run)
    i=0;
    ratio=0.1;
    K = containers.Map();
    K('REALSIM')=ceil(20959*ratio);
    K('CIFAR10B')=ceil(3073*ratio);
    K('MNISTB')=ceil(785*ratio);
    K('EPSILON')=ceil(2001*ratio);
    K('W8A')=ceil(301*ratio);
    K('ADULT')=ceil(124*ratio);
    K('ALLAML')=ceil(7130*ratio);
    
    for j = 1:run
        if rho > -1
            Name = strcat(dataset, sprintf('K%d_B64_%s_%.1e_R_%.1e_rho_%.1e_run_%d.mat', K(dataset), method, eta, lambda, rho, j));
        else
            Name = strcat(dataset, sprintf('B64_%s_%.1e_R_%.1e_run_%d.mat', method, eta, lambda, j));
        end
%         disp(Name);
        if exist(Name, 'file')
            d = load(Name);
            if strcmp(method,'NEWTON')
                EPOCHS=50;
            end
            endi=min(length(d.info_s1.iter),EPOCHS+1);
            if ~isfield(d.info_s1,'val_cost') || length(d.info_s1.iter) < EPOCHS+1
                continue;
            else
%                 disp(Name);
            end
            
            i = i + 1;
            cost(i,:) = d.info_s1.cost(1:endi);
            vcost(i,:) = d.info_s1.val_cost(2:endi);
            acc(i,:) = d.info_s1.acc_tr(1:endi);
            vacc(i,:) = d.info_s1.acc_val(2:endi);
            time(i,:) = d.info_s1.time(1:endi);
        else
%             disp(Name);
            continue;
        end
    end
    mu = containers.Map();
    sg = containers.Map();
    if i>0
        isbar=1;
        if 0
            mu('cost') = round(10000*mean(cost, 1))/10000;
            sg('cost') = isbar*round(10000*std(cost, [], 1))/10000;
            mu('val_cost') = round(10000*mean(vcost, 1))/10000;
            sg('val_cost') = isbar*round(10000*std(vcost, [], 1))/10000;
            mu('acc_tr') = round(10000*mean(acc, 1))/10000;
            sg('acc_tr') = isbar*round(10000*std(acc, [], 1))/10000;
            mu('acc_val') = round(10000*mean(vacc, 1))/10000;
            sg('acc_val') = isbar*round(10000*std(vacc, [], 1))/10000;
            mu('time') = round(10000*mean(time, 1))/10000;
        else
            mu('cost') = mean(cost, 1);
            sg('cost') = isbar*std(cost, [], 1);
            mu('val_cost') = mean(vcost, 1);
            sg('val_cost') = isbar*std(vcost, [], 1);
            mu('acc_tr') = mean(acc, 1);
            sg('acc_tr') = isbar*std(acc, [], 1);
            mu('acc_val') = mean(vacc, 1);
            sg('acc_val') = isbar*std(vacc, [], 1);
            mu('time') = mean(time, 1);
        end
        mu('epoch') = [0:EPOCHS];
    end
end

function [methods] = load_settings(lw, ms, lambdas, etas, rhos, RUNS, EPOCHS)
    methods=containers.Map;

    params.name = 'SGD';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0 0.5 1];
    params.marker = '*';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'SVRG';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0 0 1];
    params.marker = 'diamond';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'adam';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 1];
    params.linecolor = [0.5 0 1];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'SVRG-LBFGS';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[0 0 1];
    params.linecolor = [0 0 1];
    params.marker = '^';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'OBFGS';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0 .8 1];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'SVRG-SQN';
    params.line = ':';
    params.linewidth = lw;
    params.facecolor = 'none'; %[0 0 0];
    params.linecolor = [0 0.5 0];
    params.marker = '>';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'SQN';
    params.line = ':'; %c
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0 0.8 0.35];
    params.marker = 'p';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    %Proposed
    params.name = 'NSVRG-D';
    params.line = '-';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [1 0 0];
    params.marker = 'o';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'NSVRG';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [0.75 0 0];
    params.marker = 's';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'NSGD-D';
    params.line = '-';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [1 0 1];
    params.marker = 'o';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'NSGD';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [1 0.5 0.5];
    params.marker = 's';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_svrg_LM_dpp';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.linecolor = [1 0 0];
    params.marker = 'o';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_svrg_LM';
    params.line = '-';
    params.linewidth = lw;
    params.linecolor = [0.75 0 0];
    params.marker = 's';
    params.facecolor = 'none'; %[1 0 0];
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_sgd_LM_dpp';
    params.line = '--';
    params.linewidth = lw;
    params.linecolor = [1 0 1];
    params.marker = 'o';
    params.markersize = ms;
    params.facecolor = 'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_sgd_LM';
    params.line = '-';
    params.linewidth = lw;
    params.linecolor = [1 0.5 0.5];
    params.marker = 's';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'Structured_QN';
    params.line = '-';
    params.linewidth = lw;
    params.linecolor = [1 0.5 0.5];
    params.marker = 's';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'NEWTON';
    params.line = '-.';
    params.linewidth = lw;
    params.linecolor = [0.25 0.25 0.75];
    params.marker = 'p';
    params.markersize = ms;
    params.facecolor = 'none'; %'none'; %[1 0 0];
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = [];
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_GD';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0.2 0.7 1];
    params.marker = '*';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;

    params.name = 'K30_Nystrom_GD_dpp';
    params.line = '--';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0.2 0.15 0.57];
    params.marker = 'diamond';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;
    
    params.name = 'K30_Nystrom_GDIM_dpp';
    params.line = '-.';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0.2 0.57 0.157];
    params.marker = 'diamond';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;
    
    params.name = 'K30_Nystrom_GDIM';
    params.line = '-.';
    params.linewidth = lw;
    params.facecolor = 'none'; %[1 0 0];
    params.linecolor = [0.57 0.15 0.25];
    params.marker = 'diamond';
    params.markersize = ms;
    params.lambdas = lambdas;
    params.etas = etas;
    params.rhos = rhos;
    params.RUNS = RUNS;
    params.EPOCHS = EPOCHS;
    methods(params.name)=params;
end