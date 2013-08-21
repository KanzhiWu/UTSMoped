% expects row data
function [nobs, nmdl] = ScottLHnm
    % random generate a translation vector
    sigma =1;
    trans = rand(1,2)*5;
    % generate random model features
    % calculate the average
    mdl = rand(20, 2);
    [mdlrow, mdlcol] = size(mdl);   
    aveMdl = zeros(1, mdlcol);
    aveMdl = sum(mdl, 1)/mdlrow;
    for i = 1:mdlrow
        for j = 1:mdlcol
            mdl(i, j) = mdl(i, j)-aveMdl(1, j);
        end
    end
 
    % generate random model features
    % calculate the average    
    obs = zeros(mdlrow/2, mdlcol);
    [obsrow, obscol] = size(obs);    
    for i = 1:obsrow
        for j = 1:obscol
            obs(i, j) = mdl(i*2, j)+trans(1, j);
        end
    end
    aveObs = zeros(1, 2);
    aveObs = sum(obs, 1)/obsrow;
    for i = 1:obsrow
        for j = 1:obscol
            obs(i, j) = obs(i, j)-aveObs(1, j);
        end
    end    
    G = zeros(obsrow, mdlrow);
    for i = 1:obsrow
        for j = 1:mdlrow
            % calculate the difference between ith observation point
            % and jth model point
            diff = zeros(1, obscol);
            for k = 1:obscol
                diff(1, k) = obs(i, k)-mdl(j, k);
            end
            rdiff = norm(diff);
            % Eq.1 on the Scott and Longuet-Higgins website
            G(i, j) = exp(-rdiff*rdiff/(2*sigma*sigma));            
        end
    end
    [U,S,V] = svd(G);
    D = eye(obsrow, mdlrow);
    P = U*D*V'
end