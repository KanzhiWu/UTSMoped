% expects row data
function [nobs, nmdl] = ScottLH
    %random generate a translation vector
    sigma = 0.5;
    trans = rand(1,3)*10;
    mdl = rand(10, 3);
    [row, col] = size(mdl);
    aveMdl = zeros(1, col);
    aveMdl = sum(mdl, 1)/row;
    for i = 1:row
        for j = 1:col
            mdl(i, j) = mdl(i, j)-aveMdl(1, j);
        end
    end
    
    obs = zeros(row, col);
    for i = 1:row
        for j = 1:col
            obs(i, j) = mdl(i, j)+trans(1, j)+rand/5;
        end
    end
    aveObs = zeros(1, 3);
    aveObs = sum(obs, 1)/row;
    for i = 1:row
        for j = 1:col
            obs(i, j) = obs(i, j)-aveObs(1, j);
        end
    end
    G = zeros(row, row);
    for i = 1:row
        for j = 1:row
            diff = zeros(1, col);
            for k = 1:col
                diff(1, k) = obs(i, k)-mdl(j, k);
            end
            rdiff = norm(diff);
            G(i, j) = exp(-rdiff*rdiff/(2*sigma*sigma));
        end
    end
    [U,S,V] = svd(G);
    D = eye(10);
    P = U*D*V'
    G
end