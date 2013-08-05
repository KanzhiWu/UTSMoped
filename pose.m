% This function finds the optimal Rigid/Euclidean transform in 3D space
% It expects as input a Nx3 matrix of 3D points.
% It returns R, t

% You can verify the correctness of the function by copying and pasting these commands:
%{

R = orth(rand(3,3)); % random rotation matrix

if det(R) < 0
    V(:,3) *= -1;
    R = V*U';
end

t = rand(3,1); % random translation

n = 10; % number of points
A = rand(n,3);
B = R*A' + repmat(t, 1, n);
B = B';

[ret_R, ret_t] = rigid_transform_3D(A, B);

A2 = (ret_R*A') + repmat(ret_t, 1, n)
A2 = A2'

% Find the error
err = A2 - B;
err = err .* err;
err = sum(err(:));
rmse = sqrt(err/n);

disp(sprintf("RMSE: %f", rmse));
disp("If RMSE is near zero, the function is correct!");

%}

% expects row data
function [R,t] = rigid_transform_3D(A, B)
    centroid_A = mean(A);
    centroid_B = mean(B);
    N = size(A,1);
    H = (A - repmat(centroid_A, N, 1))' * (B - repmat(centroid_B, N, 1));

    [U,S,V] = svd(H);

    R = V*U';

    if det(R) < 0
        V(:,3) = -V(:,3);
        R = V*U';
    end

    t = -R*centroid_A' + centroid_B';
    tr = R(1,1) + R(2,2) + R(3,3);
    rot = zeros(1,4);
    if tr > 0
        S = sqrt(tr+1.0)/2;
        rot(1,1) = (R(3,2)-R(2,3))/S;
        rot(1,2) = (R(1,3)-R(3,1))/S;
        rot(1,3) = (R(2,1)-R(1,2))/S;
        rot(1,4) = 0.25*S;
    elseif (R(1,1) > R(2,2)) & (R(1,1) > R(3,3))
        S = sqrt(1.0+R(1,1)-R(1,1)-R(2,2))*2;
        rot(1,1) = 0.25*S;
        rot(1,2) = (R(1,2)+R(2,1))/S;
        rot(1,3) = (R(1,3)+R(3,1))/S;
        rot(1,4) = (R(3,2)-R(2,3))/S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1.0+R(2,2)-R(1,1)-R(3,3))*2;
        rot(1,1) = (R(1,2)+R(2,1))/S;
        rot(1,2) = 0.25*S;
        rot(1,3) = (R(2,3)+R(3,2))/S;
        rot(1,4) = (R(1,3)-R(3,1))/S;
    else
        S = sqrt(1.0+R(3,3)-R(1,1)-R(2,2))*2;
        rot(1,1) = (R(1,3)+R(3,1))/S;
        rot(1,2) = (R(2,3)+R(3,2))/S;
        rot(1,3) = 0.25*S;
        rot(1,4) = (R(2,1)-R(1,2))/S;
    end
    w = rot(1,1); x = rot(1,2);
    y = rot(1,3); z = rot(1,4);
    phi = atan2(2*(w*x+y*z), 1-2*(x*x+y*y));
    theta = asin(2*(w*y-z*x));
    psi = atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
    ret = zeros(1,6);
    ret(1,1) = phi; ret(1,2) = theta; ret(1,3) = psi;
    ret(1,4) = t(1,1);
    ret(1,5) = t(2,1);
    ret(1,6) = t(3,1);
    ret
end