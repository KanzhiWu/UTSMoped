% This function finds the optimal Rigid/Euclidean transform in 3D space
% It expects as input a Nx3 matrix of 3D points.
% It returns R, t

% You can verify the correctness of the function by copying and pasting these commands:


R = orth(rand(3,3)); % random rotation matrix

if det(R) < 0
    V(:,3) = -V(:,3);
    R = V*U';
end

t = rand(3,1); % random translation

n = 4; % number of points
%A = rand(n,3);
%B = R*A' + repmat(t, 1, n);
%B = B';
A = [-1.64498 9.58304 0.502101;-0.336667 -0.862987 -0.613409;2.31831  -7.85706  0.724718;-0.336667 -0.862987 -0.613409];
%B = [1 1 0; 0 1 0; 0 1 1; 1 1 1];
B = [28.4181 0.417191 -2.7;-20.4058 -0.648047 2.6; -0.448905 0.640809 -1.3; -7.56348 -0.409952 1.4];
[ret_R, ret_t] = pose(A, B);
[ret_R_inv, ret_t_inv] = pose_inv(A, B);


A2 = (ret_R*A') + repmat(ret_t, 1, n);
A2 = A2';
% Find the error
err = A2 - B;
err = err .* err;
err = sum(err(:));
rmse = sqrt(err/n);
disp(sprintf('RMSE: %f', rmse));
disp('If RMSE is near zero, is correct!');
angle = Matrix2Eular( ret_R );
disp('----------------------------------------------');
A2_inv = (ret_R_inv*A') + repmat(ret_t_inv, 1, n);
A2_inv = A2_inv';
err_inv = A2_inv - B;
err_inv = err_inv .* err_inv;
err_inv = sum(err(:));
rmse_inv = sqrt(err_inv/n);
disp(sprintf('RMSE_inv: %f', rmse_inv));
disp('If RMSE is near zero, is correct!');
angle_inv = Matrix2Eular( ret_R_inv );


