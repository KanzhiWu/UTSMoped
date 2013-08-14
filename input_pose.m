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

 % number of points
%A = rand(n,3);
%B = R*A' + repmat(t, 1, n);
%B = B';


A = [-0.670476 -9.32572 64; 2.73429 -9.4181 63.8; 0.0615238 -7.56743 64.6; 2.88267 -7.66667 64.4; 2.19 -4.69286 65.7];
B = [-2.44275 5.86459 0.171003; 2.25009 -6.42429 -0.030773; 0.298938 -4.44794 -0.488591; 3.10154 -4.61021 -0.628942; 2.38017 -1.44464 -0.431768];

[n, m] = size(A);

[ret_R, ret_t] = pose(A, B);


A2 = (ret_R*A') + repmat(ret_t, 1, n);
A2 = A2';
% Find the error
err = A2 - B;
err = err .* err;
err = sum(err(:));
rmse = sqrt(err/n);
disp(sprintf('RMSE: %f', rmse));
disp('If RMSE is near zero, is correct!');
quat = Mat2Quat( ret_R );
euler = Quat2Euler(quat)