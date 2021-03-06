% expects row data
function [R,t] = pose_inv(A, B)
    centroid_A = mean(A);
    centroid_B = mean(B);
    N = size(A,1);
    H = (A - repmat(centroid_A, N, 1))' * (B - repmat(centroid_B, N, 1));

    [U,S,V] = svd(H);
    %disp('Coplanar points!');
    V(:,3) = -V(:,3);
    R = V*U'
    t = -R*centroid_A' + centroid_B';
end