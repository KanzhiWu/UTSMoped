function matrix = Quat2Mat( Quat )
    matrix = zeros(3,3);
%    if Quat(1,1) ~= 1.0;
%        scale = 1.0/Quat(1,1);
%        Quat(1,2) = scale*Quat(1,1);
%        Quat(1,3) = scale*Quat(1,3);
%        Quat(1,4) = scale*Quat(1,4);
%    end
    matrix(1,1) = 1-2*Quat(1,3)*Quat(1,3)-2*Quat(1,4)*Quat(1,4);
    matrix(1,2) = 2*Quat(1,2)*Quat(1,3)-2*Quat(1,4)*Quat(1,1);
    matrix(1,3) = 2*Quat(1,2)*Quat(1,4)+2*Quat(1,3)*Quat(1,1);
    matrix(2,1) = 2*Quat(1,1)*Quat(1,2)+2*Quat(1,4)*Quat(1,1);
    matrix(2,2) = 1-2*Quat(1,2)*Quat(1,2)-2*Quat(1,4)*Quat(1,4);
    matrix(2,3) = 2*Quat(1,3)*Quat(1,4)-2*Quat(1,2)*Quat(1,1);
    matrix(3,1) = 2*Quat(1,2)*Quat(1,4)-2*Quat(1,3)*Quat(1,1);
    matrix(3,2) = 2*Quat(1,3)*Quat(1,4)+2*Quat(1,2)*Quat(1,1);
    matrix(3,3) = 1-2*Quat(1,2)*Quat(1,2)-2*Quat(1,3)*Quat(1,3);
end