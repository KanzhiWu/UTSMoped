function euler = Quat2Euler(quat)
    euler = zeros(1,3);
    euler(1,1) = atan2( 2*quat(1,3)*quat(1,1)-2*quat(1,2)*quat(1,4), 1-2*quat(1,3)*quat(1,3)-2*quat(1,4)*quat(1,4) );
    euler(1,2) = asin( 2*quat(1,2)*quat(1,3)+2*quat(1,4)*quat(1,1) );
    euler(1,3) = atan2( 2*quat(1,2)*quat(1,1)-2*quat(1,3)*quat(1,4), 1-2*quat(1,2)*quat(1,2)-2*quat(1,4)*quat(1,4) );
    euler(1,:) = euler(1,:)*180/pi;
end