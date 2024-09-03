function dy = ctrDiff(yint,tint)
dt = tint(2)-tint(1);
dy = zeros(size(yint));
for k = 2:length(yint)-1
    dy(:,k) = (yint(:,k+1)-yint(:,k-1))/2/dt;
end
dy(:,1)=(yint(:,2)-yint(:,1))/dt;
dy(:,end)=(yint(:,end)-yint(:,end-1))/dt;
end