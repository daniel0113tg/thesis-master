function [W, Rs] = generateData(S, rot, Rs)

F = size(S,1)/3;
if rot == Inf                       % Create completely random rotations
    for i=1:F
        [Ri, r] = qr(rand(3));
        
        W(2*i-1:2*i, :) = Ri(1:2,:)*S(3*i-2:3*i, :);
        Rs(2*i-1:2*i, :) = Ri(1:2,:);
    end;
elseif rot ==0 & exist('Rs')        % if rotations are given as an input
    for i=1:F
        W(2*i-1:2*i, :) = Rs(2*i-1:2*i, :)*S(3*i-2:3*i, :);
    end
else                                % if camera is to be rotated about z-axis
    for i=1:F
        Ri = [cosd(rot*i) -sind(rot*i) 0; sind(rot*i) cosd(rot*i) 0; 0 0 1];
       
        W(2*i-1:2*i, :) = Ri(2:3,:)*S(3*i-2:3*i, :);
        Rs(2*i-1:2*i, :) = Ri(2:3,:);
    end;
end;