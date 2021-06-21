function [errR, x, Rs1] = compareRotations(Rs, Rs1)

Rs1 = procrust(Rs, Rs1);
F = size(Rs, 1)/2;
for i=1:F
   errR(i) = norm(Rs1(2*i-1:2*i, :) - Rs(2*i-1:2*i, :));
end
