function [R] = recoverR(Rs)
% Reconstructs R matrix (block-diagonal) from Rs matrix

for i = 1:size(Rs,1)/2
    R(2*(i-1)+1:2*(i-1)+2,3*(i-1)+1:3*(i-1)+3) = Rs(2*(i-1)+1:2*(i-1)+2,:);
end