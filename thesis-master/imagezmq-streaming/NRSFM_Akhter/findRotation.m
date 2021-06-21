function [Y] = findRotation(S, Sh)

F = size(S, 1)/3;
S1 = [];
S2 = [];
for i=1: F
    S1 = [S1, S(3*i-2:3*i, :)];
    S2 = [S2, Sh(3*i-2:3*i, :)];
end;
Y = S1/S2;
% Y = inv(S2*S2')*S1*S2';
[U, D, V] = svd(Y);
Y = U*V';