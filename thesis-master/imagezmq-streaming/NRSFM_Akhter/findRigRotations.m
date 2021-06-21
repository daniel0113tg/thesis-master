function [RecR, RecS] = findRigRotations(W)

F = size(W,1)/2;
P = size(W,2);
W = W - mean(W,2)*ones(1,P);

if (2*F)<P
    [V,D,U]=svd(W',0);
else
    [U,D,V]=svd(W,0);
end;

dU=U(:,1:3);
dD=D(1:3,1:3);
dV=V(:,1:3);

Rhat=dU*sqrt(dD);
Shat=sqrt(dD)*dV';

G = findG(Rhat);
[U1,D1,V1] = svd(G);
Q = (U1*sqrt(D1))';
% [Q,p] = chol(G);  % Q'Q=G
RecR=Rhat*Q';
RecS=inv(Q')*Shat;
