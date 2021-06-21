function [Shat, Rsh, errS, errR] = NRSFM(W, K, rotStruct, S, Rs)
% Takes measurement matrix W and K as input and gives the recovered
% Structure Shat as output. The dimension of matrix W is 2F-by-P, where 
% F is the total number of frames and P is total number of points. The 
% image observation for each image is given in the pair of rows of 
% matrix W. Similar format is used for matrix Shat.
%
% K is number of DCT basis, rotStruct is optional boolean variable 
% telling (if 1) whether you want to rotate your recovered structure with 
% estimated rotations or not. S (actual structure) and Rs(actual rotations) 
% are optional parameters. They can be used for measuring structure 
% estimation and rotation estimation error
%
% Output parameters Shat and Rsh are recoverd structure and rotations,
% while errS and errR are structure estimation and rotations estimation
% errors

F = size(W,1)/2;
P = size(W,2);

if ~exist('rotStruct')
    rotStruct = 0;
end;
if exist('S')       % Normalize the S and W matrices
    s = mean(std(S, 1, 2));
    S = S/s;
    sm = mean(S,2);
    S = S - sm*ones(1,size(S,2));
    W = W/s;
end;

wm = mean(W,2);
disp(wm);
W = double(W) - wm*ones(1,size(W,2));

if (2*F)>P
    [U,D,V] = svd(W,0);
else
    [V,D,U] = svd(W',0);
end;

if K==1
    Rsh = findRigRotations(W);
else
    LambdaHat = U(:,1:3*K)*sqrt(D(1:3*K,1:3*K));
    Ahat = sqrt(D(1:3*K,1:3*K))*V(:,1:3*K)';

%     Q0 = inv(LambdaHat'*LambdaHat)*LambdaHat'*Rs1;          % initial guess for triplet using rigid rotations
    [Q0] = metricUpgrade(LambdaHat);
    Rsh = LambdaHat*Q0;
end;

% Rs1 = imposeOrthonormality(Rs1);
Rp = recoverR(Rsh);
Theta_k = generateDCT(F,K);
G = Rp*Theta_k;
AA = inv(G'*G)*G'*W;
Shat = Theta_k*AA;

if rotStruct==1
    [Shat] = rotateStruct(Shat, Rsh);
end;

if exist('S')
    Y2 = findRotation(S, Shat);
    for i=1:F
        Shat(3*i-2:3*i,:) = Y2*Shat(3*i-2:3*i,:);
        errS(i)  = sum(sqrt( sum( (S(3*i-2:3*i, :)-Shat(3*i-2:3*i, :)).^2) ) )/P;
    end
    disp('Struct Error')
    mean(errS)
    Shat = s*Shat;
end;
if exist('Rs')
    [errR] = compareRotations(Rs, Rsh);
    disp('Rotation Error')
    mean(errR)
end
