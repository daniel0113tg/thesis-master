import numpy as np
import math 
import statistics
from sklearn.manifold import TSNE


def NRSFM(W, K)
    '''
    % Takes measurement matrix W and K as input and gives the recovered
    % Structure Shat as output. The dimension of matrix W is 2F-by-P, where 
    % F is the total number of frames and P is total number of points. The 
    % image observation for each image is given in the pair of rows of 
    % matrix W. 
    %
    % K is number of DCT basis
    %
    % Output parameters Shat and Rsh are recovered structure and rotations,
    % while errS and errR are structure estimation and rotations estimation
    % errors
    '''
    F = size(W[0])/2
    P = size(W,2)

    if (2*F)>P:
        U,D,V = np.linalg.svd(W, full_matrices=False)
    else
        V,D,U = np.linalg.svd(np.transpose(W), full_matrices=False)

'''
    if K==1:
        Rsh = findRigRotations(W)
'''
    else
        LambdaHat = U[:,0:2*K]*math.sqrt(D[0:2*K,0:2*K])
        Ahat = math.sqrt(D[0:2*K,0:2*K])*V[:,0:2*K]
        global Q_START
        if isempty(Q_START):
            Q0 = Q_START;    
            Q, Q0 = metricUpgrade(LambdaHat, Q0)
        else
            Q, Q0 = metricUpgrade(LambdaHat)
        Rsh = LambdaHat*Q

    Rp = recoverR(Rsh)
    Theta_k = generateDCT(F,K)
    G = Rp*Theta_k

    print('-- Stability of reconstruction: --')
    print('Lambda =' Rp * np.transpose(Theta_K))
    conditionNumber = cond(np.transpose(G) * G)
    print('Condition number of  transpose(Lambda) * Lambda = ', str(conditionNumber), ' for window size = ', str(size(W,1)/2), ' and K = ', str(K)))
    print(' ')

    AA = numpy.linalg.inv(np.transpose(G)*G)*np.transpose(G)*W
    Shat = Theta_k*AA
    return Shat, Rsh

'''
def findRigRotations(W):
    F = size(W,1)/2
    P = size(W,2)
    W = W - statistics.mean(W,2)*np.ones(1,P)

    if (2*F)<P
        V,D,U=svdnp.linalg.svd(np.transpose(W), full_matrices=False)
    else
        U,D,V=svdnp.linalg.svd((W, full_matrices=False)

    dU=U[:,0:2]
    dD=D[0:2,0:2]
    dV=V[:,0:2)]

    Rhat=dU*math.sqrt(dD)
    Shat=math.sqrt(dD)*np.transpose(dV)

    G = findG(Rhat)
    U1 ,D1, V1 = svdnp.linalg.svd(G, full_matrices=False)
    Q = np.transpose(U1*math.sqrt(D1))
    RecR=Rhat*np.transpose(Q)
    RecS=numpy.linalg.inv(np.transpose(Q))*Shat
    return RecR, RecS

def findG(Rhat):
    for i in len(Rh[1])/2
        r1=Rh(2*i-1[1]); r2=Rh(2*i-1[2]);r3=Rh(2*i-1[3]);
        r4=Rh(2*i[1]); r5=Rh(2*i[2]);r6=Rh(2*i[3]);

        A[3*i-2:3*i, :]=[r1*r1 r1*r2+r2*r1 r1*r3+r3*r1 r2*r2 r2*r3+r3*r2 r3*r3,
                        r1*r4 r1*r5+r2*r4 r1*r6+r3*r4 r2*r5 r2*r6+r3*r5 r3*r6,
                        r4*r4 r4*r5+r5*r4 r4*r6+r6*r4 r5*r5 r5*r6+r6*r5 r6*r6]
        B[3*i-2:3*i, 1] = [1;0;1]

    g=A \ B
    G = [np.transpose(g[0:2]); g[1],np.transpose(g[3:4]); g[2],g[4],g[5]];

    % g=numpy.linalg.inv(np.transpose(A)*A)*np.transpose(A)*B;
    % G = np.reshape(g,3,3);
    % G = (G + np.transpose(G))/2;
'''

def metricUpgrade(LambdaHat,Q0):
    k = size(LambdaHat[1])/3
    F = size(LambdaHat[0])/2

    if(~exist('q0'))
        q0 = zeros(2*k,2)
        q0[0*k+1][0] = 1
        q0[1*k+1][1] = 1
        q0[2*k+1][2] = 1
    end

    global Q_FINAL
    if isempty(Q_FINAL):
        q = Q_FINAL
        printf('**USING Q_FINAL evaluating at f(q) = %f\n', evalQ(q,LambdaHat))
    else
        options = optimset('Diagnostics','on','Display','final','MaxFunEval',1000000,'MaxIter',20000,'TolFun',1e-10,'TolX',1e-10);%,'PlotFcn',@optimplotfirstorderopt);
        [q, fval,exitflag,output] = fminunc(@evalQ,q0,options,LambdaHat); 
    Q = np.reshape(q,3*k,3)
    Q0 = q0
    return Q, Q0

def evalQ():
    function [f] = evalQ(q,LambdaH)

    Rbar = LambdaH*q;        % Rbar(2F*3), rotational matrices

    Rbar1 = Rbar(1:2:end, :);
    Rbar2 = Rbar(2:2:end, :);

    onesArr = [ sum(Rbar1.^2, 2); sum(Rbar2.^2, 2)];
    zerosArr = sum(Rbar1.*Rbar2, 2);

    f = sum( (onesArr-1).^2 ) + sum(zerosArr.^2);

def recoverR(Rsh):
    function [R] = recoverR(Rs)
% Reconstructs R matrix (block-diagonal) from Rs matrix

for i = 1:size(Rs,1)/2
    R(2*(i-1)+1:2*(i-1)+2,3*(i-1)+1:3*(i-1)+3) = Rs(2*(i-1)+1:2*(i-1)+2,:);
end

def generateDCT(F,K):
    function out = generateDCT(f, k)

%generate dct matrix of required size
d = idct(eye(f));

%truncate values not required
d = d(:, 1:k);

%rearrange into theta matrix
out = [];

for i = 1:f
    out = [out; [d(i,:) zeros(1,2*k); zeros(1,k) d(i,:) zeros(1,k); zeros(1,2*k) d(i,:)]];
end;


def main():
    K = 12
    rotStruct = 0
    Shat, Rsh = NRSFM(W, K)

main()