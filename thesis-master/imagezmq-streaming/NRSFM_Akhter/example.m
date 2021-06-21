%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K Values for different datasets
% matrix          3
% PIE face        2
% cubes           2
% Dinosaur        12
% 
% MultiRigid      4
% Yoga            11
% Stretch         12
% drink           13
% pickup          12
% shark           2
% dance           5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Real Data example
load dinosaur      % it will load W matrix
K = 12;
rotStruct = 0;
[Shat, Rsh] = NRSFM(W, K);

% viewCubes(Shat, tri);
viewDinoSaur(Shat, tri);
% viewMatrix(Shat)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Synthetic data example 1
% load datasets\pickup           % It will load S & W matrix
% K = 12;
% theta = 5;          % 5 degree rotation per frame
% rotStruct = 0;
% [W, Rs] = generateData(S, theta);        % create W matrix
% 
% [Shat, Rsh] = NRSFM(W, K);
% errS = compareStructs(S, Shat);
% disp('Struct Error')
% mean(errS)
% 
% [errR] = compareRotations(Rs, Rsh);
% disp('Rotation Error')
% mean(errR)
% viewMocap(S, Shat, list);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Synthetic data example 2
% load datasets\shark           % It will load S and W matrix
%                       % W matrix is the orthographic projection of S with
%                       % static camera
% [W] = generateData(S, 0);                      
% rotStruct = 1;
% K = 2;
% [Shat] = NRSFM(W, K, rotStruct);
% errS = compareStructs(S, Shat);
% disp('Struct Error')
% mean(errS)
% viewStruct(S, Shat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%