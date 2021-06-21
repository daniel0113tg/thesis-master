function Reconstruction3D = run_trajectory_space_factorisation(input_file, name)
	% load the input
	disp(input_file)
	disp(pwd)
	inp = load(input_file);
	W = inp.W;
	K = 1;
	P = size(W,2);
	F = size(W,1) / 2;
	fig1 = [];

	
	% plot the 2D image features as spatio-temporal trajectories:
	%fig1 = figure;
	%for p=1:P
		%plot3(1:F, W(1:2:end,p), W(2:2:end,p)); hold on;
	%end;
	%title('spatio-temporal image trajectories');
	%xlabel('time');
	%ylabel('image x');
	%zlabel('image y');

	%saveas(fig1, name)


	addpath('NRSFM_Akhter')
	[Shat, Rsh] = NRSFM(W, K)
	Reconstruction3D = Shat
	
	
	
