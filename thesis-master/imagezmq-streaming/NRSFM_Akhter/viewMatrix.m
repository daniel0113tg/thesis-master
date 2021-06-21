function viewMatrix(Shat, inPath)
% inPath is the path of images containing jpegs of the datasets
% This function also requires you to adjust the view of the reconstruction,
% for the detail of how to adjust the view, please see function adjustView


F = size(Shat,1)/3;
P = size(Shat,2);
x1 = min(min(Shat(1:3:end, :)));
y1 = min(min(Shat(2:3:end, :)));
z1 = min(min(Shat(3:3:end, :)));
x2 = max(max(Shat(1:3:end, :)));
y2 = max(max(Shat(2:3:end, :)));
z2 = max(max(Shat(3:3:end, :)));

if exist('inPath')
    
    for i=1: F
        clf
        im = imread(sprintf('%s\\%03d.jpeg', inPath, i));
        
        subplot('position', [0.2, 0.65, 0.6, 0.3])
        imagesc(im);
        axis image off
    
        Si = [Shat(3*i-2, :); Shat(3*i-1, :); Shat(3*i, :)];
        subplot('position', [0.05, 0.15, 0.4, 0.45]);
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.', 'MarkerSize', 15);
        drawLines(Si);
        axis equal off
        
        if exist('cva')
            campos(cpos);
            camtarget(ctar);
            camup(cup);
            camva(cva);
        else
            [cpos, ctar, cup, cva] = adjustView;
        end;
        
        subplot('position', [0.55, 0.15, 0.4, 0.45]);
        Si = [Shat(3*i-2, :); Shat(3*i-1, :); Shat(3*i, :)];
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.', 'MarkerSize', 15);
        drawLines(Si);
        axis equal off
        
        if exist('cva2')
            campos(cpos2);
            camtarget(ctar2);
            camup(cup2);
            camva(cva2);
        else
            [cpos2, ctar2, cup2, cva2] = adjustView;
        end;
        grid on
        
        pause(0.1)
    end;
else
    for i=1: F
        Si = [Shat(3*i-2, :); Shat(3*i-1, :); Shat(3*i, :)];
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.', 'MarkerSize', 15);
        drawLines(Si);
        
        axis equal off
        if exist('cva')
            campos(cpos);
            camtarget(ctar);
            camup(cup);
            camva(cva);
        else
            [cpos, ctar, cup, cva] = adjustView;
        end;
        pause(0.1)
    end;
end;

function drawLines(Si)

list1 = [1,2; 2,3; 3,6; 5,6; 6,10; 10,14; 14,18; 18,1; 3,4; 4,5; 6,7; 7,8; 9,10; 10,11;11,12;12,13;13,14;14,15;15,16;16,17;17,18;18,19;19,20;20,21];
list = [list1; list1+15];

hold on
for i=1:length(list1)
    plot3(Si(1,[list(i,1),list(i,2)]), Si(2,[list(i,1),list(i,2)]), Si(3,[list(i,1),list(i,2)]), 'r-');
end;
for i=length(list1)+1:length(list)
    plot3(Si(1,[list(i,1),list(i,2)]), Si(2,[list(i,1),list(i,2)]), Si(3,[list(i,1),list(i,2)]), 'k-');
end;
hold off
