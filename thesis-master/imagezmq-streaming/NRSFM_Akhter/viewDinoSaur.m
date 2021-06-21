function viewDinoSaur(Shat, tri, inPath)
% inPath is the path of images containing bmps of the datasets
% This function also requires you to adjust the view of the reconstruction,
% for the detail of how to adjust the view, please see function adjustView

F = size(Shat,1)/3;
x1 = min(min(Shat(1:3:end, :)));
y1 = min(min(Shat(2:3:end, :)));
z1 = min(min(Shat(3:3:end, :)));
x2 = max(max(Shat(1:3:end, :)));
y2 = max(max(Shat(2:3:end, :)));
z2 = max(max(Shat(3:3:end, :)));

if exist('inPath')
    
    for i=1: F
        im = imread(sprintf('%s\\%03d.bmp', inPath, i));
        
        subplot('position', [0.2, 0.65, 0.6, 0.3])
        imagesc(im);
        axis image off

        Si = [-Shat(3*i-2, :); Shat(3*i-1, :); -Shat(3*i, :)];
        subplot('position', [0.05, 0.15, 0.4, 0.45]);
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.');
        hold on
        h = trimesh(tri, Si(1,:), Si(2,:), Si(3,:));
        hold off;
%         shading interp
%         light
%         lighting gouraud
%         colormap white
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
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.');
        hold on
        h = trimesh(tri, Si(1,:), Si(2,:), Si(3,:));
        hold off;
%         shading interp
%         light
%         lighting gouraud
%         colormap copper
        axis equal off
        if exist('cva2')
            campos(cpos2);
            camtarget(ctar2);
            camup(cup2);
            camva(cva2);
        else
            [cpos2, ctar2, cup2, cva2] = adjustView;
        end;
        
        pause(0.1)
    end;
else
    for i=1: F
        Si = [-Shat(3*i-2, :); Shat(3*i-1, :); -Shat(3*i, :)];
        plot3(Si(1,:), Si(2,:), Si(3,:), 'b.');
        hold on
        h = trimesh(tri, Si(1,:), Si(2,:), Si(3,:));
        hold off;
%         shading interp
%         light
%         lighting gouraud
%         colormap white
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