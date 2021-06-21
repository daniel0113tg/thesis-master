function viewMocap(S, Shat, list, a, b)

F = size(S,1)/3;
sm = mean(S,2);
S = S - sm*ones(1,size(S,2));

Y = findRotation(S, Shat);         % Procrust Alignment
for i=1: F
    Shat(3*i-2:3*i,:) = Y*Shat(3*i-2:3*i,:);
end;

for i=1:F
    plot3(S(3*i-2, :), S(3*i-1, :), S(3*i, :), 'b.');
    hold on
    plot3(Shat(3*i-2, :), Shat(3*i-1, :), Shat(3*i, :), 'rO');
    hold off
    
    drawLines3(S(3*i-2:3*i, :), list);
    drawLines3(Shat(3*i-2:3*i, :), list);  
    axis equal off vis3d
    if exist('b')
        view(a, b);
    end;
    pause(0.1)
end;

function drawLines3(Si, list)
hold on
for i=1:length(list)
    h = plot3(Si(1,[list(i,1),list(i,2)]), Si(2,[list(i,1),list(i,2)]), Si(3,[list(i,1),list(i,2)]), 'k-');
end;
hold off