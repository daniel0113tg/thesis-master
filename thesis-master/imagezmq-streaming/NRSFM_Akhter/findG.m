function G = findG(Rh)


for i=1:size(Rh,1)/2
    r1=Rh(2*i-1,1); r2=Rh(2*i-1,2);r3=Rh(2*i-1,3);
    r4=Rh(2*i,1); r5=Rh(2*i,2);r6=Rh(2*i,3);

    A(3*i-2:3*i, :)=[r1*r1 r1*r2+r2*r1 r1*r3+r3*r1 r2*r2 r2*r3+r3*r2 r3*r3;
                     r1*r4 r1*r5+r2*r4 r1*r6+r3*r4 r2*r5 r2*r6+r3*r5 r3*r6;
                     r4*r4 r4*r5+r5*r4 r4*r6+r6*r4 r5*r5 r5*r6+r6*r5 r6*r6];
    B(3*i-2:3*i, 1) = [1;0;1];    
    
%     A(3*i-2:3*i, :) = [r1*r1 r1*r2 r1*r3 r2*r1 r2*r2 r2*r3 r3*r1 r3*r2 r3*r3;
%                        r1*r4 r1*r5 r1*r6 r2*r4 r2*r5 r2*r6 r3*r4 r3*r5 r3*r6;
%                        r4*r4 r4*r5 r4*r6 r5*r4 r5*r5 r5*r6 r6*r4 r6*r5 r6*r6];
%     B(3*i-2:3*i, 1) = [1;0;1];
 
end

g=A\B;
G = [g(1:3)'; g(2),g(4:5)'; g(3),g(5),g(6)];

% g=inv(A'*A)*A'*B;
% G = reshape(g, 3,3);
% G = (G + G')/2;