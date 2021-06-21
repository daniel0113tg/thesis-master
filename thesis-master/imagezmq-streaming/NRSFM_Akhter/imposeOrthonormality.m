function Rs = imposeOrthonormality(Rs)

F = size(Rs,1)/2;

for i=1:F
    [V,D,U] = svd(Rs(2*i-1:2*i, :)', 0);
    Rs(2*i-1:2*i, :) = U*V';
end;