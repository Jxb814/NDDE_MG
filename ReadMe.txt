% for the multiple delay learning in scalar systems, internal MATLAB function needs to be modified
% MATLAB/R2021b/toolbox/nnet/deep/internal_interp1Backward.m
% add the following 
%%%%%%%%%%%%%%%%%%%% Add %%%%%%%%%%%%%%%%%
dv = full(dv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in function iValueLinear
% between "dv = reshape(dv, szV);" and "dv = cast(dv, 'like', dvq);"


function dv = iValueLinear(x, xq, dvq, mapToQueryGrid)

szV = size(dvq);
szV(1) = length(x);

numInterp = length(x);
numQuery = length(xq);

iivec = find(mapToQueryGrid(:));
jjvec = double(mapToQueryGrid(iivec));
iivec = iivec(:);
jjvec = jjvec(:);


xdiff = x(jjvec+1) - x(jjvec);
v1 = (xq(iivec) - x(jjvec)) ./ xdiff;
v2 = (x(jjvec+1) - xq(iivec)) ./ xdiff;

AM1 = sparse(jjvec+1, iivec, double(v1), numInterp, numQuery);
AM2 = sparse(jjvec, iivec, double(v2), numInterp, numQuery);
AM = AM1 + AM2;

dvq = reshape(dvq, size(dvq, 1), []);
dv = AM*double(dvq);
%%%%%%%%%%%%%%%%%%%% Add %%%%%%%%%%%%%%%%%
dv = full(dv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dv = cast(dv, 'like', dvq);

%{
% Direct indexing variant:
dv = zeros(szV, 'like', dvq);
for ii=1:numQuery
    jj = indxqInx(ii);
    if jj == 0, continue; end
    commonFactor = dvq(ii, :) / (x(jj+1) - x(jj));
    dv(jj+1, :) = dv(jj+1, :) + commonFactor .* (xq(ii) - x(jj));
    dv(jj, :) = dv(jj, :) + commonFactor .* (x(jj+1) - xq(ii));
end
%}

end
