function weights = initializeGlorot(sz)
Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (sz(2)+ sz(1)));
weights = bound * Z;
weights = dlarray(weights);
end