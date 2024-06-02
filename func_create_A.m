function [Amc, Ael] = func_create_A(raw,  kernels)

    [Nx, Ny, Nz, Nc] = size(raw);
%     Nkx = single(kSize(1)); Nky = single(kSize(2)); Nkz = single(kSize(3));
    [Nkx, Nky, Nkz, Nk] = size(kernels);
%     Nkx = single(Nkx); Nky = single(Nky); Nkz = single(Nkz);
    


    elRawData = single(reshape((1 : 1 : prod(size(raw, 1, 2, 3))), [size(raw, 1, 2, 3)]));
    % elRawData = gpuArray(elRawData);
    if gpuDeviceCount 
        elRawData = gpuArray(elRawData);
    end


    testCK = zeros(Nx - (Nkx - 1), Ny - (Nky - 1), Nz - (Nkz - 1), Nc, Nk, 'like',raw);
    testCKel =  zeros(Nx - (Nkx - 1), Ny - (Nky - 1), Nz - (Nkz - 1),   Nk, 'like', elRawData); %originally a single

    for k = 1 : Nk
        testCK(:, :, :, :, k)   = convn( (raw(:, :, :, :) ), (kernels(:, :, :, k)), 'valid');
        testCKel(:, :, :, k) = convn( (elRawData     (:, :, :) ), (kernels(:, :, :, k)), 'valid');
    end

    %permute so that when you reshape to the A matrix, for each row, the
    %neighbors of each channel will be placed together.
    testCK   = permute(testCK  , [1, 2, 3, 5, 4]);

    Amc   = reshape(testCK  ,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)),  Nk, Nc] );
    Ael = reshape(testCKel,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)),  Nk] );

end