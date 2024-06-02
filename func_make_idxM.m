function idxM = func_make_idxM(rawData, kSize)
    % rawData is Nx, Ny, Nz, Nc
    % kSize is Nkx, Nky, Nkz
    rawData = gpuArray(rawData);
    [Nx, Ny, Nz, Nc] = size(rawData);
    Nkx = kSize(1); Nky = kSize(2); Nkz = kSize(3);
    
    elRawData = single(reshape((1 : 1 : Nx * Ny * Nz), [Nx, Ny, Nz]));
    elRawData = gpuArray(elRawData);
    
    Nk = prod(kSize);
    kernels = eye(Nk, Nk);
    kernels = flip(kernels, 2);
    kernels = reshape(kernels, [Nkx, Nky, Nkz, Nk]);
    kernels = permute(kernels, [2, 1, 3, 4]);

    testCK = zeros(Nx - (Nkx - 1), Ny - (Nky - 1), Nz - (Nkz - 1), Nc, Nk, 'like',rawData);
    testCKel =  zeros(Nx - (Nkx - 1), Ny - (Nky - 1), Nz - (Nkz - 1),   Nk, 'like', elRawData); %originally a single
    % testCKel = gpuArray(testCKel);%make it gpuArray earlier
    for k = 1 : Nk
        testCK(:, :, :, :, k)   = convn( (rawData(:, :, :, :) ), (kernels(:, :, :, k)), 'valid');
        testCKel(:, :, :, k) = convn( (elRawData     (:, :, :) ), (kernels(:, :, :, k)), 'valid');
    end
    
    %permute so that when you reshape to the A matrix, for each row, the
    %neighbors of each channel will be placed together.
    testCK   = permute(testCK  , [1, 2, 3, 5, 4]);

    Amc   = reshape(testCK  ,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)),  Nk, Nc] );
    Ael = reshape(testCKel,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)),  Nk] );

    % make the idxM matrix
    uqv = unique(Ael(:)); %divisors(numel(uqv))
    %BREAK HERE and run divisors(numel(uqv)) to see what value Njump should
    %be.

    idxM =  (zeros(length(unique(Ael)), prod(kSize), 'like', Ael) + numel(Ael) + 1); %originally a single
    Njump = 64; %Njump = 48; %Njump = 24;% Njump = 256;
    % Njump = 3;
    for iter = 1 : Njump : numel(uqv)

        tic
        %     M1 = sparse(   Ael(:).'== uqv( iter : iter + Njump-1 )   );
        M1 = (   Ael(:).'== uqv( iter : iter + Njump-1 )   );
        M1og = M1;
        idx_clm = sum(cumprod(~M1,2),2)+1;
        for kiter = 1 : prod(kSize) %iterate through idxM's columns.
            %step 1  - find first nonzero column for each row of M1
            idx_clm = sum(cumprod(~M1,2),2)+1;
            idxM(iter : iter + Njump-1,kiter) = idx_clm;
            %step 1B - rows with no nonzero values have value size(M1,2)+1.
            %make it 1 instead.
            idx_clm(idx_clm>numel(Ael))= 1; %idx_clm(idx_clm>size(M1,2))= 1;
            %step 2  - Make the column locations of M1 indicated by idx_clm
            %zero:
            M1 = M1.' ;
            %(0:n:(m-1)*n).' ; m-rows n-cols of M1
            idx_tmp = idx_clm + (0 : numel(Ael) : (Njump - 1) * numel(Ael)).' ;
            M1(idx_tmp) = 0;
            M1 = M1.';

        end
        toc
        disp(iter)

    end
    save(...
        strcat('./idxM/idxM_',num2str(Nx),'Nx',num2str(Ny),'Ny',num2str(Nz),'Nz_',num2str(Nkx),'Nkx',num2str(Nky),'Nky',num2str(Nkz),'Nkz.mat'),...
        'idxM', '-v7.3');

end