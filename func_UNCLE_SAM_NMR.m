function [rawUpdate, f, Scalib, normValues] = func_UNCLE_SAM_NMR(raw_og, kSize, nIter, calib, bUseGPU, idxM)
%put a description up here.  Acknowledge Mark and Ajin. Cite Mark, Lustig, and
%Haldar. 

normValues = zeros(1, nIter);

%Mark Bydder, Ajin Joy, Andres Saucedo, and Zhoahuan Zhang helped and
%contributed to developing this.  

    [Nx, Ny, Nz, Nc] = size(raw_og);
    [Nxc, Nyc, Nzc, Ncc ] = size(calib);

    %Noise taken from the minimum k-space elements.  Copied from Mark Bydder ---------------------
    %Taken from Mark's sake.  
    %sorts all acquired k-space data.
    tmp = nonzeros(raw_og); 
    %sorts all data in acending order.  for k-space data, this puts the
    %large magnitudes at the edges. 
    tmp = sort([real(tmp); imag(tmp)]);
    %gets rid of the higher values (which was moved to hte perifery in the
    %previous line.
    k = ceil(numel(tmp)/10); tmp = tmp(k:end-k+1); % trim 20%
    std1 = 1.4826 * median(abs(tmp-median(tmp))) * sqrt(2);

    noise_floor = std1 * sqrt(nnz(raw_og)/Nc);
    %----------------------------------------------------------------------------------------------


    %GPU %Thanks, Mark
    if (bUseGPU)
        gpu = gpuDevice;
        raw_og = gpuArray(raw_og);
        fprintf('GPU found: %s (%.1f Gb)\n',gpu.Name,gpu.AvailableMemory/1e9);
    end

    %prepare kernel-------------------------
    Nkx = kSize(1); Nky =  kSize(2); Nkz = kSize(3);
    Nk = prod(kSize);
    kernels = eye(Nk, Nk);
    kernels = flip(kernels, 2);
    kernels = reshape(kernels, [Nkx, Nky, Nkz, Nk]);
    kernels = permute(kernels, [2, 1, 3, 4]);

    %prepare Ael (raw_og indices of a single channel arranged in an A

    %---------------------------------------

    rawUpdate = raw_og;
    
    %calibration matrix
    [Acalib, ~] = func_create_A(calib,  kernels); 
    %Acal is actually of size 
    %(Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)),  Nk, Nc.  It
    %can reshaped to the proper size of :
    %(Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)),  Nk * Nc.
    %Aelcal is of size (Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)),  Nk
     Acalib = reshape(Acalib ,  [(Nxc - (Nkx - 1)) * (Nyc - (Nky - 1)) * (Nzc - (Nkz - 1)), Ncc * Nk] );

    [~, Scalib, Vcalib] = svd(Acalib'*Acalib, 'econ'); Scalib = sqrt(Scalib);
   f = max(0,1-noise_floor.^2./(diag(Scalib)).^2);
   %hard truncation:
%    f = max(0, (diag(Scalib) - noise_floor)./abs(diag(Scalib) - noise_floor));


% f = max(0, (diag(Scalib) - 0.75*noise_floor)./abs(diag(Scalib) - 0.75*noise_floor));
% f = max(0, (diag(Scalib) - 1.15*noise_floor)./abs(diag(Scalib) - 1.5*noise_floor));
    truncMatrix = diag(f);

    [A, Ael] = func_create_A(raw_og,  kernels); 
    %A is arranged as (Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)),  Nk, Nc
    A = reshape(A ,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)), Nc * Nk] );
    %fast indices method for data consistency:  Thanks, Mark Bydder. 
    ix = (A ~= 0);
    val = A(ix);

    Ap = A; %"previous A" for the first iteration .
    
    for iter = 1 : nIter
        
        A = A * (Vcalib * truncMatrix * (Vcalib)'); %updates A. 
        A(ix) = val; % data consistency  
        A = reshape(A, [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)), Nk, Nc]); %put A in the multi-channel format.
        rawUpdate = func_update_raw(A, Ael, idxM, [Nx, Ny, Nz, Nc]); %Enforce Hankel Structure.
        [A, Ael] = func_create_A(rawUpdate,  kernels); %Update A matrix (output is in the multichannel format).
        A = reshape(A ,  [(Nx - (Nkx - 1)) * (Ny - (Nky - 1)) * (Nz - (Nkz - 1)), Nk * Nc] );
%         disp(iter)
        normValues(1, iter) = norm(A - Ap, "fro");
        Ap = A; % Store this A (which has structural and data consistency) 
                % as the "previous A" for the next iteration.  
    end

end
