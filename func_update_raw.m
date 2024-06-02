function rawUpdate = func_update_raw(Amc, Ael, idxM, rawDataSize)
    Nx= single(rawDataSize(1)); Ny= single(rawDataSize(2)); 
    Nz= single(rawDataSize(3)); Nc= single(rawDataSize(4)); 
    Ael2 = [Ael, zeros(size(Ael, 1), 1,  'like', Ael)];
    Amc2 = [Amc, zeros(size(Amc, 1), 1, Nc,  'like', Amc)];
    rawUpdate = zeros([Nx, Ny, Nz, Nc], 'like', Amc);

    for coilIter = 1 : Nc
        Amc2_singleChannel = Amc2(:, :, coilIter);
        %     rawUpdateVec = sum(Amc2_singleChannel( (idxM)), 2)./sum(Amc2_singleChannel( (idxM)) >0, 2);
        rawUpdateVec = sum(Amc2_singleChannel( (idxM)), 2)./sum(Ael2( (idxM)) >0, 2);
        rawUpdate(:, :, :, coilIter) = reshape(rawUpdateVec, [Nx, Ny, Nz]);
    end
    
end
