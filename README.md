This code is for the method described in an upcoming NMR in Biomedicine publication, titled "Unfolding Coil Localized Errors from an imperfect slice profileusing a Structured Autocalibration Matrix: An application to
reduce outflow effects in cine bSSFP Imaging."  

I will upload a main file with some example data to demonstrate later.  The purpose is to use a slice-encoded calibration dataset to help unfold outflow effects in 2D bSSFP imaging. 


______________________________________________________________________________________________________________________________________________________________________________________________________
The core method is func_UNCLE_SAM_NMR.  It is run for each individual frame.  
Nx:  number of readout points. Ny : number of phase encoding lines.
Nse:  number of slice encoding steps.  Nz:  Number of excited slices. Nc:  Number of channels.
Nxc:  number of readout points in calibration data.  Nyc:  number of phase-encoding steps for the calibration data.
Nzc:  number of excitated slices/slabs of the calibration data.  Np:  number of cardiac phases/ frames.
Its inputs are the following:  
  1. raw_og:  The original 2D raw data.  size Nx x Ny x Nz x Nc.  In this case Nz = 1.
  2. kSize:  kernel size ([Nkx, Nky, Nkz]).
  3.  nIter:  Maximum number of iterations (I recommend 1000)
  4.  calib:  Slice-encoded calibration:  Nxc x Nyc x Nse x Nc
  5.  bUseGPU:  1 if you want to put the k-space data in the gpu.  0 if you want to keep it on the cpu. 
  6.  idxM:  A look up table of k-space indices to quickly go from the "A matrix" to the multi-channel k-space.
  7.  noise_floor:  Noise floor of the singular values of an "A matrix."  used for singular value shrinkage. This is calculated befor the function is called from a noise dataset.
      If you do not have this, this function creates an estimate by using the edges of k-space.

The outputs are: 
  1. rawUpdate:  the reconstructed k-space:  Nx x Ny x Nse x Nc
  2. f:  Shrinkage applied on the singular values.  Determined from the calibration data.
  3. Scalib:  The singular values of the Acalib matrix (the A matrix of the calibration data).
  4. normValues  Frobenius norm calculated between adjancent reconstruction iterations. 


______________________________________________________________________________________________________________________________________________________________________________________________________
func_create_A goes from multi-channel k-space (Nx x Ny x Nz x Nc) to an "A matrix."  The original A matrix should just be 2D, but our output here
is 3D (a block-hankel structure for each channel).  We call it "Amc" in the code.  the function func_UNCLE_SAM_NMR reshapes it to be 2D.  

Its inputs are: 
  1. raw (k-space) of size Nx x Ny x NSE x Nc
  2. kernels:  Nkx x Nky x Nkz x (Nkx* Nky *Nkz) : a matrix with a value of 1 at each location of the kernel.  This value of one is at a different location as you go through the last dimension.

Its outputs are: 
  1. Amc:  a block hankel structure of k-space form each channel.  When you concatenate along the channel dimension to have a 2D A matrix. 
  2. Ael: the corresponding k-space indices of each channel of Amc.


______________________________________________________________________________________________________________________________________________________________________________________________________

func_update_raw(Amc, Ael, idxM, rawDataSize) updates the k-space data by enforcing the block hankel structure of the A matrix (input in multichannel form, Amc).  Inputs are: 
  1. Amc: 
  2. Ael:
  3. idxM:  a list of indices for each A matrix to make it easy to average the repeated k-space entries of Amc.
  4. rawDataSize:  the size of the k-space.  [Nx, Ny, Nse, Nc.

Its output is rawUpdate:  the updated k-space.  

______________________________________________________________________________________________________________________________________________________________________________________________________
func_makeIdxM(rawData, kSize):  creates the idxM matrix.  It takes a while to make for large kernels and k-space datasets.  so it saves the matrix as a local .mat file.
inputs: 
  1. rawData:  the k-space (Nx x Ny x Nse x Nc)
  2. kSize:  kernel size (Nkx x Nky x Nkz).
