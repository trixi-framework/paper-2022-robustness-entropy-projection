# LGL-DGSEM simulations of the 2D KHI example (Euler) using FLUXO's subcell positivity-preserving method

## Requirements
* Fortran compiler
* HDF5 with fortran support
* CMake (minimum required version is 3.5.1)
* MPI library with Fortran support (optional)
## Instructions

To run the simulations tests, follow the instructions:

* Move to this directory and Clone the fluxo repository:
  ```
  cd paper-robustness-entropy-projection/code/fluxo_simulations
  git clone git@github.com:project-fluxo/fluxo.git
  ```
* Check out the branch of fluxo where the IDP methods are implemented:
  ```
  cd fluxo
  git checkout becd019432e9f4fdd47828a42d3865cc93ba5410
  cd ..
  ```
* Build fluxo with the right cmake parameters:
  ```
  mkdir build
  cd build
  cmake ../fluxo/ -DCMAKE_BUILD_TYPE=Release -DFLUXO_SHOCKCAPTURE=ON -DFLUXO_SHOCKCAP_NFVSE=ON -DFLUXO_FV_TIMESTEP=ON -DFLUXO_SHOCK_NFVSE_CORR -DNFVSE_LOCAL_ALPHA=ON 
  make -j
  cd ..
  ```
* Run each test using fluxo. For example:
  ```
  cd 01_N3_64elems
  mpiexec -n <number-of-ranks> ../build/bin/fluxo parameter_res0.ini
  ```

All results were obtained with [this version of fluxo](https://github.com/project-fluxo/fluxo/tree/becd019432e9f4fdd47828a42d3865cc93ba5410).
