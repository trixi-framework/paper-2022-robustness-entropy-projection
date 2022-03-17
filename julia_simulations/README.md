The Trixi.jl elixirs contained in this folder can be used to reproduce
results in the paper "On the entropy projection and the robustness of high 
order entropy stable discontinuous Galerkin schemes for under-resolved flows".

The `data` folder contains some precomputed results in the form of `.jld2` files. 
If you wish to run simulations from scratch, delete the files in this folder.

# Crash detection elixirs

The following elixirs reproduce results in Section 3 on crash times of 
different discretizations of the compressible Euler equations for various 
numerical setups.

- `detect_crashes_2d.jl` runs numerical experiments for the 
  Kelvin-Helmholtz, Rayleigh-Taylor, and Richtmeyer-Meshkov instability 
  problems and records crash times (if the numerical simulations crashes 
  at all) and saves the result in a file `crash_detection_results_2d.txt`.
  Note: the simulation for `element_type=Tri()`, `polydeg=6`, and 
  `approximation_type=Polynomial()` has a long runtime. 
- `kelvin_helmholtz_instability.jl` runs many numerical experiments
  to record the crash times (if the numerical simulations crashes
  at all) and its dependency on the Atwood number. Results are saved
  in a file `kelvin_helmholtz_instability.txt`.
- `taylor_green_vortex.jl` runs many numerical experiments to record
  the crash times (if the numerical simulations crashes at all) and
  saves the result in a file `taylor_green_vortex.txt`.

# Kelvin-Helmholtz instability (KHI) analysis elixirs

The following elixirs reproduce results in Section 4 of the paper.

- `compute_KHI_spectra.jl` runs two 2D entropy stable elixirs and 
  processes the results to produce plots of the entropy evolution 
  over time, snapshots of density and pressure, and the power spectra. 
    - `elixir_euler_khi_dgmulti.jl` and `elixir_euler_khi_dgsem_SC_PP.jl` 
      are helper elixirs for `compute_KHI_spectra.jl`.
- `compare_entropy_projection_variants.jl` compares the robustness of 
  four different solvers (DGSEM, Gauss, and two entropy projection 
  variants "hybrid" and "staggered") at various polynomial degrees, cells 
  per dimension, and Atwood numbers. 
- `compute_entropy_evolution.jl` computes and plots the evolution of 
  entropy over time for three different entropy stable solvers (DGSEM
  with shock capturing, DGSEM with shock capturing and a positivity 
  preserving limiter, and Gauss collocation). 
