# Analyze the blow-up time of numerical simulations of A 2D inviscid
# Kelvin-Helmholtz instability for different SBP discretizations
# implemented in Trixi.jl:
# - DGSEM with and without shock capturing
# - Gauss collocation
# - Polynomial and SBP approximations on triangles
#
# For optimal performance, run this script with
#   julia --threads=1 --check-bounds=no kelvin_helmholtz_instability.jl
# to let Julia use half of your `Sys.CPU_THREADS` cores. You can also specify
# the number of processes to use via
#   julia --threads=1 --procs=N --check-bounds=no kelvin_helmholtz_instability.jl
# where `N` is the number of processes you want to use.

# Package setup on one processor to avoid duplicating precompilation work etc.
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.precompile()

using Distributed

# Add worker processes if none are present
if nprocs() == 1
  addprocs(Sys.CPU_THREADS ÷ 2)
end

# Package setup required on each processor
@everywhere begin
  using Pkg
  Pkg.activate(@__DIR__)
end
@everywhere using Trixi
@everywhere using OrdinaryDiffEq

# Setup on the driver process
using CSV, DataFrames


# Further setup required on each processor
@everywhere begin
  function set_initial_condition(atwood_number)
    A = atwood_number
    rho1 = 0.5 * one(A) # recover original with A = 3/7
    rho2 = rho1 * (1 + A) / (1 - A)
    function initial_condition_khi_nonsmooth(
        # domain size is [-1,+1]^2
        x, t, equations::CompressibleEulerEquations2D)

        # B is a discontinuous function with value 1 for -.5 <= x <= .5 and 0 elsewhere
        slope = 15
        B = 0.5 * (tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5))

        rho = rho1 + B * (rho2 - rho1)  # rho ∈ [rho_1, rho_2]
        v1 = B - 0.5                    # v1  ∈ [-.5, .5]
        v2 = 0.1 * sin(2 * pi * x[1])
        p = 1.0
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end
    return initial_condition_khi_nonsmooth
  end

  function blow_up_time(atwood_number, args...; abstol=1.0e-7, reltol=1.0e-7)

    # 2D compressible Euler equations with ratio of specific heats 7/5
    equations = CompressibleEulerEquations2D(1.4)

    # dissipative setup using EC and ED fluxes
    surface_flux = flux_lax_friedrichs
    volume_flux  = flux_ranocha

    # use a try catch block since not all parameter combinations might work
    local mesh, solver
    try
      solver, mesh = setup_solver_mesh(args..., equations, surface_flux, volume_flux)
    catch e
      @info "solver setup faild" args e
      return NaN
    end

    semi = SemidiscretizationHyperbolic(
      mesh, equations, set_initial_condition(atwood_number), solver)

    tspan = (0.0, 10.0)
    ode = semidiscretize(semi, tspan)

    alive_callback = AliveCallback(alive_interval=500)

    @info "Initializing" atwood_number args

    alg = RDPK3SpFSAL49()
    integrator = init(ode, alg, abstol=abstol, reltol=reltol,
                      dt = 1.0e-4, # small initial dt
                      save_everystep=false, callback=alive_callback)

    try
      solve!(integrator)
    catch e
      @info "Blow-up" integrator.t args e
    end

    @info "Finished" atwood_number args

    return integrator.t
  end

  function setup_solver_mesh(::Type{DGSEM},
                             polydeg, refinement_level,
                             equations, surface_flux, volume_flux)

    solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
                   volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=refinement_level,
                    periodicity=(true, true),
                    n_cells_max=100_000)

    return solver, mesh
  end

  function setup_solver_mesh(::Type{VolumeIntegralShockCapturingHG},
                             polydeg, refinement_level,
                             equations, surface_flux, volume_flux)

    basis = LobattoLegendreBasis(polydeg)
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max=0.002,
                                             alpha_min=0.0001,
                                             alpha_smooth=true,
                                             variable=density_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=volume_flux,
                                                     volume_flux_fv=surface_flux)
    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=refinement_level,
                    periodicity=(true, true),
                    n_cells_max=100_000)

    return solver, mesh
  end

  function setup_solver_mesh(::Type{GaussSBP},
                             polydeg, refinement_level,
                             equations, surface_flux, volume_flux)

    solver = DGMulti(polydeg=polydeg, element_type=Quad(),
                     approximation_type=GaussSBP(),
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    cells_per_dimension = 2^refinement_level .* (1, 1)
    mesh = DGMultiMesh(solver;
                       coordinates_min, coordinates_max, cells_per_dimension,
                       periodicity=(true, true))

    return solver, mesh
  end

  function setup_solver_mesh(::Type{SBP},
                             polydeg, refinement_level,
                             equations, surface_flux, volume_flux)

    solver = DGMulti(polydeg=polydeg, element_type=Tri(),
                     approximation_type=SBP{StartUpDG.Kubatko{StartUpDG.LegendreFaceNodes}}(),
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    cells_per_dimension = 2^refinement_level .* (1, 1)
    mesh = DGMultiMesh(solver;
                       coordinates_min, coordinates_max, cells_per_dimension,
                       periodicity=(true, true))

    return solver, mesh
  end

  function setup_solver_mesh(::Type{Polynomial},
                             polydeg, refinement_level,
                             equations, surface_flux, volume_flux)

    solver = DGMulti(polydeg=polydeg, element_type=Tri(),
                     approximation_type=Polynomial(),
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    cells_per_dimension = 2^refinement_level .* (1, 1)
    mesh = DGMultiMesh(solver;
                       coordinates_min, coordinates_max, cells_per_dimension,
                       periodicity=(true, true))

    return solver, mesh
  end
end # @everywhere


# Prepare parameters
polydeg_and_refinement_level_quad = [(3, 5), (7, 4)]
polydeg_and_refinement_level_tri = [(3, 5), (6, 4)] # SBP() is not defined for polydeg=7
atwood_numbers = range(0.1, 0.9, length=20)

# Run simulations and save data
dgsem_parameters = [(atwood_number, DGSEM, polydeg, refinement_level) for (polydeg, refinement_level) in polydeg_and_refinement_level_quad for atwood_number in atwood_numbers]
sc_parameters = [(atwood_number, VolumeIntegralShockCapturingHG, polydeg, refinement_level) for (polydeg, refinement_level) in polydeg_and_refinement_level_quad for atwood_number in atwood_numbers]
gauss_parameters = [(atwood_number, GaussSBP, polydeg, refinement_level) for (polydeg, refinement_level) in polydeg_and_refinement_level_quad for atwood_number in atwood_numbers]
polynomial_parameters = [(atwood_number, Polynomial, polydeg, refinement_level) for (polydeg, refinement_level) in polydeg_and_refinement_level_tri for atwood_number in atwood_numbers]
sbp_parameters = [(atwood_number, SBP, polydeg, refinement_level) for (polydeg, refinement_level) in polydeg_and_refinement_level_tri for atwood_number in atwood_numbers]

parameters = vcat(dgsem_parameters, sc_parameters, gauss_parameters,
                  polynomial_parameters, sbp_parameters)
blow_up_times = pmap(args -> blow_up_time(args...), parameters)
data = DataFrame(variant = map(x -> x[2], parameters),
                 polydeg = map(x -> x[3], parameters),
                 refinement_level = map(x -> x[4], parameters),
                 atwood_number = map(x -> x[1], parameters),
                 blow_up_time = blow_up_times)
CSV.write(joinpath(@__DIR__, "kelvin_helmholtz_instability.txt"), data)
