# Analyze the blow-up time of numerical simulations of the 3D inviscid
# Taylor-Green vortex for different dissipation-free SBP discretizations
# implemented in Trixi.jl:
# - DGSEM
# - Gauss collocation
# - Periodic FD SBP methods
# - TODO: One-block FD SBP methods?
#
# For optimal performance, run this script with
#   julia --threads=1 --check-bounds=no taylor_green_vortex.jl
# to let Julia use half of your `Sys.CPU_THREADS` cores. You can also specify
# the number of processes to use via
#   julia --threads=1 --procs=N --check-bounds=no taylor_green_vortex.jl
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
@everywhere using SummationByPartsOperators
@everywhere using Trixi
@everywhere using OrdinaryDiffEq

# Setup on the driver process
using CSV, DataFrames


# Further setup required on each processor
@everywhere begin
  # classical inviscid Taylor-Green vortex
  function initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)
    A  = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1  =  A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2  = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3  = 0.0
    p   = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p   = p + A^2 / 16 * rho * (cos(2*x[1])*cos(2*x[3]) + 2*cos(2*x[2]) + 2*cos(2*x[1]) + cos(2*x[2])*cos(2*x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
  end

  function blow_up_time(args...; abstol=1.0e-7, reltol=1.0e-7)

    # 3D compressible Euler equations with ratio of specific heats 7/5
    equations = CompressibleEulerEquations3D(1.4)

    # dissipation-free setup using EC fluxes
    surface_flux = flux_ranocha
    volume_flux  = flux_ranocha

    # use a try catch block since not all accuracy order of FD operators
    # work with all numbers of nodes
    local mesh, solver
    try
      solver, mesh = setup_solver_mesh(args..., surface_flux, volume_flux)
    catch e
      @info "solver setup faild" args e
      return NaN
    end

    semi = SemidiscretizationHyperbolic(
      mesh, equations, initial_condition_taylor_green_vortex, solver)

    tspan = (0.0, 20.0) # We used (0.0, 50.0) before but crashes are earlier
    ode = semidiscretize(semi, tspan)

    alive_callback = AliveCallback(alive_interval=1000)

    integrator = init(ode, RDPK3SpFSAL49(), abstol=abstol, reltol=reltol,
                      save_everystep=false, callback=alive_callback)

    try
      solve!(integrator)
    catch e
      @info "Blow-up" integrator.t args e
    end

    return integrator.t
  end

  function setup_solver_mesh(::Type{DGSEM}, polydeg, refinement_level,
                             surface_flux, volume_flux)

    solver = DGSEM(polydeg=polydeg, surface_flux=surface_flux,
                   volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0, -1.0) .* π
    coordinates_max = ( 1.0,  1.0,  1.0) .* π
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=refinement_level,
                    periodicity=(true, true, true),
                    n_cells_max=100_000)

    return solver, mesh
  end

  function setup_solver_mesh(::Type{GaussSBP}, polydeg, refinement_level,
                             surface_flux, volume_flux)

    solver = DGMulti(polydeg=polydeg, element_type=Hex(),
                     approximation_type=GaussSBP(),
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0, -1.0) .* π
    coordinates_max = ( 1.0,  1.0,  1.0) .* π
    cells_per_dimension = 2^refinement_level .* (1, 1, 1)
    mesh = DGMultiMesh(solver;
                       coordinates_min, coordinates_max, cells_per_dimension,
                       periodicity=(true, true, true))

    return solver, mesh
  end

  # CGSEM
  function setup_solver_mesh(::typeof(couple_continuously),
                             polydeg, refinement_level,
                             surface_flux, volume_flux)

    approximation_type = couple_continuously(
      legendre_derivative_operator(xmin=0.0, xmax=1.0, N=polydeg+1),
      UniformPeriodicMesh1D(xmin=0.0, xmax=1.0, Nx=2^refinement_level))
    solver = DGMulti(element_type=Hex(), approximation_type=approximation_type,
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0, -1.0) .* π
    coordinates_max = ( 1.0,  1.0,  1.0) .* π
    mesh = DGMultiMesh(solver; coordinates_min, coordinates_max)

    return solver, mesh
  end

  # Periodic FD
  function setup_solver_mesh(::typeof(periodic_derivative_operator),
                             accuracy_order, nnodes_per_dimension,
                             surface_flux, volume_flux)

    approximation_type = periodic_derivative_operator(
      derivative_order=1, accuracy_order=accuracy_order,
      xmin=0.0, xmax=1.0, N=nnodes_per_dimension)
    solver = DGMulti(element_type=Hex(), approximation_type=approximation_type,
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0, -1.0) .* π
    coordinates_max = ( 1.0,  1.0,  1.0) .* π
    mesh = DGMultiMesh(solver; coordinates_min, coordinates_max)

    return solver, mesh
  end

  # SBP FD
  function setup_solver_mesh(::typeof(derivative_operator),
                             accuracy_order, nnodes_per_dimension,
                             surface_flux, volume_flux)

    approximation_type = derivative_operator(
      MattssonNordström2004(),
      # MattssonAlmquistCarpenter2014Extended(),
      # MattssonAlmquistCarpenter2014Optimal(),
      # MattssonAlmquistVanDerWeide2018Minimal(),
      # MattssonAlmquistVanDerWeide2018Accurate(),
      derivative_order=1, accuracy_order=accuracy_order,
      xmin=0.0, xmax=1.0, N=nnodes_per_dimension)
    solver = DGMulti(element_type=Hex(), approximation_type=approximation_type,
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

    coordinates_min = (-1.0, -1.0, -1.0) .* π
    coordinates_max = ( 1.0,  1.0,  1.0) .* π
    cells_per_dimension = (1, 1, 1)
    mesh = DGMultiMesh(solver; coordinates_min, coordinates_max, cells_per_dimension)

    return solver, mesh
  end
end # @everywhere


# Prepare parameters
## DG
polydegs = 1:7
refinement_levels = 1:3
## FD
accuracy_orders = 2:2:10
nnodes_per_dimension = 4:2:12

# Run simulations and save data
dgsem_parameters = [(DGSEM, polydeg, refinement_level) for refinement_level in refinement_levels for polydeg in polydegs]
gauss_parameters = [(GaussSBP, polydeg, refinement_level) for refinement_level in refinement_levels for polydeg in polydegs]
cgsem_parameters = [(couple_continuously, polydeg, refinement_level) for refinement_level in refinement_levels for polydeg in polydegs]
fd_periodic_parameters = [(periodic_derivative_operator, accuracy_order, nnodes) for accuracy_order in accuracy_orders for nnodes in nnodes_per_dimension]

parameters = vcat(dgsem_parameters, gauss_parameters, cgsem_parameters,
                  fd_periodic_parameters)
blow_up_times = pmap(args -> blow_up_time(args...), parameters)
data = DataFrame(variant = map(x -> x[1], parameters),
                 polydeg_or_accuracy_order = map(x -> x[2], parameters),
                 refinement_level_or_nnodes = map(x -> x[3], parameters),
                 blow_up_time = blow_up_times)
CSV.write(joinpath(@__DIR__, "taylor_green_vortex.txt"), data)
