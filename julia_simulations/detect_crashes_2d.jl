# For optimal performance, run this script with
#   julia --threads=1 --check-bounds=no detect_crashes_2d.jl
# to let Julia use half of your `Sys.CPU_THREADS` cores. You can also specify
# the number of processes to use via
#   julia --threads=1 --procs=N --check-bounds=no detect_crashes_2d.jl
# where `N` is the number of processes you want to use.

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

@everywhere begin

   Base.@kwdef struct ProblemSetup{NameT, InitialConditionT, TspanT, CoordinateT, SourceT, PeriodicityT, BCT}
        problem_name::NameT
        initial_condition::InitialConditionT
        tspan::TspanT
        coordinates_min::CoordinateT
        coordinates_max::CoordinateT
        source_terms::SourceT
        periodicity::PeriodicityT
        boundary_conditions::BCT
    end

    ## Create problem setups

    function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D; slope = 15)
        # change discontinuity to tanh, domain size is [-1,+1]^2    
        # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
        B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
        rho = 0.5 + 0.75 * B # rho ∈ [.5, 2]
        v1 = 0.5 * (B - 1)
        v2 = 0.1 * sin(2 * pi * x[1])
        p = 1.0
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    @inline function initial_condition_rayleigh_taylor_instability(x, t,
                                                                equations::CompressibleEulerEquations2D,
                                                                slope=15)

        rho1 = 1.0
        rho2 = 2.0
        if x[2] < 0.5
            p = 2 * x[2] + 1
        else
            p = x[2] + 3 / 2
        end

        # smooth the discontinuity to avoid ambiguity at element interfaces
        smoothed_heaviside(x, left, right; slope=slope) = left + 0.5 * (1 + tanh(slope * x)) * (right - left)
        rho = smoothed_heaviside(x[2] - 0.5, rho2, rho1)

        c = sqrt(equations.gamma * p / rho)
        # the velocity is multiplied by sin(pi*y)^6 as in Remacle et al. 2003 to ensure that the
        # initial condition satisfies reflective boundary conditions at the top/bottom boundaries.
        k = 1
        v = -0.025 * c * cos(k * 8 * pi * x[1]) * sin(pi * x[2])^6
        u = 0.0

        return prim2cons(SVector(rho, u, v, p), equations)
    end

    @inline function source_terms_rayleigh_taylor_instability(u, x, t,
                                                            equations::CompressibleEulerEquations2D)
        rho, rho_v1, rho_v2, rho_e = u
        g = 1.0
        return SVector(0.0, 0.0, g * rho, g * rho_v2)
    end                         

    @inline function initial_condition_richtmeyer_meshkov_instability(x, t,
                                                                    equations::CompressibleEulerEquations2D; 
                                                                    slope = 2.0)

        #   Setup used for the Richtmeyer-Meshkov instability. Initial condition adapted from the description of
        #   https://www.youtube.com/watch?v=8K_oe_GhKzM.
        #   The domain is [0, 40/3] x [0, 40]. Boundary conditions are all reflective walls.
        
        # smooth the discontinuity to avoid ambiguity at element interfaces
        smoothed_heaviside(x, left, right; slope=slope) = left + 0.5 * (1 + tanh(slope * x)) * (right - left)

        L = 40 # domain size
        rho = smoothed_heaviside(x[2] - (18 + 2 * cos(2 * pi * 3 / L * x[1])), 1.0, .25)
        rho = rho + smoothed_heaviside(abs(x[2] - 4) - 2, 3.22, 0.0) # 2 < x < 6
        p = smoothed_heaviside(abs(x[2] - 4) - 2, 4.9, 1.0)
        u = 0.0
        v = 0.0

        return prim2cons(SVector(rho, u, v, p), equations)
    end

    function blow_up_time(element_type, approximation_type, 
                          polydeg, cells_per_dimension, problem_setup::ProblemSetup;
                          abstol=1.0e-7, reltol=1.0e-7)

        @unpack tspan, periodicity, coordinates_min, coordinates_max = problem_setup                         
        @unpack initial_condition, source_terms, boundary_conditions = problem_setup

	@show problem_setup.problem_name, polydeg, element_type, approximation_type, cells_per_dimension
            
        # 3D compressible Euler equations with ratio of specific heats 7/5
        equations = CompressibleEulerEquations2D(1.4)        
        surface_flux = flux_lax_friedrichs
        volume_flux  = flux_ranocha
    
        dg = DGMulti(polydeg=polydeg, element_type=element_type, approximation_type=approximation_type,
                     surface_integral=SurfaceIntegralWeakForm(surface_flux),
                     volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

        mesh = DGMultiMesh(dg; coordinates_min, coordinates_max, 
                           cells_per_dimension=cells_per_dimension,
                           periodicity=periodicity)
    
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                            source_terms = source_terms,
                                            boundary_conditions = boundary_conditions)
    
        ode = semidiscretize(semi, tspan)
    
        alive_callback = AliveCallback(alive_interval = 10000)
    
        integrator = init(ode, RDPK3SpFSAL49(), abstol=abstol, reltol=reltol,
                          dt = .1 * estimate_dt(mesh, dg), save_everystep=false, 
                          callback=alive_callback)
    
        try
          solve!(integrator)
        catch error
          @info "Blow-up" integrator.t error
        end
    
        return integrator.t
    end
end

khi_setup = ProblemSetup(problem_name="KHI", initial_condition = initial_condition_kelvin_helmholtz_instability, 
                         tspan=(0, 15.0), coordinates_min = (-1.0, -1.0), coordinates_max = (1.0, 1.0),                        
                         source_terms = nothing, periodicity = true, 
                         boundary_conditions = Trixi.BoundaryConditionPeriodic())

wall_bcs = (; :entire_boundary => boundary_condition_slip_wall)

rti_setup = ProblemSetup(problem_name="RTI", initial_condition = initial_condition_rayleigh_taylor_instability, 
                         tspan=(0, 15.0), coordinates_min = (0.0, 0.0), coordinates_max = (0.25, 1.0),
                         source_terms = source_terms_rayleigh_taylor_instability, periodicity = (false, false), 
                         boundary_conditions = wall_bcs)

rmi_setup = ProblemSetup(problem_name="RMI", initial_condition = initial_condition_richtmeyer_meshkov_instability, 
                         tspan=(0, 30.0), coordinates_min = (0.0, 0.0), coordinates_max = (40.0 / 3.0, 40.0),
                         source_terms = nothing, periodicity = (false, false), 
                         boundary_conditions = wall_bcs)

# Prepare parameters
## DG
mesh_resolutions = [16, 32]

element_types = (Quad(), Tri())
polydegs(elem::Quad) = 1:7
polydegs(elem::Tri) = 1:6 # SBP operators only available up to polydeg = 6
approximation_types(elem::Quad) = (SBP(), GaussSBP())
approximation_types(elem::Tri) = (SBP{StartUpDG.Kubatko{StartUpDG.LegendreFaceNodes}}(), Polynomial())

# TODO: remove later. This is to use Chen-Shu nodes in the SBP tri operators
element_types = (Tri(),)
approximation_types(elem::Tri) = (SBP{StartUpDG.Kubatko{StartUpDG.LegendreFaceNodes}}(),)

khi_parameters = [(element_type, approximation_type, polydeg, (cells_per_dimension, cells_per_dimension), khi_setup) 
                  for element_type in element_types for approximation_type in approximation_types(element_type) 
                  for cells_per_dimension in mesh_resolutions for polydeg in polydegs(element_type)]
rti_parameters = [(element_type, approximation_type, polydeg, (cells_per_dimension, 4 * cells_per_dimension), rti_setup) 
                  for element_type in element_types for approximation_type in approximation_types(element_type) 
                  for cells_per_dimension in mesh_resolutions for polydeg in polydegs(element_type)]
rmi_parameters = [(element_type, approximation_type, polydeg, (cells_per_dimension, 3 * cells_per_dimension), rmi_setup) 
                  for element_type in element_types for approximation_type in approximation_types(element_type) 
                  for cells_per_dimension in mesh_resolutions for polydeg in polydegs(element_type)]

parameters = vcat(khi_parameters, rti_parameters, rmi_parameters)                  
println("Total $(length(parameters)) parameters runs. ")

blow_up_times = pmap(args -> blow_up_time(args...), parameters)

data = DataFrame(element_type = map(x -> x[1], parameters),
                 approximation_type = map(x -> x[2], parameters),
                 polydeg = map(x -> x[3], parameters),
                 cells_per_dimension = map(x -> x[4], parameters),
                 problem_name = map(x -> x[5].problem_name, parameters),
                 blow_up_time = blow_up_times)

CSV.write(joinpath(@__DIR__, "crash_detection_results_2d.txt"), data)
