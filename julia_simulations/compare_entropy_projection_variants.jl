# For optimal performance, run this script with
#   julia --threads=1 --check-bounds=no compare_entropy_projection_variants.jl
# to let Julia use half of your `Sys.CPU_THREADS` cores. You can also specify
# the number of processes to use via
#   julia --threads=1 --procs=N --check-bounds=no compare_entropy_projection_variants.jl
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
@everywhere using Setfield

# used to modify discretization matrices for approximation_type = Val{:staggered}()
@everywhere using Trixi.StartUpDG
@everywhere using LinearAlgebra: diagm, norm, I

# Setup on the driver process
using CSV, DataFrames

@everywhere begin
    function set_initial_condition(atwood_number)
        A = atwood_number
        rho1 = 0.5 * one(A) # recover original with A = 3/7
        rho2 = rho1 * (1 + A) / (1 - A)
        function initial_condition_khi(x, t, equations::CompressibleEulerEquations2D)
            # domain size is [-1,+1]^2
        
            # B is a discontinuous function with value 1 for -.5 <= x <= .5 and 0 elsewhere
            slope = 15
            B = 0.5 * (tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)) 
            
            rho = rho1 + B * (rho2 - rho1)  # rho ∈ [rho_1, rho_2]
            v1 = B - 0.5                    # v1  ∈ [-.5, .5]
            v2 = 0.1 * sin(2 * pi * x[1])
            p = 1.0
            return prim2cons(SVector(rho, v1, v2, p), equations)
        end
        return initial_condition_khi
    end

    function build_solver(approximation_type::Val{:dgsem}, polydeg, cells_per_dimension, initial_condition, equations)
        dg = DGMulti(polydeg = polydeg, element_type = Quad(), approximation_type = SBP(),
                    surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                    volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
        mesh = DGMultiMesh(dg, cells_per_dimension=cells_per_dimension, periodicity=true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)
        return semi
    end
            
    function build_solver(approximation_type::Val{:gauss}, polydeg, cells_per_dimension, initial_condition, equations)
        dg = DGMulti(polydeg = polydeg, element_type = Quad(), approximation_type = GaussSBP(),
                    surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                    volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))
        mesh = DGMultiMesh(dg, cells_per_dimension=cells_per_dimension, periodicity=true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)
        return semi
    end

    # hybrid DGSEM (Lobatto on volume, Gauss on face)
    function build_solver(approximation_type::Val{:hybrid}, polydeg, cells_per_dimension, initial_condition, equations)
        N = polydeg            
        r1D, w1D = StartUpDG.gauss_lobatto_quad(0, 0, N)
        rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
        wq = .*(vec.(StartUpDG.NodesAndModes.meshgrid(w1D))...)

        # Clenshaw-Curtis quadrature
        function clenshaw_curtis(polydeg)
            n = polydeg + 1 # number of Chebyshev nodes
            r = sort(@. cos((2*(1:n)-1)/(2*n)*pi)) # first kind
            Vq_approx = StartUpDG.vandermonde(StartUpDG.Line(), length(r)-1, r)
            w = Vq_approx' \ [sqrt(2); zeros(size(Vq_approx, 2)-1)]
            return r, w
        end
        r1D, w1D = clenshaw_curtis(N)

        dg = DGMulti(polydeg = N, element_type = Quad(), approximation_type = Polynomial(),
                    surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                    volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
                    quad_rule_vol = (rq, sq, wq), quad_rule_face=(r1D, w1D)) 
        mesh = DGMultiMesh(dg, cells_per_dimension=cells_per_dimension, periodicity=true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)
    end

    function build_solver(approximation_type::Val{:staggered}, polydeg, cells_per_dimension, initial_condition, equations)            
        N = polydeg

        # (N+1)-Lobatto/Gauss staggering. Uses entropy projection at Gauss nodes
        # but evaluates fluxes at degree (N+1) Lobatto volume and face nodes
        r1D, w1D = StartUpDG.gauss_lobatto_quad(0, 0, N)
        rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
        wq = .*(vec.(StartUpDG.NodesAndModes.meshgrid(w1D))...)
        dg = DGMulti(polydeg = N, element_type = Quad(), approximation_type = Polynomial(),
                    surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                    volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha),
                    quad_rule_vol = (rq, sq, wq), quad_rule_face=(r1D, w1D))

        mesh = DGMultiMesh(dg, cells_per_dimension=cells_per_dimension, periodicity=true)
        semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

        # replace Lobatto volume quadrature with Gauss nodes
        r1D, w1D = StartUpDG.gauss_quad(0, 0, N)
        rq, sq = vec.(StartUpDG.NodesAndModes.meshgrid(r1D))
        wq = .*(vec.(StartUpDG.NodesAndModes.meshgrid(w1D))...)

        # keep face quadrature = Lobatto nodes
        r1D, w1D = StartUpDG.gauss_lobatto_quad(0, 0, N)

        # build lobatto RefElemData
        rd = RefElemData(Quad(), N, quad_rule_vol=(rq,sq,wq), quad_rule_face=(r1D,w1D))
        Vh = [I(rd.Np); rd.Vf] # map from nodes (Lobatto) to interpolation points (Lobatto)

        VhP = Vh * rd.Pq # rebuild VhP - rd.Pq maps from Gauss to Lobatto
        Ph = rd.M \ Matrix(Vh') # rebuild Ph - rd.M is now dense instead of full

        _, VhP, Ph = hybridized_SBP_operators(rd)

        # swap out quadrature interpolation, projection, lift operators 
        # (anything involving the volume quadrature nodes or mass matrix).
        # These operators are not used in the actual flux differencing kernel
        dg.basis.Vq .= rd.Vq # maps from Lobatto nodes to Gauss nodes
        semi.cache.VhP .= VhP
        semi.cache.Ph .= Ph    

        # new LIFT built from Gauss volume quadrature and Lobatto surface quadrature
        dg.basis.LIFT .= rd.LIFT 

        # just used for AnalysisCallback; not strictly necessary
        mesh.md.wJq .= diagm(rd.wq) * (rd.Vq * mesh.md.J)                         

        return semi
    end

    function blow_up_time(approximation_type, polydeg, cells_per_dimension, Atwood_number; 
                          abstol=1.0e-7, reltol=1.0e-7)

    	println("Running with approximation_type=$(approximation_type), polydeg=$polydeg, cells_per_dimension=$cells_per_dimension, Atwood=$Atwood_number")

        initial_condition = set_initial_condition(Atwood_number)
        equations = CompressibleEulerEquations2D(1.4)

        semi = build_solver(Val{approximation_type}(), polydeg, cells_per_dimension, 
                            initial_condition, equations)

        tspan = (0.0, 10.0)
        ode = semidiscretize(semi, tspan)

        alive_callback = AliveCallback(alive_interval = 10)

        integrator = init(ode, RDPK3SpFSAL49(), abstol=abstol, reltol=reltol,
                          dt = 1e-3 * estimate_dt(semi.mesh, semi.solver), 
                          save_everystep=false, callback=alive_callback)
        try
            solve!(integrator)
        catch e
            @warn "Blow-up" e
        end
        integrator.t
    end
end

polydegs = (3, 7)
mesh_resolutions = (32, 16)

approximation_types = (:dgsem, :gauss, :hybrid, :staggered)
atwood_numbers = range(0.1, 0.9, length=20)

parameters = [(approximation_type, polydeg, cells_per_dimension, A) 
               for (polydeg, cells_per_dimension) in zip(polydegs, mesh_resolutions)
               for approximation_type in approximation_types for A in atwood_numbers]

blow_up_times = pmap(args -> blow_up_time(args...), parameters)

data = DataFrame(approximation_type = map(x -> x[1], parameters),
                 polydeg = map(x -> x[2], parameters),
                 cells_per_dimension = map(x -> x[3], parameters),
                 atwood_number = map(x -> x[4], parameters),
                 blow_up_time = blow_up_times)

CSV.write(joinpath(@__DIR__, "compare_entropy_projection_variants.txt"), data)


# extra code to plot snapshots of density for Atwood number 3/7 
function plot_sol(approximation_type, polydeg, cells_per_dimension, Atwood_number=3/7; 
                  abstol=1.0e-7, reltol=1.0e-7)

    println("Running with approximation_type=$(approximation_type), polydeg=$polydeg, cells_per_dimension=$cells_per_dimension, Atwood=$Atwood_number")

    initial_condition = set_initial_condition(Atwood_number)
    equations = CompressibleEulerEquations2D(1.4)

    semi = build_solver(Val{approximation_type}(), polydeg, cells_per_dimension, 
                        initial_condition, equations)

    tspan = (0.0, 5.0)
    ode = semidiscretize(semi, tspan)

    alive_callback = AliveCallback(alive_interval = 10)

    integrator = init(ode, RDPK3SpFSAL49(), abstol=abstol, reltol=reltol,
                        dt = 1e-3 * estimate_dt(semi.mesh, semi.solver), 
                        save_everystep=false, callback=alive_callback)
    sol = solve!(integrator)

    fig, ax, plt = plot(Trixi.PlotData2D(sol)["rho"], plot_mesh=false)
    Colorbar(fig[1,2], plt)
    ax.aspect=DataAspect(); Makie.tightlimits!(ax); hidedecorations!(ax); hidespines!(ax)
    set_theme!(Theme(fontsize=20))
    save("khi_density_$(string(approximation_type))_p$(polydeg)_$(cells_per_dimension)_cells_t5.png", fig.scene)

    return sol
end
# plot_sol(:gauss, 3, 64)
# plot_sol(:hybrid, 3, 64)
# plot_sol(:staggered, 3, 64)