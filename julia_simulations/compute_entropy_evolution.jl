
using OrdinaryDiffEq
using Trixi
using CairoMakie
using DataFrames, DelimitedFiles

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

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

function setup_solver(solver::Val{:dgsem}, polydeg, cells_per_dimension, initial_condition; 
                      surface_flux = flux_lax_friedrichs
    )
    basis = LobattoLegendreBasis(polydeg)
    volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)
    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=Int(log2(cells_per_dimension)),
                    n_cells_max=100_000)
                    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    return semi
end

function setup_solver(solver::Val{:dgsem_SC}, polydeg, cells_per_dimension, initial_condition;
                      surface_flux = flux_lax_friedrichs)
    basis = LobattoLegendreBasis(polydeg)
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                            alpha_max=0.0025,
                                            alpha_min=0.0001,
                                            alpha_smooth=true,
                                            variable=density_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg=flux_ranocha,
                                                     volume_flux_fv=surface_flux)
    # volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)
    solver = DGSEM(basis, surface_flux, volume_integral)

    coordinates_min = (-1.0, -1.0)
    coordinates_max = ( 1.0,  1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max,
                    initial_refinement_level=Int(log2(cells_per_dimension)),
                    n_cells_max=100_000)
                    
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)
    return semi
end

function setup_solver(solver::Val{:gauss}, polydeg, cells_per_dimension, initial_condition;
                      surface_flux = flux_lax_friedrichs)
    dg = DGMulti(polydeg = polydeg, element_type = Quad(), approximation_type = GaussSBP(),
                 surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
                 volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

    equations = CompressibleEulerEquations2D(1.4)
    mesh = DGMultiMesh(dg, cells_per_dimension=(cells_per_dimension, cells_per_dimension), periodicity=true)
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)
    return semi
end

atwood_number = 3/7 

entropy_over_time = Dict{Tuple{Symbol, Int, Int}, Tuple{Vector{Float64}, Vector{Float64}}}()
for (polydeg, cells_per_dimension) in ((3, 64), (7, 32))

    initial_condition = set_initial_condition(atwood_number)
    semi_dgsem = setup_solver(Val{:dgsem}(), polydeg, cells_per_dimension, initial_condition)
    semi_dgsem_SC = setup_solver(Val{:dgsem_SC}(), polydeg, cells_per_dimension, initial_condition)
    semi_gauss = setup_solver(Val{:gauss}(), polydeg, cells_per_dimension, initial_condition)

    ###############################################################################
    # run the simulations

    # a very conservative initial estimate for dt
    h = 2 / cells_per_dimension
    polydeg_factor = (polydeg + 1)^2 
    dt0 = 1e-2 / (polydeg_factor * h)

    # dgsem, tspan is set to be right before the crash
    if polydeg==3 && cells_per_dimension==64
        tspan = (0.0, 3.9618)
    elseif polydeg==7 && cells_per_dimension==32
        tspan = (0.0, 3.4380)
    else
        tspan = (0.0, 7.5)
    end
    sol_DGSEM = solve(semidiscretize(semi_dgsem, tspan), RDPK3SpFSAL49(), abstol=1.0e-7, reltol=1.0e-7,
                    dt = dt0, save_everystep=false, saveat=LinRange(tspan..., 100), 
                    callback=AliveCallback(alive_interval=100))

    # # dgsem with shock capturing, tspan is set to be right before the crash
    # if polydeg==3 && cells_per_dimension==64
    #     tspan=(0.0, 4.8891)
    # elseif polydeg==7 && cells_per_dimension==32
    #     tspan=(0.0, 5.0569)
    # else
    #     tspan = (0.0, 7.5)        
    # end
    # sol_DGSEM_SC = solve(semidiscretize(semi_dgsem_SC, tspan), RDPK3SpFSAL49(), abstol=1.0e-7, reltol=1.0e-7,
    #                      dt = dt0, save_everystep=false, saveat=LinRange(tspan..., 100), 
    #                      callback=AliveCallback(alive_interval=100))                             

    # dgsem with both shock capturing and positivity-preserving limiting
    tspan = (0.0, 7.5)
    limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                   variables=(Trixi.density, pressure))
    sol_DGSEM_SC_PP = solve(semidiscretize(semi_dgsem_SC, tspan), SSPRK43(limiter!), 
                            saveat=LinRange(tspan..., 100), save_everystep=false, 
                            callback=AliveCallback(alive_interval=100))

    # gauss collocation
    tspan = (0.0, 7.5)
    sol_Gauss = solve(semidiscretize(semi_gauss, tspan), RDPK3SpFSAL49(), abstol=1.0e-7, reltol=1.0e-7,
                      dt = dt0, save_everystep=false, saveat=LinRange(tspan..., 100), 
                      callback=AliveCallback(alive_interval=100))

    # ==================== plot entropy evolution over time ==================== 

    include("utilities.jl")
    S_DGSEM = compute_entropy_over_time(sol_DGSEM)
    S_Gauss = compute_entropy_over_time(sol_Gauss)
    S_DGSEM_SC_PP = compute_entropy_over_time(sol_DGSEM_SC_PP)

    entropy_over_time[(:dgsem, polydeg, cells_per_dimension)] = (S_DGSEM, sol_DGSEM.t)
    entropy_over_time[(:gauss, polydeg, cells_per_dimension)] = (S_Gauss, sol_Gauss.t)
    entropy_over_time[(:dgsem_SC_PP, polydeg, cells_per_dimension)] = (S_DGSEM_SC_PP, sol_DGSEM_SC_PP.t)
end

filename = "entropy_over_time.jld2"
@save filename entropy_over_time 
@load filename entropy_over_time

df = DataFrame(solver="", polydeg=Int[], cells_per_dimension=Int[], entropy=Float64[], time=Float64[])
for solver in ("dgsem", "gauss", "dgsem_SC_PP")
    for (polydeg, cells_per_dimension) in ((3,64), (7, 32))
        append!(df, DataFrame(solver=solver, 
                              polydeg=polydeg, 
                              cells_per_dimension=cells_per_dimension, 
                              entropy=entropy_over_time[(Symbol(solver), polydeg, cells_per_dimension)][1],
                              time=entropy_over_time[(Symbol(solver), polydeg, cells_per_dimension)][2]))
    end
end

# read FLUXO results in
data = readdlm("data/out.Euler_KHI_N3_64elem_res0.dat")
t = map(x -> parse(Float64, x[1:end-1]), data[5:end, 1])
S = map(x -> parse(Float64, x[1:end-1]), data[5:end, 30])
t, S = map(x->getindex(x, 1:findlast(@. t <= 7.5)), (t, S * (equations.gamma - 1) / 2))
S .-= minimum(S)
append!(df, DataFrame(solver="dgsem_subcell", polydeg=3, cells_per_dimension=64, entropy=S, time=t))

data = readdlm("data/out.Euler_KHI_N7_32elem_res0.dat")
t = map(x -> parse(Float64, x[1:end-1]), data[5:end, 1])
S = map(x -> parse(Float64, x[1:end-1]), data[5:end, 30])
t, S = map(x->getindex(x, 1:findlast(@. t <= 7.5)), (t, S * (equations.gamma - 1) / 2))
S .-= minimum(S)
append!(df, DataFrame(solver="dgsem_subcell", polydeg=7, cells_per_dimension=32, entropy=S, time=t))

with_theme(Theme(fontsize=22, linewidth=4)) do 
    fig = Figure(resolution=(1500, 600))

    data = filter("polydeg" => polydeg->polydeg == 3, df)
    S_DGSEM, t_DGSEM = data[findall(data[:,:solver] .== "dgsem"), :entropy], data[findall(data[:,:solver] .== "dgsem"), :time]
    S_Gauss, t_Gauss = data[findall(data[:,:solver] .== "gauss"), :entropy], data[findall(data[:,:solver] .== "gauss"), :time]
    S_DGSEM_SC_PP, t_DGSEM_SC_PP = data[findall(data[:,:solver] .== "dgsem_SC_PP"), :entropy], data[findall(data[:,:solver] .== "dgsem_SC_PP"), :time]
    S_DGSEM_subcell, t_DGSEM_subcell = data[findall(data[:,:solver] .== "dgsem_subcell"), :entropy], data[findall(data[:,:solver] .== "dgsem_subcell"), :time]
    S_DGSEM_subcell .+= (S_Gauss[1] - S_DGSEM_subcell[1])

    ax = Axis(fig[1, 1], xlabel="Time")
    Makie.lines!(ax, t_DGSEM, S_DGSEM, label="DGSEM")
    Makie.lines!(ax, t_DGSEM_SC_PP, S_DGSEM_SC_PP, label="DGSEM-SC-PP", linestyle=:dash)
    Makie.lines!(ax, t_DGSEM_subcell, S_DGSEM_subcell, label="DGSEM-subcell", linestyle=:dot)
    Makie.lines!(ax, t_Gauss, S_Gauss, label = "Gauss")
    axislegend(ax, position=:lb)
    Label(fig[1, 1, Bottom()], L"$N = 3$ mesh of $64^2$ elements", padding=(0, 0, 0, 80), textsize=32)

    data = filter("polydeg" => polydeg->polydeg == 7, df)
    S_DGSEM, t_DGSEM = data[findall(data[:,:solver] .== "dgsem"), :entropy], data[findall(data[:,:solver] .== "dgsem"), :time]
    S_Gauss, t_Gauss = data[findall(data[:,:solver] .== "gauss"), :entropy], data[findall(data[:,:solver] .== "gauss"), :time]
    S_DGSEM_SC_PP, t_DGSEM_SC_PP = data[findall(data[:,:solver] .== "dgsem_SC_PP"), :entropy], data[findall(data[:,:solver] .== "dgsem_SC_PP"), :time]
    S_DGSEM_subcell, t_DGSEM_subcell = data[findall(data[:,:solver] .== "dgsem_subcell"), :entropy], data[findall(data[:,:solver] .== "dgsem_subcell"), :time]
    S_DGSEM_subcell .+= (S_Gauss[1] - S_DGSEM_subcell[1])

    ax = Axis(fig[1, 2], xlabel="Time")
    Makie.lines!(ax, t_DGSEM, S_DGSEM, label="DGSEM")
    Makie.lines!(ax, t_DGSEM_SC_PP, S_DGSEM_SC_PP, label="DGSEM-SC-PP", linestyle=:dash)
    Makie.lines!(ax, t_DGSEM_subcell, S_DGSEM_subcell, label="DGSEM-subcell", linestyle=:dot)
    Makie.lines!(ax, t_Gauss, S_Gauss, label = "Gauss")
    axislegend(ax, position=:lb)
    Label(fig[1, 2, Bottom()], L"$N = 7$ mesh of $32^2$ elements", padding=(0, 0, 0, 80), textsize=32)

    fig
    save("entropy_over_time.png", fig.scene)    
end