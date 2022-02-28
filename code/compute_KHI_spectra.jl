# using Plots
using CairoMakie
using Trixi, OrdinaryDiffEq
using LinearAlgebra: diagm, norm, kron
using FFTW
using JLD2
using LaTeXStrings

# generate data and save to a JLD2 file
for (polydeg, cells_per_dimension) in ((3, 64), (7, 32))

    T_final = 25.0 # replace with very small final time if we're just loading JLD2 files
    tsave = LinRange(0, T_final, 2) # no need to save time history

    # Same as the default Trixi example, but slope = 25 and v2 is a non-symmetric perturbation
    function initial_condition_khi_nonsymmetric_perturbation(x, t, equations::CompressibleEulerEquations2D)
        # change discontinuity to tanh
        # typical resolution 128^2, 256^2
        # domain size is [-1,+1]^2
        slope = 15
        # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
        B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
        rho = 0.5 + 0.75 * B
        v1 = 0.5 * (B - 1)
        # scale by (1 + .01 * sin(pi*x) * sin(pi*y)) to break symmetry
        v2 = 0.1 * sin(2 * pi * x[1]) * (1.0 + 1e-2 * sin(pi * x[1]) * sin(pi * x[2]))
        p = 1.0
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    filename = "data/sol_Gauss_p$(polydeg)_$(cells_per_dimension)_cells.jld2"
    if isfile(filename)==false
        trixi_include("elixir_euler_khi_dgmulti.jl", approximation_type = GaussSBP(),
                    initial_condition = initial_condition_khi_nonsymmetric_perturbation,
                    tspan = (0, T_final), polydeg = polydeg, tsave = tsave, 
                    cells_per_dimension = (cells_per_dimension))
        sol_Gauss = deepcopy(sol)
        @save filename sol_Gauss
    end

    filename = "data/sol_DGSEM_SC_PP_p$(polydeg)_$(cells_per_dimension)_cells.jld2"
    if isfile(filename)==false
        trixi_include("elixir_euler_khi_dgsem_SC_PP.jl",
                    tspan = (0, T_final), polydeg = polydeg, tsave = tsave, 
                    cells_per_dimension = (cells_per_dimension),
                    alpha_max=0.005, 
                    initial_condition=initial_condition_khi_nonsymmetric_perturbation)
        sol_DGSEM_SC_PP = deepcopy(sol)
        @save filename sol_DGSEM_SC_PP
    end
end

# ==================== plot solution heatmaps ==================== 

include("utilities.jl")
include("read_fluxo_utilities.jl")

for (polydeg, cells_per_dimension) in ((3, 64), (7, 32))

    @load "data/sol_Gauss_p$(polydeg)_$(cells_per_dimension)_cells.jld2" sol_Gauss
    @load "data/sol_DGSEM_SC_PP_p$(polydeg)_$(cells_per_dimension)_cells.jld2" sol_DGSEM_SC_PP

    # read in flux solution
    filename = "data/sol_ascii_N$(polydeg)_$(cells_per_dimension)elems.dat"
    polydeg, total_cells, x, y, rho, u, v, p = read_fluxo_sol(filename)

    # hack workaround - copy FLUXO's solution into Trixi's solution container
    # and use existing Trixi visualization routines. 
    sol_DGSEM_subcell = deepcopy(sol_DGSEM_SC_PP)
    sol_u = Trixi.wrap_array(sol_DGSEM_subcell.u[end], sol_DGSEM_subcell.prob.p)
    cache = sol_DGSEM_subcell.prob.p.cache
    cache.elements.node_coordinates[1, :, :, :] .= copy(reshape(x, polydeg + 1, polydeg + 1, total_cells))
    cache.elements.node_coordinates[2, :, :, :] .= copy(reshape(y, polydeg + 1, polydeg + 1, total_cells))
    sol_u[1, :, :, :] .= reshape(rho, polydeg + 1, polydeg + 1, total_cells)
    sol_u[2, :, :, :] .= reshape(u, polydeg + 1, polydeg + 1, total_cells)
    sol_u[3, :, :, :] .= reshape(v, polydeg + 1, polydeg + 1, total_cells)
    sol_u[4, :, :, :] .= reshape(p, polydeg + 1, polydeg + 1, total_cells)

    with_theme(Theme(markersize = 5, fontsize=18)) do 
        fig = Figure()
        rows, cols = 3, 2
        axes = [Makie.Axis(fig[i, j]) for j in 1:rows, i in 1:cols]
        row_list, col_list = [i for j in 1:rows, i in 1:cols], [j for j in 1:rows, i in 1:cols]

        limits = extrema(getindex.(sol_Gauss.u[end], 1))
        plot!(fig[1, 1], Trixi.PlotData2DTriangulated(sol_DGSEM_SC_PP)["rho"], plot_mesh=false, colorrange=limits)
        plot!(fig[1, 2], Trixi.PlotData2DTriangulated(sol_DGSEM_subcell)["rho"], plot_mesh=false, colorrange=limits)
        plot!(fig[1, 3], Trixi.PlotData2D(sol_Gauss)["rho"], plot_mesh=false, colorrange=limits)
        Label(fig[1, 1, Bottom()], LaTeXString("DGSEM-SC-PP density"), padding=(0, 0, 5, 5))
        Label(fig[1, 2, Bottom()], LaTeXString("DGSEM-subcell density"))
        Label(fig[1, 3, Bottom()], LaTeXString("Gauss density"))
        Colorbar(fig[1, 4], limits=limits, colormap=Trixi.default_Makie_colormap())
        rowsize!(fig.layout, 1, Aspect(2, 1))

        limits = extrema(getindex.(sol_Gauss.u[end], 4))
        plot!(fig[2, 1], Trixi.PlotData2DTriangulated(sol_DGSEM_SC_PP)["p"], plot_mesh=false, colorrange=limits)
        plot!(fig[2, 2], Trixi.PlotData2DTriangulated(sol_DGSEM_subcell)["p"], plot_mesh=false, colorrange=limits)
        plot!(fig[2, 3], Trixi.PlotData2D(sol_Gauss)["p"], plot_mesh=false, colorrange=limits)
        Label(fig[2, 1, Bottom()], LaTeXString("DGSEM-SC-PP pressure"), padding=(0, 0, 5, 5))
        Label(fig[2, 2, Bottom()], LaTeXString("DGSEM-subcell pressure"))
        Label(fig[2, 3, Bottom()], LaTeXString("Gauss pressure"))
        Colorbar(fig[2, 4], limits=limits, colormap=Trixi.default_Makie_colormap())
        rowsize!(fig.layout, 2, Aspect(2, 1))

        for ax in axes
            ax.aspect=DataAspect(); Makie.tightlimits!(ax); hidedecorations!(ax); hidespines!(ax)
        end
        fig

        save("khi_polydeg_$(polydeg)_$(cells_per_dimension)_cells.png", fig.scene)
    end
end

# ==================== plot spectra ==================== 

set_theme!()
with_theme(Theme(markersize = 4, fontsize=22)) do    
    fig = Figure(resolution=(1500, 600))
    rows, cols = 1, 2
    axes = [Makie.Axis(fig[i, j], xscale=log10, yscale=log10) for i in 1:rows, j in 1:cols]
    
    @load "data/sol_Gauss_p3_64_cells.jld2" sol_Gauss
    @load "data/sol_DGSEM_SC_PP_p3_64_cells.jld2" sol_DGSEM_SC_PP
    filename = "data/sol_ascii_N3_64elems.dat"
    polydeg, total_cells, x, y, rho, u, v, p = read_fluxo_sol(filename)
    
    Ek_Gauss, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(sol_Gauss.u[end], sol_Gauss.prob.p)[1:4]...)
    Ek_DGSEM_SC_PP, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(sol_DGSEM_SC_PP.u[end], sol_DGSEM_SC_PP.prob.p)[1:4]...)
    Ek_subcell, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(polydeg, total_cells, x, y, rho, u, v, p)[1:4]...)
    p = sortperm(k1D)

    Makie.scatter!(axes[1, 1], k1D, Ek_Gauss, label="Gauss")
    Makie.scatter!(axes[1, 1], k1D, Ek_subcell, label="DGSEM-Subcell")
    Makie.scatter!(axes[1, 1], k1D, Ek_DGSEM_SC_PP, label="DGSEM-SC-PP")
    Makie.lines!(axes[1, 1], k1D[p], 1.5e8*k1D[p] .^ (-7/3), label=L"k^{-7/3}", linestyle=:dash) #, title="Energy spectra, T = $(sol_Gauss.t[time_index])")    
    axislegend(axes[1, 1], position=:lb)
    Makie.ylims!(axes[1, 1], (1e-5, 1e9))
    Label(fig[1, 1, Bottom()], L"$N = 3$ mesh of $64^2$ elements", padding=(0, 0, 0, 50), textsize=32)

    @load "data/sol_Gauss_p7_32_cells.jld2" sol_Gauss
    @load "data/sol_DGSEM_SC_PP_p7_32_cells.jld2" sol_DGSEM_SC_PP
    filename = "data/sol_ascii_N7_32elems.dat"
    polydeg, total_cells, x, y, rho, u, v, p = read_fluxo_sol(filename)

    Ek_Gauss, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(sol_Gauss.u[end], sol_Gauss.prob.p)[1:4]...)
    Ek_DGSEM_SC_PP, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(sol_DGSEM_SC_PP.u[end], sol_DGSEM_SC_PP.prob.p)[1:4]...)
    Ek_subcell, k1D = compute_2D_energy_spectrum(convert_to_Cartesian_arrays(polydeg, total_cells, x, y, rho, u, v, p)[1:4]...)
    p = sortperm(k1D)

    Makie.scatter!(axes[1, 2], k1D, Ek_Gauss, label="Gauss")
    Makie.scatter!(axes[1, 2], k1D, Ek_subcell, label="DGSEM-Subcell")
    Makie.scatter!(axes[1, 2], k1D, Ek_DGSEM_SC_PP, label="DGSEM-SC-PP")
    Makie.lines!(axes[1, 2], k1D[p], 1.5e8*k1D[p] .^ (-7/3), label=L"k^{-7/3}", linestyle=:dash) #, title="Energy spectra, T = $(sol_Gauss.t[time_index])")    
    axislegend(axes[1, 2], position=:lb)
    Makie.ylims!(axes[1, 2], (1e-5, 1e9))
    # rowsize!(fig.layout, 1, Aspect(2, 1))
    Label(fig[1, 2, Bottom()], L"$N = 7$ mesh of $32^2$ elements", padding=(0, 0, 0, 50), textsize=32)

    save("spectra.png", fig.scene)    
end