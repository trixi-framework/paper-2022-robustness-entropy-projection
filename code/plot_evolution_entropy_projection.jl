using Trixi, OrdinaryDiffEq
using DiffEqCallbacks

N = 3
surface_flux = flux_lax_friedrichs

dgsem = DGMulti(polydeg = N, element_type = Quad(), approximation_type = SBP(),
                surface_integral = SurfaceIntegralWeakForm(surface_flux),
                volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

gauss = DGMulti(polydeg = N, element_type = Quad(), approximation_type = GaussSBP(),
                surface_integral = SurfaceIntegralWeakForm(surface_flux),
                volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

# Default Trixi initial condition
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # domain size is [-1,+1]^2
    slope = 15
     # discontinuous function with value 2 for -.5 <= x <= .5 and 0 elsewhere
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + .75 * B # rho ∈ [.5, 2]
    v1 = 0.5 * (B - 1) # v1 ∈ [-.]
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability

equations = CompressibleEulerEquations2D(1.4)

semi_dgsem = SemidiscretizationHyperbolic(DGMultiMesh(dgsem, cells_per_dimension=(32, 32), periodicity=true), 
                                          equations, initial_condition, dgsem)
semi_gauss = SemidiscretizationHyperbolic(DGMultiMesh(gauss, cells_per_dimension=(32, 32), periodicity=true),
                                          equations, initial_condition, gauss)

function compute_entropy_variable_error(u, t, integrator)
    @unpack mesh, equations, cache = integrator.p

    dg = integrator.p.solver

    N = dg.basis.N
    element_type = dg.basis.element_type
    rq, sq, wq = Trixi.StartUpDG.quad_nodes(element_type, N+1)
    Vq = Trixi.StartUpDG.vandermonde(element_type, N, rq, sq) / dg.basis.VDM
    VqPq = Vq * (dg.basis.M \ (Vq' * diagm(wq)))

    @unpack J = mesh.md    

    u_e = StructArray{SVector{4, Float64}}(ntuple(_ -> zeros(size(u, 1)), 4))
    u_q = StructArray{SVector{4, Float64}}(ntuple(_ -> zeros(size(VqPq, 1)), 4))
    projected_entropy_vars = similar(u_q)
    u_diff, v_diff, u_norm, v_norm = ntuple(_->zeros(size(u, 2)), 4)
    for e in 1:size(u, 2)
        # interpolate
        u_e .= view(u, :, e)
        StructArrays.foreachfield(Trixi.mul_by!(Vq), u_q, u_e)

        # map to entropy vars and project
        entropy_vars = cons2entropy.(u_q, equations)        
        StructArrays.foreachfield(Trixi.mul_by!(VqPq), projected_entropy_vars, entropy_vars)
        u_entropy_projection = entropy2cons.(projected_entropy_vars, equations)            

        v_diff_local = projected_entropy_vars - entropy_vars
        u_diff_local = u_q - u_entropy_projection
        
        u_norm[e] = sum(sum(wq .* J[1, e] .* map(x->x.^2, u_q)))
        u_diff[e] = sum(sum(wq .* J[1, e] .* map(x->x.^2, u_diff_local))) 

        v_norm[e] = sum(sum(wq .* J[1, e] .* map(x->x.^2, projected_entropy_vars)))
        v_diff[e] = sum(sum(wq .* J[1, e] .* map(x->x.^2, v_diff_local))) 

        # u_norm[e] = maximum(maximum(map(x->abs.(x), u_q)))
        # u_diff[e] = maximum(maximum(map(x->abs.(x), u_diff_local))) 

        # v_norm[e] = maximum(maximum(map(x->abs.(x), projected_entropy_vars)))
        # v_diff[e] = maximum(maximum(map(x->abs.(x), v_diff_local)))
    end

    rho = getindex.(u, 1)
    p = Trixi.pressure.(u, equations)

    # ||u-ũ||, ||u||, ||v-ṽ||, ||v||, rho_max, rho_min, p_max, p_min
    return sqrt(sum(u_diff)), sqrt(sum(u_norm)), sqrt(sum(v_diff)), sqrt(sum(v_norm)), extrema(rho)..., extrema(p)...
    # return maximum(u_diff)/maximum(u_norm), maximum(v_diff)/maximum(v_norm)
end
  
alive_callback = AliveCallback(alive_interval = 100)

tspan_dgsem = (0.0, 3.6523)
saved_values_dgsem = SavedValues(Float64, NTuple{8, Float64})
saving_callback_dgsem = SavingCallback(compute_entropy_variable_error, saved_values_dgsem, saveat=LinRange(tspan_dgsem..., 1000))

tspan_gauss = (0.0, 7.5)
saved_values_gauss = SavedValues(Float64, NTuple{8, Float64})
saving_callback_gauss = SavingCallback(compute_entropy_variable_error, saved_values_gauss, saveat=LinRange(tspan_gauss..., 1000))

###############################################################################
# run the simulation

sol_dgsem = solve(semidiscretize(semi_dgsem, tspan_dgsem), RDPK3SpFSAL49(), 
                  abstol = 1.0e-7, reltol = 1.0e-7,
                  callback = CallbackSet(alive_callback, saving_callback_dgsem),
                  dt = 1e-4, save_everystep = false)

sol_gauss = solve(semidiscretize(semi_gauss, tspan_gauss), RDPK3SpFSAL49(), 
                  abstol = 1.0e-7, reltol = 1.0e-7,
                  callback = CallbackSet(alive_callback, saving_callback_gauss),
                  dt = 1e-4, save_everystep = false) 

using Plots: Plots
save_index = 5
Plots.plot(saved_values_dgsem.t, getindex.(saved_values_dgsem.saveval, save_index), label="DGSEM", legend=:topleft)
Plots.plot!(saved_values_gauss.t, getindex.(saved_values_gauss.saveval, save_index), label="Gauss", legend=:topleft)

u_diff = getindex.(saved_values_dgsem.saveval, 1) ./ getindex.(saved_values_dgsem.saveval, 2)
v_diff = getindex.(saved_values_dgsem.saveval, 3) ./ getindex.(saved_values_dgsem.saveval, 4)
Plots.plot(saved_values_dgsem.t, u_diff, label="DGSEM", legend=:topleft)

u_diff = getindex.(saved_values_gauss.saveval, 1) ./ getindex.(saved_values_gauss.saveval, 2)
v_diff = getindex.(saved_values_gauss.saveval, 3) ./ getindex.(saved_values_gauss.saveval, 4)
Plots.plot!(saved_values_gauss.t, u_diff, label="Gauss", legend=:topleft)
# Plots.plot!(title="Norm of diff: cons vars and entropy projection")