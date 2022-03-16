using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Quad(), approximation_type = GaussSBP(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations2D(1.4)

# Same as the default Trixi example, but v2 is a non-symmetric perturbation
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
initial_condition = initial_condition_khi_nonsymmetric_perturbation

mesh = DGMultiMesh(dg, cells_per_dimension=(32, 32), periodicity=true)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback)

###############################################################################
# run the simulation

tsave = LinRange(tspan..., 100)
sol = solve(ode, RDPK3SpFSAL49(), abstol = 1.0e-7, reltol = 1.0e-7,
            dt = 1e-4,  save_everystep = false, saveat = tsave, callback = callbacks)

summary_callback() # print the timer summary
