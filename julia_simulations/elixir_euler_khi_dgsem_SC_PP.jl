
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

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

# surface_flux = FluxPlusDissipation(flux_ranocha, DissipationLocalLaxFriedrichs())
surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.0025,
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
# volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.0, -1.0)
coordinates_max = ( 1.0,  1.0)
cells_per_dimension = 32
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Int(log2(cells_per_dimension)),
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(alive_interval=100, analysis_interval=analysis_interval)

stepsize_callback = StepsizeCallback(cfl=0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

tsave = LinRange(tspan..., 2)
# use_positivity_limiting = true
# if use_positivity_limiting==true
    limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
                                                variables=(Trixi.density, pressure))
    sol = solve(ode, SSPRK43(limiter!), saveat=tsave, save_everystep=false, callback=callbacks);
# else
#     sol = solve(ode, RDPK3SpFSAL49(), abstol = 1.0e-7, reltol = 1.0e-7,
#                 save_everystep = false, saveat = tsave, callback = callbacks)
# end

summary_callback() # print the timer summary
