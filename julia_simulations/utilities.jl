function catch_blow_up_time(ode; time_stepper=RDPK3SpFSAL49(), abstol=1.0e-7, reltol=1.0e-7, 
                            callbacks=AliveCallback(alive_interval = 100))
    integrator = init(ode, time_stepper, abstol=abstol, reltol=reltol,
                      save_everystep=false, callback=callbacks)
    try
        solve!(integrator)
    catch e
        @warn "Blow-up at time " e
    end
    return integrator.t
end

reinterpret_DGSEM_solution_as_DGMulti(u, semi) = 
    reinterpret_DGSEM_solution_as_DGMulti(u, Trixi.mesh_equations_solver_cache(semi)...)

function reinterpret_DGSEM_solution_as_DGMulti(u, mesh::TreeMesh{NDIMS}, equations, dg::DGSEM, cache) where {NDIMS}
    nvars = nvariables(equations)
    uEltype = eltype(u)
    reshaped_u_raw = reinterpret(reshape, SVector{nvars, uEltype},
                                 reshape(u, nvars, nnodes(dg)^NDIMS, nelements(mesh, dg, cache)))
    reshaped_u = StructArray{SVector{nvars, uEltype}}(ntuple(_ -> zeros(uEltype, size(reshaped_u_raw)), nvars))
    reshaped_u .= reshaped_u_raw

    return reshaped_u
end

function integrate_entropy(sol::Trixi.TrixiODESolution)
    S = zeros(length(sol.t))
    for i in eachindex(sol.t)
        S[i] = integrate_entropy(sol.u[i], sol.prob.p)
    end
    return S
end

integrate_entropy(u, semi) = integrate_entropy(Trixi.wrap_array(u, semi), Trixi.mesh_equations_solver_cache(semi)...)

function integrate_entropy(u, mesh, equations, dg::DGMulti, cache)
    rd = dg.basis

    # Note: this evaluates at the inherent quadrature used by the DGMulti solver,
    # which may not be consistent with Lobatto DGSEM quadrature. However, we have 
    # observed that using Lobatto quadrature can result in negative density/pressure
    # even when the Gauss collocation solver does not crash (since it uses different 
    # points). 
    @unpack Vq = rd
    @unpack wJq = mesh.md

    uq = similar(u, rd.Nq, size(u, 2))
    Trixi.apply_to_each_field(Trixi.mul_by!(Vq), uq, u)
    S = sum(wJq .* entropy.(uq, equations))
    return S
end

using LinearAlgebra: diagm

function integrate_entropy(u, mesh, equations, solver::DGSEM, cache)
    nvars = nvariables(equations)
    reshaped_u = reinterpret(reshape, SVector{nvars, Float64}, u)
    J = 1.0 / nelements(mesh, solver, cache)
    w1D = solver.basis.weights
    wq = vec([w1D[i] * w1D[j] for j in eachnode(solver), i in eachnode(solver)])
    wJq = diagm(wq) * J * ones(nnodes(solver)^2, nelements(mesh, solver, cache))
    S = sum(reshape(wJq, size(reshaped_u)) .* entropy.(reshaped_u, equations))
    return S
end

function integrate_entropy(u, mesh::TreeMesh{3}, equations, solver::DGSEM, cache)
    nvars = nvariables(equations)
    reshaped_u = reinterpret(reshape, SVector{nvars, Float64}, u)
    J = 1.0 / nelements(mesh, solver, cache)
    w1D = solver.basis.weights
    wq = vec([w1D[i] * w1D[j] * w1D[k] for k in eachnode(solver), j in eachnode(solver), i in eachnode(solver)])
    wJq = diagm(wq) * J * ones(nnodes(solver)^3, nelements(mesh, solver, cache))
    S = sum(reshape(wJq, size(reshaped_u)) .* entropy.(reshaped_u, equations))
    return S
end

function compute_entropy_over_time(sol::ODESolution)
    semi = sol.prob.p
    S = zeros(length(sol.u))
    for (i, u) in enumerate(sol.u)
      S[i] = compute_entropy_over_time(u, semi)
    end
    return S
end

function compute_entropy_over_time(u::AbstractVector, semi)
    reshaped_u = reinterpret_DGSEM_solution_as_DGMulti(u, semi)
    return compute_entropy_over_time(reshaped_u, semi)
end

function compute_entropy_over_time(u::AbstractMatrix, semi)
    if typeof(u) <: StructArray
        rho = StructArrays.component(u, 1)
    else
        rho = getindex.(u, 1)
    end
    @unpack equations = semi
    p = pressure.(u, equations)
    S = integrate_entropy(u, Trixi.mesh_equations_solver_cache(semi)...)
    return S
end

## Routines which depend on dimension

# converts a solution array `u` with size (nnodes, cells_per_dimension^2) to a logically 
# Cartesian array. Assumes the mesh is Cartesian.
function convert_dg2D_array_to_Cartesian(x, y, u, polydeg, cells_per_dimension)
    h = 1 / cells_per_dimension
    cell_centroids = LinRange(-1 + h, 1 - h, cells_per_dimension)  
    my_mean(x) = sum(x)/length(x)
  
    num_pts_per_element = polydeg + 1
    Nfft = cells_per_dimension * num_pts_per_element
    u_cartesian = zeros(eltype(u), Nfft, Nfft)
    for e in 1:size(u, 2)
        xc = my_mean(view(x, :, e))
        yc = my_mean(view(y, :, e))
        u_e = reshape(view(u, :, e), polydeg + 1, polydeg + 1)'

        tol = 1e2*eps()
        ex = findfirst(@. abs(cell_centroids - xc) < tol)
        ey = findfirst(@. abs(cell_centroids - yc) < tol)

        row_ids = (1:num_pts_per_element) .+ (ey - 1) * num_pts_per_element
        col_ids = (1:num_pts_per_element) .+ (ex - 1) * num_pts_per_element
        u_cartesian[row_ids, col_ids] .= u_e
    end
    return u_cartesian
end

# Given an ODESolution, returns Cartesian arrays representing the primitive solution fields.
convert_to_Cartesian_arrays(sol) = 
    convert_to_Cartesian_arrays(sol.u[end], Trixi.mesh_equations_solver_cache(sol.prob.p)...)

convert_to_Cartesian_arrays(u, semi) = 
    convert_to_Cartesian_arrays(u, Trixi.mesh_equations_solver_cache(semi)...)

function convert_to_Cartesian_arrays(u, mesh, equations::CompressibleEulerEquations2D, dg::DGMulti, cache)
    rd = dg.basis
    polydeg = rd.N
    
    # This scales the node positions to create an equispaced grid of nodes which avoids
    # boundary evaluations. The solution sampled at these points can be used to compute
    # the FFT and the turbulent energy spectra
    equispaced_fft_nodes = map(x -> x * (1 - 1 / (polydeg + 1)), StartUpDG.equi_nodes(Quad(), polydeg))
    interp_matrix_fft_nodes = StartUpDG.vandermonde(Quad(), polydeg, equispaced_fft_nodes...) / rd.VDM

    # interpolate to equispaced FFT nodes
    primitive_variables = StructArrays.components(cons2prim.(u, equations))
    rho, u, v, p = map(u -> interp_matrix_fft_nodes * u, primitive_variables)
    x, y = map(u -> interp_matrix_fft_nodes * u, mesh.md.xyz)

    cells_per_dimension = Int(sqrt(mesh.md.num_elements)) # assume uniform Cartesian mesh
    rho_cartesian = convert_dg2D_array_to_Cartesian(x, y, rho, polydeg, cells_per_dimension)
    u_cartesian   = convert_dg2D_array_to_Cartesian(x, y, u, polydeg, cells_per_dimension)
    v_cartesian   = convert_dg2D_array_to_Cartesian(x, y, v, polydeg, cells_per_dimension)
    p_cartesian   = convert_dg2D_array_to_Cartesian(x, y, p, polydeg, cells_per_dimension)
    return rho_cartesian, u_cartesian, v_cartesian, p_cartesian, x, y
end

function convert_to_Cartesian_arrays(u, mesh, equations::CompressibleEulerEquations2D, dg::DGSEM, cache)
    polydeg = Trixi.polydeg(dg)

    # This scales the node positions to create an equispaced grid of nodes which avoids
    # boundary evaluations. The solution sampled at these points can be used to compute
    # the FFT and the turbulent energy spectra
    h = 1 / (polydeg + 1)
    equispaced_fft_nodes_1D = LinRange(-1 + h, 1 - h, polydeg + 1)
    interp_matrix_fft_1D = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, equispaced_fft_nodes_1D)
    interp_matrix_fft_nodes = kron(interp_matrix_fft_1D, interp_matrix_fft_1D)

    # interpolate solution to FFT points
    u_solution = reinterpret_DGSEM_solution_as_DGMulti(u, mesh, equations, dg, cache)
    primitive_variables = StructArrays.components(cons2prim.(u_solution, equations))
    rho, u, v, p = map(u -> interp_matrix_fft_nodes * u, primitive_variables)

    # reshape arrays to have the same format as DGMulti
    x_coordinates = view(cache.elements.node_coordinates, 1, :, :, :)
    y_coordinates = view(cache.elements.node_coordinates, 2, :, :, :)
    x = reshape(x_coordinates, nnodes(dg)^2, nelements(mesh, dg, cache))
    y = reshape(y_coordinates, nnodes(dg)^2, nelements(mesh, dg, cache))
    x, y = map(u -> interp_matrix_fft_nodes * u, (x, y))

    cells_per_dimension = Int(sqrt(nelements(mesh, dg, cache))) # assume uniform Cartesian mesh
    rho_cartesian = convert_dg2D_array_to_Cartesian(x, y, rho, polydeg, cells_per_dimension)
    u_cartesian   = convert_dg2D_array_to_Cartesian(x, y, u, polydeg, cells_per_dimension)
    v_cartesian   = convert_dg2D_array_to_Cartesian(x, y, v, polydeg, cells_per_dimension)
    p_cartesian   = convert_dg2D_array_to_Cartesian(x, y, p, polydeg, cells_per_dimension)
    return rho_cartesian, u_cartesian, v_cartesian, p_cartesian, x, y
end

function compute_2D_energy_spectrum(rho_cartesian, u_cartesian, v_cartesian, p_cartesian)
    uhat = fft(sqrt.(rho_cartesian) .* u_cartesian)
    vhat = fft(sqrt.(rho_cartesian) .* v_cartesian)
    Ek_full = @. 0.5 * (abs(uhat)^2 + abs(vhat)^2)

    Ek = Ek_full[1:end÷2+1, 1:end÷2+1] # remove duplicate modes

    wavenumbers = 1:size(Ek, 1)
    effective_wavenumbers = @. sqrt(wavenumbers^2 + wavenumbers'^2)

    N = length(wavenumbers)
    Ek_1D = zeros(N * (N - 1) ÷ 2 + N) # number of unique wavenumbers = triangular part + diagonal
    effective_wavenumber_1D = zeros(N * (N - 1) ÷ 2 + N)
    sk = 1
    for i in wavenumbers
        for j in wavenumbers[i:end] # use that the effective_wavenumber matrix is symmetric
            kk = sqrt(i^2 + j^2)
            ids = findall(@. kk - 0.5 < effective_wavenumbers < kk + 0.5) # find wavenumbers in LinRange
            effective_wavenumber_1D[sk] = kk
            Ek_1D[sk] = sum(Ek[ids])
            sk += 1
        end
    end
    return Ek_1D, effective_wavenumber_1D
end