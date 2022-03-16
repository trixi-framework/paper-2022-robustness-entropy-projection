using DelimitedFiles

function read_fluxo_sol(filename)
    f = readdlm(filename)
    polydeg, total_cells = f[1, 1:2]
    x, y, rho, u, v, p = ntuple(_ -> zeros((polydeg + 1)^2, total_cells), 6)
    sk = 1
    sk_u = 1
    for i in 2:size(f, 1)
        if mod(i, 2) == 0        
            x[sk] = f[i, 1]
            y[sk] = f[i, 2]
            sk += 1
        else
            rho[sk_u], u[sk_u], v[sk_u], p[sk_u] = f[i, 1:4]
            sk_u += 1
        end
    end
    return polydeg, total_cells, x, y, rho, u, v, p
end

# specialize for FLUXO outputs
function convert_to_Cartesian_arrays(polydeg, total_cells, x, y, rho, u, v, p)    

    # This scales the node positions to create an equispaced grid of nodes which avoids
    # boundary evaluations. The solution sampled at these points can be used to compute
    # the FFT and the turbulent energy spectra
    h = 1 / (polydeg + 1)
    equispaced_fft_nodes_1D = LinRange(-1 + h, 1 - h, polydeg + 1)
    LGL_nodes = Trixi.StartUpDG.gauss_lobatto_quad(0, 0, polydeg)[1]
    interp_matrix_fft_1D = Trixi.polynomial_interpolation_matrix(LGL_nodes, equispaced_fft_nodes_1D)
    interp_matrix_fft_nodes = kron(interp_matrix_fft_1D, interp_matrix_fft_1D)

    # interpolate solution to FFT points
    x, y, rho, u, v, p = map(u -> interp_matrix_fft_nodes * u, (x, y, rho, u, v, p))

    cells_per_dimension = Int(sqrt(total_cells)) # assume uniform Cartesian mesh
    rho_cartesian = convert_dg2D_array_to_Cartesian(x, y, rho, polydeg, cells_per_dimension)
    u_cartesian   = convert_dg2D_array_to_Cartesian(x, y, u, polydeg, cells_per_dimension)
    v_cartesian   = convert_dg2D_array_to_Cartesian(x, y, v, polydeg, cells_per_dimension)
    p_cartesian   = convert_dg2D_array_to_Cartesian(x, y, p, polydeg, cells_per_dimension)
    return rho_cartesian, u_cartesian, v_cartesian, p_cartesian, x, y
end

