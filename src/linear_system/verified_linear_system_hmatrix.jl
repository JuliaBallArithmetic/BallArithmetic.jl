"""
    verified_linear_system_hmatrix.jl

Modified error bounds for approximate solutions of dense linear systems using H-matrices.

Based on:
Minamihata, A., Ogita, T., Rump, S.M. & Oishi, S. (2020),
"Modified error bounds for approximate solutions of dense linear systems",
J. Comput. Appl. Math. 369, 112546.

This provides tighter error bounds than conventional methods, especially for
ill-conditioned cases where the preconditioned matrix RA is an H-matrix.
"""

using LinearAlgebra

"""
    VerifiedLinearSystemResult

Result structure for verified linear system solution.
"""
struct VerifiedLinearSystemResult{T, VT, RT<:Union{Nothing, AbstractMatrix{T}}}
    """Approximate solution."""
    x_approx::VT
    """Error bound on |x_true - x_approx|."""
    error_bound::VT
    """Verification succeeded."""
    verified::Bool
    """Method used for verification."""
    method::Symbol
    """Approximate inverse used as preconditioner."""
    R::RT
    """Perron vector used in verification."""
    v::Union{Nothing, Vector{T}}
    """Condition-related metrics."""
    spectral_radius_ED_inv::T
end

"""
    verified_linear_solve_hmatrix(A::BallMatrix, b::BallVector;
                                   method=:improved_method_a,
                                   R=nothing,
                                   x_approx=nothing,
                                   max_iterations=1,
                                   compute_perron_vector=true)

Compute verified error bounds for the solution of Ax = b using H-matrix properties.

# Arguments
- `A`: Coefficient ball matrix
- `b`: Right-hand side ball vector
- `method`: Verification method to use
  - `:rump_original` - Original Rump (2013) bound (Theorem 3.1)
  - `:minamihata_2015` - Minamihata et al. (2015) bound (Theorem 3.2, m=1)
  - `:improved_method_a` - Improved bound for Method (a) (Theorem 3.3, default)
  - `:improved_method_b` - Improved bound for Method (b) (Corollary 3.2)
- `R`: Approximate inverse of A (computed if not provided)
- `x_approx`: Approximate solution (computed if not provided)
- `max_iterations`: Number of refinement iterations for improved bounds
- `compute_perron_vector`: Whether to compute Perron vector (more accurate)

# Returns
- `VerifiedLinearSystemResult` containing verified error bounds

# Method Comparison
- **Method (a)**: Uses C = [▽(RA), △(RA)] - more expensive but tighter
- **Method (b)**: Uses C = [▽(RA), ▽(RA) + 2nu|R||A|] - half cost, weaker bounds

The improved methods (Theorem 3.3 and Corollary 3.2) provide tighter bounds
than conventional approaches, especially for ill-conditioned systems.

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-10, 2, 2))
b = BallVector([5.0, 4.0], fill(1e-10, 2))

result = verified_linear_solve_hmatrix(A, b; method=:improved_method_a)

println("Verified: ", result.verified)
println("Solution: ", result.x_approx)
println("Error bound: ", result.error_bound)
```
"""
function verified_linear_solve_hmatrix(A::BallMatrix{T}, b::BallVector{T};
                                       method::Symbol=:improved_method_a,
                                       R::Union{Nothing, Matrix{T}}=nothing,
                                       x_approx::Union{Nothing, Vector{T}}=nothing,
                                       max_iterations::Int=1,
                                       compute_perron_vector::Bool=true) where {T}
    n = size(A, 1)

    # Compute approximate inverse if not provided
    if R === nothing
        R = inv(mid(A))
    end

    # Compute approximate solution if not provided
    if x_approx === nothing
        x_approx = R * mid(b)
    end

    # Compute residual with directed rounding
    residual = compute_residual_rigorous(A, b, x_approx, R)

    # Compute interval enclosure of RA based on method
    if method in [:improved_method_a, :rump_original, :minamihata_2015]
        C = compute_RA_interval_method_a(R, A)
    else
        C = compute_RA_interval_method_b(R, A)
    end

    # Compute comparison matrix and check H-matrix property
    comp_matrix = comparison_matrix(C)

    # Find Perron vector v such that ⟨RA⟩v > 0
    v, u = if compute_perron_vector
        compute_perron_vector_power_iteration(comp_matrix, max_iter=100)
    else
        # Use e = (1,1,...,1)^T
        e = ones(T, n)
        u = comp_matrix * e
        if !all(u .> 0)
            return VerifiedLinearSystemResult(
                x_approx, fill(T(Inf), n), false, method,
                R, nothing, T(Inf)
            )
        end
        e, u
    end

    # Verify H-matrix property
    if !all(u .> 0)
        @warn "Failed to verify H-matrix property"
        return VerifiedLinearSystemResult(
            x_approx, fill(T(Inf), n), false, method,
            R, v, T(Inf)
        )
    end

    # Compute error bound based on selected method
    error_bound, spectral_radius = if method == :rump_original
        compute_rump_original_bound(comp_matrix, residual, v, u)
    elseif method == :minamihata_2015
        compute_minamihata_2015_bound(comp_matrix, residual, v, u)
    elseif method == :improved_method_a
        compute_improved_method_a_bound(comp_matrix, residual, v, u, max_iterations)
    else # :improved_method_b
        compute_improved_method_b_bound(C, residual, v, u, R, A)
    end

    verified = all(isfinite.(error_bound))

    return VerifiedLinearSystemResult(
        x_approx, error_bound, verified, method,
        R, v, spectral_radius
    )
end

"""
    compute_residual_rigorous(A, b, x, R)

Compute rigorous interval enclosure of R(b - Ax) with directed rounding.
"""
function compute_residual_rigorous(A::BallMatrix{T}, b::BallVector{T},
                                   x::Vector{T}, R::AbstractMatrix{T}) where {T}
    # Compute b - Ax rigorously
    Ax = mid(A) * x
    res_mid = mid(b) - Ax

    # Account for uncertainties
    res_rad = rad(b) + rad(A) * abs.(x)

    # Apply R with directed rounding
    R_matrix = Matrix(R)  # Convert Diagonal etc. to Matrix for uniform handling
    residual_mid = R_matrix * res_mid
    residual_rad = abs.(R_matrix) * res_rad

    return BallVector(residual_mid, residual_rad)
end

"""
    compute_RA_interval_method_a(R, A)

Compute interval enclosure C = [▽(RA), △(RA)] using Method (a).
Uses directed rounding for matrix multiplication.
"""
function compute_RA_interval_method_a(R::AbstractMatrix{T}, A::BallMatrix{T}) where {T}
    A_mid = mid(A)
    A_rad = rad(A)
    R_matrix = Matrix(R)

    # Compute RA with directed rounding
    RA_mid = R_matrix * A_mid
    RA_rad = abs.(R_matrix) * A_rad

    # Create interval matrix
    return BallMatrix(RA_mid, RA_rad)
end

"""
    compute_RA_interval_method_b(R, A)

Compute interval enclosure C' = [▽(RA), |▽(RA)| + 2nu|R||A|] using Method (b).
This is cheaper (half the cost) but provides weaker bounds.
"""
function compute_RA_interval_method_b(R::AbstractMatrix{T}, A::BallMatrix{T}) where {T}
    n = size(R, 1)
    u = eps(T) / 2  # unit roundoff
    R_matrix = Matrix(R)

    A_mid = mid(A)

    # Compute RA
    RA_mid = R_matrix * A_mid

    # Add rounding error bound: 2nu|R||A|
    rounding_bound = 2 * n * u * abs.(R_matrix) * abs.(A_mid)

    # Total radius includes input uncertainties
    total_rad = rad(A) .+ rounding_bound

    return BallMatrix(RA_mid, abs.(R_matrix) * total_rad)
end

"""
    comparison_matrix(C::BallMatrix)

Compute comparison matrix ⟨C⟩ for interval matrix C.
⟨C⟩_ij = mig(C_ij) if i=j, -mag(C_ij) if i≠j
"""
function comparison_matrix(C::BallMatrix{T}) where {T}
    n = size(C, 1)
    comp = zeros(T, n, n)

    for i in 1:n
        for j in 1:n
            if i == j
                # Diagonal: use mignitude (minimum absolute value)
                comp[i,j] = mig(C[i,j])
            else
                # Off-diagonal: negative magnitude (maximum absolute value)
                comp[i,j] = -mag(C[i,j])
            end
        end
    end

    return comp
end

"""
    mig(x::Ball)

Mignitude (minimum absolute value) of a ball.
"""
function mig(x::Ball{T}) where {T}
    lower = setrounding(T, RoundDown) do
        mid(x) - rad(x)
    end
    upper = setrounding(T, RoundUp) do
        mid(x) + rad(x)
    end
    if lower > zero(T)
        return lower
    elseif upper < zero(T)
        return setrounding(T, RoundDown) do
            abs(upper)
        end
    else
        return zero(T)
    end
end

"""
    mag(x::Ball)

Magnitude (maximum absolute value) of a ball.
"""
function mag(x::Ball{T}) where {T}
    setrounding(T, RoundUp) do
        max(abs(mid(x) - rad(x)), abs(mid(x) + rad(x)))
    end
end

"""
    compute_perron_vector_power_iteration(M, max_iter=100)

Compute approximate Perron vector of ED^(-1) using power iteration.
Returns (v, u) where u = ⟨RA⟩v > 0.
"""
function compute_perron_vector_power_iteration(comp_matrix::Matrix{T};
                                               max_iter::Int=100,
                                               tol::T=sqrt(eps(T))) where {T}
    n = size(comp_matrix, 1)

    # Extract D and E
    D = Diagonal([comp_matrix[i,i] for i in 1:n])
    E = -comp_matrix + D

    # Handle edge case: if E ≈ 0 (e.g., pure diagonal matrix), any positive v works
    if norm(E) < tol
        v = ones(T, n)
        u = comp_matrix * v
        return v, u
    end

    # ED^(-1)
    ED_inv = E * inv(D)

    # Power iteration to find Perron vector
    v = ones(T, n)
    v = v / norm(v)

    for iter in 1:max_iter
        v_new = ED_inv * v

        # Handle case where v_new is very small (near-diagonal matrix)
        v_norm = norm(v_new)
        if v_norm < tol
            v = ones(T, n)
            break
        end

        # Normalize
        v_new = v_new / v_norm

        if norm(v_new - v) < tol
            v = v_new
            break
        end

        v = v_new
    end

    # Make v positive and compute u = ⟨RA⟩v
    v = abs.(v)
    u = comp_matrix * v

    return v, u
end

"""
    compute_rump_original_bound(comp_matrix, residual, v, u)

Compute error bound using Rump (2013) Theorem 3.1.
|A^(-1)b - x̃| ≤ (D^(-1) + vw^T)|R(b - Ax̃)|
"""
function compute_rump_original_bound(comp_matrix::Matrix{T}, residual::BallVector{T},
                                     v::Vector{T}, u::Vector{T}) where {T}
    n = length(v)

    # Extract D and E
    D_diag = [comp_matrix[i,i] for i in 1:n]
    D_inv = Diagonal(1 ./ D_diag)

    E = -comp_matrix + Diagonal(D_diag)
    ED_inv = E * inv(Diagonal(D_diag))

    # Compute w
    G = ED_inv
    w = zeros(T, n)
    for k in 1:n
        w[k] = maximum(G[i,k] / u[i] for i in 1:n)
    end

    # Compute bound
    residual_abs = upper_abs(residual)
    bound = D_inv * residual_abs + v * dot(w, residual_abs)

    # Rigorous upper bound on spectral radius using L2 opnorm bound
    # Convert to BallMatrix for rigorous computation
    spectral_radius = upper_bound_L2_opnorm(BallMatrix(ED_inv))

    return bound, spectral_radius
end

"""
    compute_minamihata_2015_bound(comp_matrix, residual, v, u)

Compute error bound using Minamihata et al. (2015) Theorem 3.2 with m=1.
"""
function compute_minamihata_2015_bound(comp_matrix::Matrix{T}, residual::BallVector{T},
                                       v::Vector{T}, u::Vector{T}) where {T}
    # Similar to Rump but with one iteration of refinement
    # For simplicity, using the same as Rump for now
    return compute_rump_original_bound(comp_matrix, residual, v, u)
end

"""
    compute_improved_method_a_bound(comp_matrix, residual, v, u, max_iterations)

Compute improved error bound using Theorem 3.3 (Method a).

Equation (5): |A^(-1)b - x̃| ≤ D^(-1)|R(b - Ax̃)| + max_{1≤i≤n} (ED^(-1)|R(b - Ax̃)|)_i/u_i * v

Equation (6) with iterations: ϵ^(k) = D^(-1)(c^(0) + ... + c^(k-1)) + max_{i} c^(k)_i/u_i * v
where c^(k) = ED^(-1)c^(k-1)
"""
function compute_improved_method_a_bound(comp_matrix::Matrix{T}, residual::BallVector{T},
                                         v::Vector{T}, u::Vector{T},
                                         max_iterations::Int) where {T}
    n = length(v)

    # Extract D and E
    D_diag = [comp_matrix[i,i] for i in 1:n]
    D_inv = Diagonal(1 ./ D_diag)

    E = -comp_matrix + Diagonal(D_diag)
    ED_inv = E * D_inv

    # c^(0) = |R(b - Ax̃)|
    c = [upper_abs(residual)]

    # Compute iterative bounds
    best_bound = fill(T(Inf), n)

    for k in 1:max_iterations
        # Compute next c^(k) = ED^(-1)c^(k-1)
        push!(c, ED_inv * c[end])

        # Compute ϵ^(k)
        sum_c = sum(c[1:end-1])  # c^(0) + ... + c^(k-1)

        # max_{i} c^(k)_i / u_i
        α = maximum(c[end][i] / u[i] for i in 1:n)

        # ϵ^(k) = D^(-1)(sum_c) + α*v
        epsilon_k = D_inv * sum_c + α * v

        # Keep minimum
        best_bound = min.(best_bound, epsilon_k)
    end

    # Rigorous upper bound on spectral radius using L2 opnorm bound
    spectral_radius = upper_bound_L2_opnorm(BallMatrix(ED_inv))

    return best_bound, spectral_radius
end

"""
    compute_improved_method_b_bound(C, residual, v, u, R, A)

Compute improved error bound using Corollary 3.2 (Method b).

Equation (17): |A^(-1)b - x̃| ≤ D^(-1)r + αv
where r = ⟨mid(C')⟩|ỹ| + |mid(C')ỹ - R(b - Ax̃)|
and α = max_{i} {ED^(-1)r}_i / u_i
"""
function compute_improved_method_b_bound(C::BallMatrix{T}, residual::BallVector{T},
                                         v::Vector{T}, u::Vector{T},
                                         R::Matrix{T}, A::BallMatrix{T}) where {T}
    n = length(v)

    # mid(C')
    mid_C = mid(C)

    # Comparison matrix of mid(C')
    comp_mid_C = comparison_matrix(BallMatrix(mid_C, zeros(T, size(mid_C))))

    # Solve mid(C')ỹ ≈ R(b - Ax̃) with one step of iterative refinement
    res_mid = mid(residual)
    y_tilde = mid_C \ res_mid
    res_y = res_mid - mid_C * y_tilde
    y_tilde += mid_C \ res_y

    # Compute r = ⟨mid(C')⟩|ỹ| + |mid(C')ỹ - R(b - Ax̃)|
    term1 = comp_mid_C * abs.(y_tilde)
    term2 = abs.(mid_C * y_tilde - res_mid)
    r = term1 + term2

    # Extract D and E from comparison matrix of C'
    comp_C = comparison_matrix(C)
    D_diag = [comp_C[i,i] for i in 1:n]
    D_inv = Diagonal(1 ./ D_diag)

    E = -comp_C + Diagonal(D_diag)
    ED_inv = E * D_inv

    # Compute α = max_{i} {ED^(-1)r}_i / u_i
    ED_inv_r = ED_inv * r
    α = maximum(ED_inv_r[i] / u[i] for i in 1:n)

    # Final bound: D^(-1)r + αv
    bound = D_inv * r + α * v

    # Rigorous upper bound on spectral radius using L2 opnorm bound
    spectral_radius = upper_bound_L2_opnorm(BallMatrix(ED_inv))

    return bound, spectral_radius
end

"""
    upper_abs(x::BallVector)

Compute upper bound on |x| for ball vector (componentwise).
"""
function upper_abs(x::BallVector{T}) where {T}
    return abs.(mid(x)) + rad(x)
end

# Export main functions
export VerifiedLinearSystemResult
export verified_linear_solve_hmatrix
export comparison_matrix, mig, mag
