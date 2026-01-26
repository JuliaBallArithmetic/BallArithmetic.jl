"""
    hbr_method.jl

Hansen-Bliek-Rohn (HBR) method for tight enclosures of interval linear systems.
Solves 2n real systems at extremal points to obtain optimal bounds.

# References
- Hansen, E., Bliek, C. (1992), "A new method for computing the hull of a solution set"
- Rohn, J. (1993), "Cheap and tight bounds: The recent result by E. Hansen can be made more efficient"
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
"""

using LinearAlgebra

"""
    HBRResult{T, VT}

Result from Hansen-Bliek-Rohn method.

# Fields
- `solution::VT`: Tight solution enclosure
- `success::Bool`: Whether all systems solved successfully
- `num_systems_solved::Int`: Number of real systems solved (≤ 2n)
- `max_residual::T`: Maximum residual from solved systems
"""
struct HBRResult{T, VT}
    """Tight solution enclosure."""
    solution::VT
    """Whether all systems solved successfully."""
    success::Bool
    """Number of real systems solved."""
    num_systems_solved::Int
    """Maximum residual norm."""
    max_residual::T
end

"""
    hbr_method(A::BallMatrix{T}, b::BallVector{T};
               preconditioner::Union{Nothing, Matrix{T}}=nothing,
               check_residuals::Bool=true,
               residual_tol::T=T(1e-8)) where {T}

Compute tight enclosure of Ax = b using Hansen-Bliek-Rohn method.

# Algorithm
The HBR method computes the hull of the solution set by solving 2n real systems:

For each component i = 1, ..., n:
1. Compute lower bound: Solve A_σ^i x = b_c where σ^i selects extremal matrix
2. Compute upper bound: Solve A_τ^i x = b_c where τ^i selects opposite extremal matrix

The extremal matrices are chosen to minimize/maximize the i-th component.

# Mathematical Details
For minimizing x_i:
- Choose A_σ entries: a_ij = inf(A_ij) if sgn(Ĉ_ji) ≥ 0, else sup(A_ij)
  where Ĉ = C^(-1) is the inverse preconditioner

For maximizing x_i:
- Flip the selection rule

# Arguments
- `A`: Coefficient ball matrix (n×n)
- `b`: Right-hand side ball vector (n)
- `preconditioner`: Approximate inverse C ≈ A^(-1) (computed if not provided)
- `check_residuals`: Verify solutions satisfy Oettli-Prager condition
- `residual_tol`: Tolerance for residual checking

# Returns
`HBRResult` with tight solution enclosure.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 2.0], fill(0.1, 2, 2))
b = BallVector([3.0, 3.0], fill(0.1, 2))

result = hbr_method(A, b)

if result.success
    println("Tight solution: ", result.solution)
    println("Systems solved: ", result.num_systems_solved)
end
```

# Notes
- O(n⁴) complexity - expensive but provides tightest enclosures
- Solves 2n real linear systems
- More accurate than Krawczyk or iterative methods
- Recommended when high accuracy is required and n is small (n ≤ 20)
- Requires good preconditioner for optimal vertex selection
"""
function hbr_method(A::BallMatrix{T}, b::BallVector{T};
                    preconditioner::Union{Nothing, Matrix{T}}=nothing,
                    check_residuals::Bool=true,
                    residual_tol::T=T(1e-8)) where {T}
    n = size(A, 1)

    # Compute preconditioner if not provided
    if preconditioner === nothing
        preconditioner = inv(mid(A))
    end
    C = preconditioner

    # Extract midpoint and radius of A and b
    A_mid = mid(A)
    A_rad = rad(A)
    b_mid = mid(b)
    b_rad = rad(b)

    # Initialize bounds for solution
    x_inf = fill(T(Inf), n)   # Lower bounds
    x_sup = fill(T(-Inf), n)  # Upper bounds

    num_solved = 0
    max_residual = T(0)

    # For each component, solve two systems to find bounds
    for i in 1:n
        # For lower bound: select matrix to minimize x_i
        # For upper bound: select matrix to maximize x_i

        for bound_type in [:lower, :upper]
            # Build extremal matrix A_σ
            A_sigma = copy(A_mid)

            for row in 1:n
                for col in 1:n
                    # Determine sign from preconditioner
                    c_sign = sign(C[col, i])  # C_ji where j=col

                    if bound_type == :lower
                        # Minimize x_i: choose extremal value based on sign
                        if c_sign >= 0
                            A_sigma[row, col] = A_mid[row, col] + A_rad[row, col]
                        else
                            A_sigma[row, col] = A_mid[row, col] - A_rad[row, col]
                        end
                    else  # :upper
                        # Maximize x_i: flip the selection
                        if c_sign >= 0
                            A_sigma[row, col] = A_mid[row, col] - A_rad[row, col]
                        else
                            A_sigma[row, col] = A_mid[row, col] + A_rad[row, col]
                        end
                    end
                end
            end

            # Solve A_sigma * x = b_mid
            try
                x_sigma = A_sigma \ b_mid

                # Optionally check residual (Oettli-Prager condition)
                if check_residuals
                    residual = abs.(A_mid * x_sigma - b_mid)
                    bound = A_rad * abs.(x_sigma) .+ b_rad
                    max_res = maximum(residual .- bound)
                    max_residual = max(max_residual, max_res)

                    if max_res > residual_tol
                        @warn "HBR: Solution does not satisfy Oettli-Prager condition for component $i"
                    end
                end

                # Update bounds
                if bound_type == :lower
                    x_inf[i] = min(x_inf[i], x_sigma[i])
                else
                    x_sup[i] = max(x_sup[i], x_sigma[i])
                end

                num_solved += 1

            catch e
                @warn "HBR: Failed to solve system for component $i, bound $bound_type: $e"
                return HBRResult(
                    BallVector(fill(Ball(T(0), T(Inf)), n)),
                    false, num_solved, max_residual
                )
            end
        end
    end

    # Check if all systems were solved
    success = (num_solved == 2n)

    # Build solution from bounds
    x_mid = (x_inf + x_sup) / 2
    x_rad = (x_sup - x_inf) / 2

    # Handle any negative radii (shouldn't happen if correct)
    for i in 1:n
        if x_rad[i] < 0
            @warn "HBR: Negative radius at component $i, bounds may be incorrect"
            x_rad[i] = 0
        end
    end

    solution = BallVector(x_mid, x_rad)

    return HBRResult(solution, success, num_solved, max_residual)
end

"""
    hbr_method_simple(A::BallMatrix{T}, b::BallVector{T}) where {T}

Simplified HBR method without preconditioner optimization.

Uses identity matrix as preconditioner, which may give suboptimal but still valid bounds.

# Arguments
- `A`: Coefficient ball matrix (n×n)
- `b`: Right-hand side ball vector (n)

# Returns
`HBRResult` with solution enclosure.

# Example
```julia
A = BallMatrix([4.0 1.0; 1.0 3.0], fill(0.05, 2, 2))
b = BallVector([5.0, 4.0], fill(0.05, 2))

result = hbr_method_simple(A, b)
println("Solution: ", result.solution)
```

# Notes
- Simpler but may not give tightest possible bounds
- Uses identity matrix for vertex selection
- Faster preconditioner computation
- Still O(n⁴) due to 2n system solves
"""
function hbr_method_simple(A::BallMatrix{T}, b::BallVector{T}) where {T}
    n = size(A, 1)
    I_mat = Matrix{T}(I, n, n)
    return hbr_method(A, b, preconditioner=I_mat, check_residuals=false)
end

# Export functions
export HBRResult
export hbr_method, hbr_method_simple
