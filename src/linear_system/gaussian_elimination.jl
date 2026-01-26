"""
    gaussian_elimination.jl

Interval Gaussian elimination for solving linear systems Ax = b.
Provides direct solution method with singularity detection.

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
- Rump, S.M. (1999), "INTLAB - INTerval LABoratory"
"""

using LinearAlgebra

"""
    GaussianEliminationResult{T, VT}

Result from interval Gaussian elimination.

# Fields
- `solution::VT`: Solution enclosure (or empty if singular)
- `success::Bool`: Whether elimination succeeded
- `singular::Bool`: Whether matrix is provably singular
- `L::Matrix`: Lower triangular factor (if computed)
- `U::BallMatrix`: Upper triangular factor
- `p::Vector{Int}`: Permutation vector (pivoting)
"""
struct GaussianEliminationResult{T, MT, VT}
    """Solution enclosure."""
    solution::VT
    """Whether elimination succeeded."""
    success::Bool
    """Whether matrix is provably singular."""
    singular::Bool
    """Lower triangular factor."""
    L::Matrix{T}
    """Upper triangular factor."""
    U::MT
    """Permutation vector."""
    p::Vector{Int}
end

"""
    interval_gaussian_elimination(A::BallMatrix{T}, b::BallVector{T};
                                  pivoting::Symbol=:partial,
                                  store_factors::Bool=false) where {T}

Solve Ax = b using interval Gaussian elimination with partial pivoting.

# Algorithm
1. Forward elimination with partial pivoting:
   - For each column k, find pivot with maximum magnitude
   - Swap rows if needed
   - Eliminate entries below pivot
   - Check for singular matrix (zero on diagonal)

2. Backward substitution:
   - Solve Ux = y from bottom to top
   - x_i = (y_i - Σ_{j>i} u_{ij}x_j) / u_{ii}

# Arguments
- `A`: Coefficient ball matrix (n×n)
- `b`: Right-hand side ball vector (n)
- `pivoting`: Pivoting strategy (`:partial` or `:none`)
- `store_factors`: Whether to store L and U factors

# Returns
`GaussianEliminationResult` with solution and factorization information.

# Example
```julia
A = BallMatrix([2.0 1.0 1.0; 4.0 3.0 3.0; 8.0 7.0 9.0], fill(1e-10, 3, 3))
b = BallVector([4.0, 10.0, 24.0], fill(1e-10, 3))

result = interval_gaussian_elimination(A, b)

if result.success
    println("Solution: ", result.solution)
elseif result.singular
    println("Matrix is singular")
end
```

# Notes
- O(n³) complexity
- Detects singularity during elimination
- Partial pivoting improves numerical stability
- Can suffer from overestimation (wrapping effect)
- Preconditioning recommended for better accuracy
"""
function interval_gaussian_elimination(A::BallMatrix{T}, b::BallVector{T};
                                        pivoting::Symbol=:partial,
                                        store_factors::Bool=false) where {T}
    n = size(A, 1)

    # Copy A and b (will be modified)
    U = copy(A)
    y = copy(b)

    # Initialize permutation and L factor
    p = collect(1:n)
    L = Matrix{T}(I, n, n)

    # Forward elimination
    for k in 1:(n-1)
        # Partial pivoting: find row with largest magnitude pivot
        if pivoting == :partial
            pivot_row = k
            max_mag = abs(inf(U[k, k]))

            for i in (k+1):n
                mag = abs(inf(U[i, k]))
                if mag > max_mag
                    max_mag = mag
                    pivot_row = i
                end
            end

            # Swap rows if needed
            if pivot_row != k
                # Swap in U
                for j in 1:n
                    U[k, j], U[pivot_row, j] = U[pivot_row, j], U[k, j]
                end
                # Swap in y
                y[k], y[pivot_row] = y[pivot_row], y[k]
                # Swap in L (only lower part)
                if store_factors
                    for j in 1:(k-1)
                        L[k, j], L[pivot_row, j] = L[pivot_row, j], L[k, j]
                    end
                end
                # Record permutation
                p[k], p[pivot_row] = p[pivot_row], p[k]
            end
        end

        # Check for zero pivot (singularity)
        pivot = U[k, k]
        if 0.0 ∈ pivot
            @warn "Gaussian elimination: Zero pivot detected at column $k, matrix may be singular"
            return GaussianEliminationResult(
                BallVector(fill(Ball(T(0), T(Inf)), n)),
                false, true, L, U, p
            )
        end

        # Eliminate entries below pivot
        for i in (k+1):n
            # Compute multiplier
            mult = U[i, k] / U[k, k]

            if store_factors
                L[i, k] = mid(mult)
            end

            # Update row i
            for j in (k+1):n
                U[i, j] = U[i, j] - mult * U[k, j]
            end
            U[i, k] = Ball(T(0), T(0))  # Explicitly zero out

            # Update right-hand side
            y[i] = y[i] - mult * y[k]
        end
    end

    # Check last diagonal element
    if 0.0 ∈ U[n, n]
        @warn "Gaussian elimination: Zero pivot at last position, matrix may be singular"
        return GaussianEliminationResult(
            BallVector(fill(Ball(T(0), T(Inf)), n)),
            false, true, L, U, p
        )
    end

    # Backward substitution
    x = similar(y)

    for i in n:-1:1
        # Compute sum of known terms
        sum_term = y[i]
        for j in (i+1):n
            sum_term = sum_term - U[i, j] * x[j]
        end

        # Solve for x[i]
        x[i] = sum_term / U[i, i]

        # Check for excessive growth
        if rad(x[i]) > 1e10 * (abs(mid(x[i])) + 1)
            @warn "Gaussian elimination: Excessive interval growth at component $i"
        end
    end

    return GaussianEliminationResult(
        x, true, false, L, U, p
    )
end

"""
    interval_gaussian_elimination_det(A::BallMatrix{T}) where {T}

Compute determinant enclosure using Gaussian elimination.

# Algorithm
The determinant is the product of diagonal elements after elimination:
    det(PA) = ∏_{i=1}^n u_{ii} × sign(permutation)

# Arguments
- `A`: Square ball matrix (n×n)

# Returns
Ball enclosure of the determinant.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 2.0], fill(1e-10, 2, 2))
det_enclosure = interval_gaussian_elimination_det(A)
println("det(A) ∈ ", det_enclosure)
```

# Notes
- O(n³) complexity
- Can suffer from overestimation
- Zero in result indicates possible singularity
- Useful for singularity detection
"""
function interval_gaussian_elimination_det(A::BallMatrix{T}) where {T}
    n = size(A, 1)

    # Dummy right-hand side (not needed for determinant)
    b = BallVector(zeros(T, n), zeros(T, n))

    # Perform elimination
    result = interval_gaussian_elimination(A, b, pivoting=:partial, store_factors=false)

    if result.singular
        return Ball(T(0), T(Inf))  # Singular or potentially singular
    end

    # Compute product of diagonal elements
    det_val = result.U[1, 1]
    for i in 2:n
        det_val = det_val * result.U[i, i]
    end

    # Account for permutation sign
    # Count number of swaps
    sign_factor = 1
    p = result.p
    for i in 1:n
        if p[i] != i
            # Find where i ended up and swap back
            j = findfirst(==(i), p[(i+1):end])
            if j !== nothing
                j = j + i
                p[i], p[j] = p[j], p[i]
                sign_factor *= -1
            end
        end
    end

    return det_val * sign_factor
end

"""
    is_regular_gaussian_elimination(A::BallMatrix{T}) where {T}

Test if matrix [A] contains only regular (nonsingular) matrices using Gaussian elimination.

# Algorithm
Perform Gaussian elimination and check if zero appears on diagonal.
If zero does NOT appear, matrix is proven regular.

# Arguments
- `A`: Square ball matrix (n×n)

# Returns
- `true` if matrix is proven regular
- `false` if test is inconclusive

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(1e-10, 2, 2))

if is_regular_gaussian_elimination(A)
    println("Matrix is regular")
else
    println("Cannot determine regularity")
end
```

# Notes
- Sufficient but not necessary test
- O(n³) complexity
- If returns true, regularity is guaranteed
- If returns false, matrix may still be regular
"""
function is_regular_gaussian_elimination(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    b = BallVector(zeros(T, n), zeros(T, n))

    result = interval_gaussian_elimination(A, b, pivoting=:none, store_factors=false)

    return result.success && !result.singular
end

# Export functions
export GaussianEliminationResult
export interval_gaussian_elimination, interval_gaussian_elimination_det
export is_regular_gaussian_elimination
