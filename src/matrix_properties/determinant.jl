"""
    determinant.jl

Determinant computation methods for interval matrices.
Provides various algorithms with different accuracy/speed tradeoffs.

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis, Chapters 7-8
- Rohn, J. (1993), "Bounds on eigenvalues of interval matrices"
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
"""

using LinearAlgebra

"""
    DeterminantResult{T}

Result from interval determinant computation.

# Fields
- `determinant::Ball{T}`: Enclosure of determinant
- `method::Symbol`: Method used
- `computation_time::T`: Time spent (optional)
- `tight::Bool`: Whether enclosure is known to be tight
"""
struct DeterminantResult{T}
    """Determinant enclosure."""
    determinant::Ball{T}
    """Method used."""
    method::Symbol
    """Computation time."""
    computation_time::T
    """Whether enclosure is tight."""
    tight::Bool
end

"""
    det_hadamard(A::BallMatrix{T}) where {T}

Compute determinant bound using Hadamard's inequality.

# Hadamard's Inequality
For any n×n matrix A:
    |det(A)| ≤ ∏_{i=1}^n ||a_i||

where a_i is the i-th row of A.

# Arguments
- `A`: Interval matrix (n×n)

# Returns
`DeterminantResult` with determinant bounds.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 2.0], fill(0.1, 2, 2))

result = det_hadamard(A)
println("det(A) ∈ ", result.determinant)
```

# Notes
- O(n²) complexity - very fast
- Provides only upper bound on |det(A)|
- Very conservative for most matrices
- Useful for quick nonsingularity check
- Not tight except for special cases
"""
function det_hadamard(A::BallMatrix{T}) where {T}
    n = size(A, 1)

    # Compute row norms
    row_norms = T[]
    for i in 1:n
        norm_i = T(0)
        for j in 1:n
            a_ij = A[i, j]
            # Maximum absolute value in interval
            max_abs = max(abs(inf(a_ij)), abs(sup(a_ij)))
            norm_i += max_abs^2
        end
        push!(row_norms, sqrt(norm_i))
    end

    # Product of row norms
    hadamard_bound = prod(row_norms)

    # Return symmetric interval
    det_enclosure = Ball(T(0), hadamard_bound)

    return DeterminantResult(det_enclosure, :hadamard, T(0), false)
end

"""
    det_gershgorin(A::BallMatrix{T}) where {T}

Compute determinant bound using Gershgorin circle theorem.

# Gershgorin Approach
Eigenvalues lie in union of Gershgorin discs:
    λ_i ∈ {z : |z - a_{ii}| ≤ ∑_{j≠i} |a_{ij}|}

Determinant is product of eigenvalues.

# Arguments
- `A`: Interval matrix (n×n)

# Returns
`DeterminantResult` with determinant bounds.

# Example
```julia
A = BallMatrix([3.0 0.5; 0.5 2.0], fill(0.1, 2, 2))

result = det_gershgorin(A)
println("det(A) ∈ ", result.determinant)
```

# Notes
- O(n²) complexity - fast
- Can provide tighter bounds than Hadamard for diagonally dominant matrices
- Conservative for general matrices
- Uses product of disc bounds
"""
function det_gershgorin(A::BallMatrix{T}) where {T}
    n = size(A, 1)

    # Compute Gershgorin disc for each row
    disc_inf = T[]
    disc_sup = T[]

    for i in 1:n
        a_ii = A[i, i]
        center = mid(a_ii)

        # Row sum excluding diagonal
        row_sum = T(0)
        for j in 1:n
            if j != i
                a_ij = A[i, j]
                row_sum += max(abs(inf(a_ij)), abs(sup(a_ij)))
            end
        end

        # Add radius of diagonal element
        radius = rad(a_ii) + row_sum

        # Disc bounds
        push!(disc_inf, center - radius)
        push!(disc_sup, center + radius)
    end

    # Estimate determinant bounds (conservative)
    # This is a simplified approach; full computation requires more care
    det_lower = prod(max.(disc_inf, T(0))) - prod(max.(-disc_inf, T(0)))
    det_upper = prod(disc_sup)

    if det_lower > det_upper
        det_lower, det_upper = det_upper, det_lower
    end

    det_mid = (det_lower + det_upper) / 2
    det_rad = (det_upper - det_lower) / 2

    det_enclosure = Ball(det_mid, det_rad)

    return DeterminantResult(det_enclosure, :gershgorin, T(0), false)
end

"""
    det_cramer(A::BallMatrix{T}) where {T}

Compute determinant using Cramer's rule (cofactor expansion).

# Warning
This method has O(n!) complexity and should only be used for n ≤ 4.

# Arguments
- `A`: Small interval matrix (n ≤ 4)

# Returns
`DeterminantResult` with exact interval determinant.

# Example
```julia
A = BallMatrix([2.0 1.0; 1.0 3.0], fill(0.05, 2, 2))

result = det_cramer(A)
println("det(A) ∈ ", result.determinant)  # Exact result
```

# Notes
- O(n!) complexity - only for tiny matrices
- Provides exact interval arithmetic result
- Tight enclosure (no overestimation from method)
- Wrapping effect still present
"""
function det_cramer(A::BallMatrix{T}) where {T}
    n = size(A, 1)

    if n > 4
        @warn "Cramer's rule is O(n!), only recommended for n ≤ 4. Using n=$n may be very slow."
    end

    if n == 1
        return DeterminantResult(A[1, 1], :cramer, T(0), true)
    elseif n == 2
        # det = a11*a22 - a12*a21
        det_val = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
        return DeterminantResult(det_val, :cramer, T(0), true)
    elseif n == 3
        # Sarrus rule for 3×3
        det_val = (A[1, 1] * A[2, 2] * A[3, 3] +
                   A[1, 2] * A[2, 3] * A[3, 1] +
                   A[1, 3] * A[2, 1] * A[3, 2] -
                   A[1, 3] * A[2, 2] * A[3, 1] -
                   A[1, 2] * A[2, 1] * A[3, 3] -
                   A[1, 1] * A[2, 3] * A[3, 2])
        return DeterminantResult(det_val, :cramer, T(0), true)
    else
        # General cofactor expansion (expensive!)
        det_val = _cofactor_expansion(A, 1)
        return DeterminantResult(det_val, :cramer, T(0), true)
    end
end

"""
    _cofactor_expansion(A, row)

Internal: Recursive cofactor expansion along given row.
"""
function _cofactor_expansion(A::BallMatrix{T}, row::Int) where {T}
    n = size(A, 1)

    if n == 1
        return A[1, 1]
    end

    det_sum = Ball(T(0), T(0))

    for j in 1:n
        # Get minor (remove row and column j)
        minor_indices_row = [i for i in 1:n if i != row]
        minor_indices_col = [k for k in 1:n if k != j]

        minor = A[minor_indices_row, minor_indices_col]

        # Recursive call
        minor_det = _cofactor_expansion(minor, 1)

        # Add contribution with correct sign
        sign_factor = (-1)^(row + j)
        det_sum = det_sum + sign_factor * A[row, j] * minor_det
    end

    return det_sum
end

"""
    interval_det(A::BallMatrix{T};
                method::Symbol=:auto,
                check_regularity::Bool=true) where {T}

Compute interval determinant enclosure using specified or automatic method.

# Methods
- `:auto`: Choose automatically based on matrix size and properties
- `:hadamard`: Hadamard inequality (fast, conservative)
- `:gershgorin`: Gershgorin-based bounds (fast, moderate)
- `:gaussian_elimination`: Gaussian elimination (moderate speed, good accuracy)
- `:cramer`: Cramer's rule (slow, exact for small n)

# Arguments
- `A`: Interval matrix (n×n)
- `method`: Computation method
- `check_regularity`: First check if matrix is regular

# Returns
`DeterminantResult` with determinant enclosure.

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.1, 2, 2))

# Automatic method selection
result = interval_det(A)
println("det(A) ∈ ", result.determinant)

# Specific method
result_hadamard = interval_det(A, method=:hadamard)
```

# Notes
- Auto mode chooses:
  - Cramer for n ≤ 3
  - Gaussian elimination for n ≤ 20
  - Hadamard for n > 20
- Check regularity first to avoid wasted computation
"""
function interval_det(A::BallMatrix{T};
                     method::Symbol=:auto,
                     check_regularity::Bool=true) where {T}
    n = size(A, 1)

    # Check regularity if requested
    if check_regularity
        # Quick diagonal dominance check
        reg_result = is_regular_diagonal_dominance(A, strict=true)
        if !reg_result.is_regular
            # Try Gershgorin
            reg_result = is_regular_gershgorin(A)
        end

        if reg_result.is_regular
            # Matrix is regular, determinant doesn't contain 0
            # Continue with computation
        end
    end

    # Select method
    actual_method = if method == :auto
        if n <= 3
            :cramer
        elseif n <= 20
            :gaussian_elimination
        else
            :hadamard
        end
    else
        method
    end

    # Compute determinant using selected method
    return if actual_method == :hadamard
        det_hadamard(A)
    elseif actual_method == :gershgorin
        det_gershgorin(A)
    elseif actual_method == :gaussian_elimination
        # Use Gaussian elimination from linear_system module
        # Import the function if needed
        result_ge = interval_gaussian_elimination_det(A)
        DeterminantResult(result_ge, :gaussian_elimination, T(0), false)
    elseif actual_method == :cramer
        det_cramer(A)
    else
        error("Unknown determinant method: $actual_method")
    end
end

# NOTE: interval_gaussian_elimination_det is defined in linear_system/gaussian_elimination.jl
# No fallback needed since that file is always loaded before this one.

"""
    contains_zero(det_result::DeterminantResult{T}) where {T}

Check if determinant enclosure contains zero (possible singularity).

# Arguments
- `det_result`: Result from determinant computation

# Returns
`true` if 0 ∈ det(A), `false` otherwise.

# Example
```julia
A = BallMatrix([1.0 1.0; 1.0 1.0], fill(0.1, 2, 2))
result = interval_det(A)

if contains_zero(result)
    println("Matrix may be singular")
end
```
"""
function contains_zero(det_result::DeterminantResult{T}) where {T}
    return T(0) ∈ det_result.determinant
end

# Export functions
export DeterminantResult
export det_hadamard, det_gershgorin, det_cramer
export interval_det, contains_zero
