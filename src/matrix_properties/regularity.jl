"""
    regularity.jl

Regularity testing for interval matrices.
Tests whether all matrices in [A] are nonsingular (regular).

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis, Chapter 11
- Rohn, J. (1989), "Systems of linear interval equations"
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
"""

using LinearAlgebra

"""
    RegularityResult{T}

Result from regularity testing.

# Fields
- `is_regular::Bool`: True if proven regular, false if inconclusive
- `method::Symbol`: Method used for testing
- `certificate::T`: Numerical certificate (e.g., separation value)
- `is_definitive::Bool`: Whether result is definitive or just sufficient condition
"""
struct RegularityResult{T}
    """Whether matrix is proven regular."""
    is_regular::Bool
    """Method used for testing."""
    method::Symbol
    """Numerical certificate."""
    certificate::T
    """Whether result is definitive."""
    is_definitive::Bool
end

"""
    is_regular_sufficient_condition(A::BallMatrix{T}) where {T}

Test regularity using sufficient condition from Theorem 11.12 (Horáček, p. 183).

# Sufficient Condition
Matrix [A] is regular if:
    λ_max(A_Δ^T A_Δ) < λ_min(A_c^T A_c)

where A_c is center and A_Δ is radius matrix.

# Arguments
- `A`: Interval matrix (n×n)

# Returns
`RegularityResult` with test outcome.

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.05, 2, 2))

result = is_regular_sufficient_condition(A)

if result.is_regular
    println("Matrix is proven regular")
    println("Separation: ", result.certificate)
else
    println("Test inconclusive")
end
```

# Notes
- O(n³) complexity (eigenvalue computation)
- Sufficient but not necessary
- If returns true, regularity is guaranteed
- If returns false, matrix may still be regular
- Works well for small radius matrices
"""
function is_regular_sufficient_condition(A::BallMatrix{T}) where {T}
    A_c = mid(A)
    A_Δ = rad(A)

    # Compute A_c^T * A_c
    AtA_c = A_c' * A_c

    # Compute A_Δ^T * A_Δ
    AtA_Δ = A_Δ' * A_Δ

    # Compute eigenvalues
    λ_max_rad = try
        eigvals(Symmetric(AtA_Δ))[end]  # Largest eigenvalue
    catch e
        @warn "Failed to compute eigenvalues of A_Δ^T A_Δ: $e"
        return RegularityResult(false, :sufficient_condition, T(NaN), false)
    end

    λ_min_center = try
        eigvals(Symmetric(AtA_c))[1]  # Smallest eigenvalue
    catch e
        @warn "Failed to compute eigenvalues of A_c^T A_c: $e"
        return RegularityResult(false, :sufficient_condition, T(NaN), false)
    end

    # Check sufficient condition
    separation = λ_min_center - λ_max_rad

    is_regular = separation > 0

    return RegularityResult(is_regular, :sufficient_condition, separation, false)
end

"""
    is_regular_gershgorin(A::BallMatrix{T}) where {T}

Test regularity using Gershgorin circle theorem.

# Gershgorin Criterion
Matrix [A] is regular if for all i, 0 is not in the Gershgorin disc:
    D_i = {z : |z - a_{ii}| ≤ ∑_{j≠i} |a_{ij}|}

# Arguments
- `A`: Interval matrix (n×n)

# Returns
`RegularityResult` with test outcome.

# Example
```julia
A = BallMatrix([4.0 1.0 0.5; 0.5 3.0 0.5; 0.5 0.5 2.0], fill(0.1, 3, 3))

result = is_regular_gershgorin(A)

if result.is_regular
    println("Matrix is proven regular by Gershgorin")
end
```

# Notes
- O(n²) complexity - very fast
- Sufficient but not necessary
- Works well for diagonally dominant matrices
- Conservative for general matrices
"""
function is_regular_gershgorin(A::BallMatrix{T}) where {T}
    n = size(A, 1)

    min_separation = T(Inf)

    for i in 1:n
        # Get diagonal element interval
        a_ii = A[i, i]
        a_ii_inf = inf(a_ii)
        a_ii_sup = sup(a_ii)

        # Compute row sum excluding diagonal
        row_sum = T(0)
        for j in 1:n
            if j != i
                a_ij = A[i, j]
                # Maximum absolute value
                row_sum += max(abs(inf(a_ij)), abs(sup(a_ij)))
            end
        end

        # Check if 0 is outside disc
        # 0 is outside if min|a_ii| > row_sum
        min_abs_diag = min(abs(a_ii_inf), abs(a_ii_sup))

        # If diagonal crosses zero, can't use this criterion
        if a_ii_inf * a_ii_sup < 0
            return RegularityResult(false, :gershgorin, T(0), false)
        end

        separation = min_abs_diag - row_sum
        min_separation = min(min_separation, separation)

        if separation <= 0
            # Zero may be in Gershgorin disc
            return RegularityResult(false, :gershgorin, separation, false)
        end
    end

    return RegularityResult(true, :gershgorin, min_separation, false)
end

"""
    is_regular_diagonal_dominance(A::BallMatrix{T};
                                  strict::Bool=true) where {T}

Test regularity using diagonal dominance.

# Criterion
Matrix [A] is regular if it is strictly diagonally dominant:
    |a_{ii}| > ∑_{j≠i} |a_{ij}|  for all i

# Arguments
- `A`: Interval matrix (n×n)
- `strict`: Require strict (>) or weak (≥) diagonal dominance

# Returns
`RegularityResult` with test outcome.

# Example
```julia
A = BallMatrix([5.0 1.0 1.0; 1.0 4.0 0.5; 0.5 1.0 3.0], fill(0.1, 3, 3))

result = is_regular_diagonal_dominance(A)

if result.is_regular
    println("Matrix is strictly diagonally dominant")
end
```

# Notes
- O(n²) complexity
- Very fast test
- Sufficient for regularity
- Many practical matrices satisfy this
"""
function is_regular_diagonal_dominance(A::BallMatrix{T};
                                      strict::Bool=true) where {T}
    n = size(A, 1)

    min_margin = T(Inf)

    for i in 1:n
        # Get diagonal element
        a_ii = A[i, i]

        # Minimum absolute value of diagonal
        min_abs_diag = min(abs(inf(a_ii)), abs(sup(a_ii)))

        # If diagonal crosses zero, not diagonally dominant
        if inf(a_ii) * sup(a_ii) < 0
            return RegularityResult(false, :diagonal_dominance, T(-Inf), false)
        end

        # Compute maximum row sum excluding diagonal
        row_sum = T(0)
        for j in 1:n
            if j != i
                a_ij = A[i, j]
                row_sum += max(abs(inf(a_ij)), abs(sup(a_ij)))
            end
        end

        margin = min_abs_diag - row_sum
        min_margin = min(min_margin, margin)

        if strict && margin <= 0
            return RegularityResult(false, :diagonal_dominance, margin, false)
        elseif !strict && margin < 0
            return RegularityResult(false, :diagonal_dominance, margin, false)
        end
    end

    return RegularityResult(true, :diagonal_dominance, min_margin, false)
end

"""
    is_regular(A::BallMatrix{T};
              methods::Vector{Symbol}=[:sufficient_condition, :gershgorin, :diagonal_dominance],
              verbose::Bool=false) where {T}

Test regularity using multiple methods.

Tries several sufficient conditions in order until one succeeds.

# Arguments
- `A`: Interval matrix (n×n)
- `methods`: Vector of methods to try (in order)
- `verbose`: Print information about each method

# Available Methods
- `:sufficient_condition`: Eigenvalue-based (Theorem 11.12)
- `:gershgorin`: Gershgorin circle theorem
- `:diagonal_dominance`: Strict diagonal dominance

# Returns
`RegularityResult` from first successful method, or last result if all fail.

# Example
```julia
A = BallMatrix([3.0 0.5; 0.5 2.0], fill(0.1, 2, 2))

result = is_regular(A, verbose=true)

if result.is_regular
    println("Matrix is regular (proven by ", result.method, ")")
else
    println("Regularity could not be proven")
end
```

# Notes
- Tries multiple sufficient conditions
- Returns true only if regularity is proven
- False result is inconclusive (matrix may still be regular)
- Order methods by speed: diagonal_dominance → gershgorin → sufficient_condition
"""
function is_regular(A::BallMatrix{T};
                   methods::Vector{Symbol}=[:diagonal_dominance, :gershgorin, :sufficient_condition],
                   verbose::Bool=false) where {T}
    last_result = RegularityResult(false, :none, T(NaN), false)

    for method in methods
        if verbose
            println("Trying method: $method")
        end

        result = if method == :sufficient_condition
            is_regular_sufficient_condition(A)
        elseif method == :gershgorin
            is_regular_gershgorin(A)
        elseif method == :diagonal_dominance
            is_regular_diagonal_dominance(A)
        else
            @warn "Unknown regularity test method: $method"
            continue
        end

        if verbose
            if result.is_regular
                println("  ✓ Regular (certificate: $(result.certificate))")
            else
                println("  ✗ Inconclusive (certificate: $(result.certificate))")
            end
        end

        if result.is_regular
            return result
        end

        last_result = result
    end

    return last_result
end

"""
    is_singular_sufficient_condition(A::BallMatrix{T}) where {T}

Test singularity using sufficient condition from Theorem 11.13 (Horáček, p. 183).

# Sufficient Condition for Singularity
Matrix [A] contains at least one singular matrix if:
    λ_min(A_c^T A_c) < λ_max(A_Δ^T A_Δ)

This is the dual of the regularity condition.

# Arguments
- `A`: Interval matrix (n×n)

# Returns
`true` if proven singular, `false` if inconclusive.

# Example
```julia
A = BallMatrix([1.0 1.0; 1.0 1.0], fill(0.05, 2, 2))

if is_singular_sufficient_condition(A)
    println("Matrix contains singular matrices")
end
```

# Notes
- O(n³) complexity
- Sufficient but not necessary
- Dual of regularity test
"""
function is_singular_sufficient_condition(A::BallMatrix{T}) where {T}
    result = is_regular_sufficient_condition(A)

    # If separation < 0, singular
    return !result.is_regular && result.certificate < 0
end

# Export functions
export RegularityResult
export is_regular_sufficient_condition, is_regular_gershgorin, is_regular_diagonal_dominance
export is_regular, is_singular_sufficient_condition
