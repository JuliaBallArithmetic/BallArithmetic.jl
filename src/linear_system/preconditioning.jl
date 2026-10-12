"""
    preconditioning.jl

Preconditioning strategies for interval linear systems.
Provides various methods to compute approximate inverses C ≈ A^(-1).

# References
- J. Horáček, Interval linear and nonlinear systems, doctoral thesis, Charles University, Prague, 2019, Sections 5.4 and 8.4.6
- Neumaier, A. (1990), "Interval Methods for Systems of Equations"
"""

using LinearAlgebra

"""
    PreconditionerType

Enumeration of available preconditioning strategies.

- `:midpoint`: C = A_c^(-1) (inverse of midpoint)
- `:lu`: C from LU factorization of midpoint
- `:ldlt`: C from LDLT factorization of midpoint (symmetric matrices)
- `:identity`: C = I (no preconditioning)
"""
@enum PreconditionerType begin
    MidpointInverse
    LUFactorization
    LDLTFactorization
    IdentityPreconditioner
end

"""
    PreconditionerResult{CT, T, F}

Result from preconditioner computation. Nothing in it is verified: the preconditioner is an
approximate inverse and the condition number an estimate.

# Fields
- `preconditioner::Matrix{CT}`: Approximate inverse C, real or complex as the midpoint is
- `method::PreconditionerType`: Method used
- `condition_number::T`: Estimated condition number, a real number
- `success::Bool`: Whether computation succeeded
- `factorization`: the stored factorization (for LU/LDLT), `nothing` otherwise
"""
struct PreconditionerResult{CT, T, F}
    """Approximate inverse matrix."""
    preconditioner::Matrix{CT}
    """Preconditioning method used."""
    method::PreconditionerType
    """Estimated condition number."""
    condition_number::T
    """Whether computation succeeded."""
    success::Bool
    """Stored factorization (if applicable)."""
    factorization::F
end

"""
    compute_preconditioner(A::BallMatrix{T};
                          method::Symbol=:midpoint,
                          check_conditioning::Bool=true) where {T}

Compute preconditioner C ≈ A^(-1) for interval matrix A.

# Methods
- `:midpoint`: Direct inverse of A_c (default, simplest)
- `:lu`: LU factorization of A_c (more stable)
- `:ldlt`: LDLT factorization of A_c (for symmetric A)
- `:identity`: Identity matrix (no preconditioning)

# Arguments
- `A`: Interval matrix to precondition
- `method`: Preconditioning strategy
- `check_conditioning`: Compute and warn about condition number

# Returns
`PreconditionerResult` with approximate inverse and diagnostic info.

# Example
```julia
A = BallMatrix([4.0 1.0; 1.0 3.0], fill(0.1, 2, 2))

# Midpoint inverse (default)
prec = compute_preconditioner(A)
C = prec.preconditioner

# LU factorization (more stable)
prec_lu = compute_preconditioner(A, method=:lu)
```

# Notes
- Midpoint inverse: O(n³), simplest but may be numerically unstable
- LU factorization: O(n³) once, O(n²) per solve, more stable
- LDLT factorization: O(n³/6) once, O(n²) per solve, only for symmetric
- Identity: O(1), use for well-conditioned or diagonally dominant matrices
"""
function compute_preconditioner(A::BallMatrix{T};
                                method::Symbol=:midpoint,
                                check_conditioning::Bool=true) where {T}
    n = size(A, 1)
    A_mid = mid(A)
    CT = eltype(A_mid)

    # Check for symmetry if using LDLT
    if method == :ldlt
        if !issymmetric(A_mid)
            @warn "LDLT requires symmetric matrix, falling back to LU"
            method = :lu
        end
    end

    # Compute preconditioner based on method
    if method == :identity
        # No preconditioning
        C = Matrix{CT}(I, n, n)
        cond_est = T(1.0)
        factorization = nothing
        prec_type = IdentityPreconditioner

    elseif method == :midpoint
        # Direct inverse of midpoint
        try
            C = inv(A_mid)
            cond_est = check_conditioning ? cond(A_mid) : T(NaN)
            factorization = nothing
            prec_type = MidpointInverse
        catch e
            @warn "Midpoint inverse failed: $e"
            return PreconditionerResult(
                Matrix{CT}(I, n, n), MidpointInverse, T(Inf), false, nothing
            )
        end

    elseif method == :lu
        # LU factorization
        try
            lu_fact = lu(A_mid)
            C = inv(lu_fact)
            cond_est = check_conditioning ? cond(A_mid) : T(NaN)
            factorization = lu_fact
            prec_type = LUFactorization
        catch e
            @warn "LU factorization failed: $e"
            return PreconditionerResult(
                Matrix{CT}(I, n, n), LUFactorization, T(Inf), false, nothing
            )
        end

    elseif method == :ldlt
        # LDLT factorization for symmetric matrices
        try
            # the LDLᵀ factorisation of a dense symmetric matrix, with the pivoting of Bunch and
            # Kaufman; `ldlt` itself has methods for tridiagonal and sparse matrices only
            ldlt_fact = bunchkaufman(Symmetric(A_mid))
            C = Matrix{CT}(inv(ldlt_fact))
            cond_est = check_conditioning ? cond(A_mid) : T(NaN)
            factorization = ldlt_fact
            prec_type = LDLTFactorization
        catch e
            @warn "LDLT factorization failed: $e, falling back to LU"
            return compute_preconditioner(A, method=:lu, check_conditioning=check_conditioning)
        end

    else
        error("Unknown preconditioning method: $method")
    end

    # Check conditioning
    if check_conditioning && cond_est > 1e10
        @warn "Matrix is poorly conditioned: cond(A) ≈ $cond_est"
    end

    return PreconditionerResult(C, prec_type, cond_est, true, factorization)
end

"""
    apply_preconditioner(prec::PreconditionerResult{T}, v::Vector{T}) where {T}

Apply preconditioner to vector: compute C * v.

If factorization is available, uses it for efficiency.

# Arguments
- `prec`: Preconditioner result from `compute_preconditioner`
- `v`: Vector to multiply

# Returns
C * v

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.1, 2, 2))
prec = compute_preconditioner(A, method=:lu)

b = [5.0, 4.0]
Cb = apply_preconditioner(prec, b)
```
"""
function apply_preconditioner(prec::PreconditionerResult{T}, v::Vector{T}) where {T}
    if prec.factorization !== nothing && prec.method == LUFactorization
        # Use LU factorization
        return prec.factorization \ v
    elseif prec.factorization !== nothing && prec.method == LDLTFactorization
        # Use LDLT factorization
        return prec.factorization \ v
    else
        # Direct matrix-vector product
        return prec.preconditioner * v
    end
end

"""
    apply_preconditioner(prec::PreconditionerResult{T}, M::Matrix{T}) where {T}

Apply preconditioner to matrix: compute C * M.

# Arguments
- `prec`: Preconditioner result
- `M`: Matrix to multiply

# Returns
C * M
"""
function apply_preconditioner(prec::PreconditionerResult{T}, M::Matrix{T}) where {T}
    if prec.factorization !== nothing && prec.method == LUFactorization
        # Use LU factorization
        return prec.factorization \ M
    elseif prec.factorization !== nothing && prec.method == LDLTFactorization
        # Use LDLT factorization
        return prec.factorization \ M
    else
        # Direct matrix-matrix product
        return prec.preconditioner * M
    end
end

"""
    apply_preconditioner(prec::PreconditionerResult, A::BallMatrix) -> BallMatrix
    apply_preconditioner(prec::PreconditionerResult, b::BallVector) -> BallVector

The preconditioned data `C A` and `C b` of Section 5.4 of Horáček (2019), formed in ball
arithmetic with `C = prec.preconditioner`, so that the solution set of `(CA) x = Cb` contains
that of `A x = b` whatever the accuracy of `C`.
"""
apply_preconditioner(prec::PreconditionerResult, A::BallMatrix) =
    BallMatrix(prec.preconditioner) * A
apply_preconditioner(prec::PreconditionerResult, b::BallVector) =
    BallMatrix(prec.preconditioner) * b

"""
    is_well_preconditioned(A::BallMatrix, prec::PreconditionerResult; threshold = 0.5) -> Bool

`true` when `‖I − C A‖_∞ < threshold` is proved for every matrix `A` of the ball, with
`C = prec.preconditioner`: the product is formed in ball arithmetic and its norm bounded from
above. With `threshold ≤ 1` this makes `CA` strictly diagonally dominant (Example 4.25 of
Horáček (2019)) and gives the initial enclosure (5.4). `false` means that the bound was not
established.
"""
function is_well_preconditioned(A::BallMatrix{T}, prec::PreconditionerResult;
        threshold::Real = 0.5) where {T}
    return upper_bound_L_inf_opnorm(BallMatrix(prec.preconditioner) * A - I) < threshold
end

export PreconditionerType, MidpointInverse, LUFactorization, LDLTFactorization, IdentityPreconditioner
export PreconditionerResult
export compute_preconditioner, apply_preconditioner, is_well_preconditioned
