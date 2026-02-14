"""
    preconditioning.jl

Preconditioning strategies for interval linear systems.
Provides various methods to compute approximate inverses C ≈ A^(-1).

# References
- Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis
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
    PreconditionerResult{T}

Result from preconditioner computation.

# Fields
- `preconditioner::Matrix{T}`: Approximate inverse C
- `method::PreconditionerType`: Method used
- `condition_number::T`: Estimated condition number
- `success::Bool`: Whether computation succeeded
- `factorization::Any`: Stored factorization (for LU/LDLT)
"""
struct PreconditionerResult{T}
    """Approximate inverse matrix."""
    preconditioner::Matrix{T}
    """Preconditioning method used."""
    method::PreconditionerType
    """Estimated condition number."""
    condition_number::T
    """Whether computation succeeded."""
    success::Bool
    """Stored factorization (if applicable)."""
    factorization::Any
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
        C = Matrix{T}(I, n, n)
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
                Matrix{T}(I, n, n), MidpointInverse, T(Inf), false, nothing
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
                Matrix{T}(I, n, n), LUFactorization, T(Inf), false, nothing
            )
        end

    elseif method == :ldlt
        # LDLT factorization for symmetric matrices
        try
            ldlt_fact = ldlt(Symmetric(A_mid))
            # Reconstruct inverse from LDLT
            C = inv(ldlt_fact)
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
    is_well_preconditioned(A::BallMatrix{T}, prec::PreconditionerResult{T};
                          threshold::T=T(0.5)) where {T}

Check if preconditioner is effective.

A good preconditioner should make ‖I - CA‖ small (ideally < 0.5).

# Arguments
- `A`: Original interval matrix
- `prec`: Preconditioner result
- `threshold`: Threshold for ‖I - CA‖ (default 0.5)

# Returns
`true` if ‖I - CA‖ < threshold

# Example
```julia
A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.1, 2, 2))
prec = compute_preconditioner(A)

if is_well_preconditioned(A, prec)
    println("Good preconditioner")
end
```
"""
function is_well_preconditioned(A::BallMatrix{T}, prec::PreconditionerResult{T};
                               threshold::T=T(0.5)) where {T}
    # Compute I - CA using interval arithmetic
    C = prec.preconditioner
    A_mid = mid(A)
    A_rad = rad(A)

    # I - CA_mid
    I_minus_CA_mid = I - C * A_mid

    # Radius part: |C| * A_rad
    CA_rad = abs.(C) * A_rad

    # Norm of interval matrix [I - CA] (round up for rigorous upper bound)
    total_norm = setrounding(T, RoundUp) do
        opnorm(I_minus_CA_mid, Inf) + opnorm(CA_rad, Inf)
    end

    return total_norm < threshold
end

# Export functions
export PreconditionerType, MidpointInverse, LUFactorization, LDLTFactorization, IdentityPreconditioner
export PreconditionerResult
export compute_preconditioner, apply_preconditioner, is_well_preconditioned
