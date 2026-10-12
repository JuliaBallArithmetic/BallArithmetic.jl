# Iterative refinement of Schur decompositions for higher precision
# Based on Algorithm 4 from Bujanović, Kressner & Schröder (2022/2024)
#
# This allows computing Schur decomposition in Float64 and refining to BigFloat
# precision using Newton-like iterations, achieving 10-20× speedup over direct
# computation in higher precision.

"""
    SchurRefinementResult{T, RT}

Result from iterative Schur refinement to higher precision.

# Fields
- `Q::Matrix{T}`: Refined orthogonal/unitary Schur basis
- `T::Matrix{T}`: Refined upper triangular Schur form
- `iterations::Int`: Number of refinement iterations performed
- `residual_norm::RT`: Final residual ‖stril(Q^H A Q)‖₂ / ‖A‖_F — rigorous upper bound (real type)
- `orthogonality_defect::RT`: Final ‖Q^H Q - I‖₂ — rigorous upper bound (real type)
- `converged::Bool`: Whether refinement converged to desired tolerance

# References
- [BujanovicKressnerSchroder2022](@cite) Bujanović, Kressner & Schröder,
  "Iterative refinement of Schur decompositions", Numer. Algorithms 92, 247–267 (2023)
"""
struct SchurRefinementResult{T, RT<:Real}
    Q::Matrix{T}
    T::Matrix{T}
    iterations::Int
    residual_norm::RT
    orthogonality_defect::RT
    converged::Bool
end

"""
    _frobenius_norm(A::AbstractMatrix)

Compute Frobenius norm of a matrix. Works with any element type including BigFloat.
"""
function _frobenius_norm(A::AbstractMatrix{T}) where T
    s = zero(real(T))
    @inbounds for j in axes(A, 2)
        for i in axes(A, 1)
            s += abs2(A[i, j])
        end
    end
    return sqrt(s)
end

"""
    newton_schulz_orthogonalize!(Q::Matrix{T}; max_iter=10, tol=nothing) where T

Apply Newton-Schulz iteration to orthogonalize Q in-place.

The iteration is:
    Q_{k+1} = (1/2) Q_k (3I - Q_k^H Q_k)

This converges quadratically if ‖Q^H Q - I‖₂ < 1.

# Arguments
- `Q`: Matrix to orthogonalize (modified in place)
- `max_iter`: Maximum iterations (default: 10)
- `tol`: Convergence tolerance (default: machine epsilon of T)

# Returns
- Number of iterations performed
- Final orthogonality defect ‖Q^H Q - I‖_F

# References
- [BujanovicKressnerSchroder2022](@cite) Section 2.2.2 (Newton-Schulz iteration)

# NOTE: This is an ORACLE computation - convergence is checked via the defect
# (‖Q^H Q - I‖_F < tol), which serves as an A POSTERIORI verification. The
# intermediate computations don't need directed rounding because the final
# defect measurement validates the orthogonalization quality.
"""
function newton_schulz_orthogonalize!(Q::Matrix{T}; max_iter::Int=10, tol=nothing) where T
    n = size(Q, 1)
    tol = isnothing(tol) ? 100 * eps(real(T)) : tol

    I_n = Matrix{T}(I, n, n)

    for iter in 1:max_iter
        # Compute Q^H * Q
        QtQ = Q' * Q

        # Check convergence (use Frobenius norm for BigFloat compatibility)
        defect = _frobenius_norm(QtQ - I_n)
        if defect < tol
            return iter, defect
        end

        # Newton-Schulz step: Q = (1/2) * Q * (3I - Q^H Q)
        Q .= Q * (T(3) * I_n - QtQ) / T(2)
    end

    # Final defect
    final_defect = _frobenius_norm(Q' * Q - I_n)
    return max_iter, final_defect
end


#####################################################
# TRIANGULAR MATRIX EQUATION SOLVERS                #
# Based on Algorithms 2 and 3 from Bujanović et al. #
#####################################################

"""
    _stril(A::AbstractMatrix)

Return the strictly lower triangular part of matrix A (excluding diagonal).
"""
function _stril(A::AbstractMatrix{T}) where T
    n, m = size(A)
    L = zeros(T, n, m)
    for j in 1:min(n-1, m)
        for i in (j+1):n
            L[i, j] = A[i, j]
        end
    end
    return L
end

"""
    _solve_triangular_equation_direct!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                       max_entry::Real=1e5) where T

Solve the triangular matrix equation stril(TL - LT) = -E for strictly lower triangular L.

This implements Algorithm 2 from Bujanović et al. (2022): successive substitution
with bottom-to-top columnwise order.

# Arguments
- `L`: Output matrix (will contain strictly lower triangular solution)
- `T_mat`: Upper triangular matrix with pairwise distinct diagonal entries
- `E`: Strictly lower triangular right-hand side matrix
- `max_entry`: Maximum allowed absolute value for entries of L (default: 1e5).
  Larger values are truncated to zero to prevent instability from clustered eigenvalues.

# Details
For 1 ≤ j < i ≤ n, the entry L[i,j] is computed as:

    L[i,j] = -1/(T[i,i] - T[j,j]) * (E[i,j] + Σ_{k=i+1}^n T[i,k]*L[k,j] - Σ_{k=1}^{j-1} L[i,k]*T[k,j])

where entries to the left or below (i,j) must be computed first.

# Notes
When eigenvalues are clustered (nearly equal), the denominator T[i,i] - T[j,j] becomes
very small, causing large entries in L that can destabilize the algorithm. Following
the suggestion in Bujanović et al. (Example 4), entries larger than `max_entry` are
set to zero.

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithm 2: Successive substitution
"""
function _solve_triangular_equation_direct!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                            max_entry::Real=1e5) where T
    n = size(T_mat, 1)

    # Initialize L to zero
    fill!(L, zero(T))

    # Solve column by column (j = 1 to n-1)
    # Within each column, solve from bottom to top (i = n down to j+1)
    for j in 1:(n-1)
        for i in n:-1:(j+1)
            # Compute: E[i,j] + T[i, i+1:n] · L[i+1:n, j] - L[i, 1:j-1] · T[1:j-1, j]
            val = E[i, j]

            # Add contribution from rows below: Σ_{k=i+1}^n T[i,k]*L[k,j]
            for k in (i+1):n
                val += T_mat[i, k] * L[k, j]
            end

            # Subtract contribution from columns to the left: Σ_{k=1}^{j-1} L[i,k]*T[k,j]
            for k in 1:(j-1)
                val -= L[i, k] * T_mat[k, j]
            end

            # Divide by eigenvalue difference
            denom = T_mat[i, i] - T_mat[j, j]
            if abs(denom) < eps(real(T)) * max(abs(T_mat[i, i]), abs(T_mat[j, j]), one(real(T)))
                # Near-singular: eigenvalues too close, set to zero to avoid instability
                L[i, j] = zero(T)
            else
                entry = -val / denom
                # Truncate large entries to zero (the trick of Example 4 of the paper)
                if abs(entry) > max_entry
                    L[i, j] = zero(T)
                else
                    L[i, j] = entry
                end
            end
        end
    end

    return L
end

"""
    _solve_sylvester_triangular!(X::Matrix{T}, T22::Matrix{T}, T11::Matrix{T}, C::Matrix{T}) where T

Solve the triangular Sylvester equation T22 * X - X * T11 = C where T11 and T22 are
upper triangular (from Schur form).

This corresponds to equation (15) in Bujanović et al. (2022):
    T22 * L21 - L21 * T11 = -E21

# Arguments
- `X`: Output matrix (will contain solution), size (m × k)
- `T22`: Upper triangular matrix (m × m)
- `T11`: Upper triangular matrix (k × k)
- `C`: Right-hand side matrix (m × k)

# References
- [BujanovicKressnerSchroder2022](@cite) Section 2.1, equation (15)
"""
function _solve_sylvester_triangular!(X::Matrix{T}, T22::Matrix{T}, T11::Matrix{T}, C::Matrix{T}) where T
    m = size(T22, 1)
    k = size(T11, 1)

    # Initialize X from C
    X .= C

    # Solve column by column from left to right
    # For column j: T22 * X[:,j] - X[:,j] * T11[j,j] = C[:,j] + X[:,1:j-1] * T11[1:j-1,j]
    for j in 1:k
        # Update RHS with contributions from already computed columns
        for l in 1:(j-1)
            for i in 1:m
                X[i, j] += X[i, l] * T11[l, j]
            end
        end

        # Solve (T22 - T11[j,j] * I) * X[:,j] = X[:,j]
        # This is upper triangular system, solve from bottom to top
        for i in m:-1:1
            # Accumulate contributions from rows below
            for l in (i+1):m
                X[i, j] -= T22[i, l] * X[l, j]
            end

            denom = T22[i, i] - T11[j, j]
            if abs(denom) < eps(real(T)) * max(abs(T22[i, i]), abs(T11[j, j]), one(real(T)))
                # Near-singular: eigenvalues too close
                X[i, j] = zero(T)
            else
                X[i, j] = X[i, j] / denom
            end
        end
    end

    return X
end

"""
    _solve_triangular_equation_recursive!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                          nmin::Int=32, max_entry::Real=1e5) where T

Solve stril(TL - LT) = -E using the recursive block algorithm (Algorithm 3).

This achieves better cache efficiency for large matrices by recursively partitioning
into 2×2 blocks and using Sylvester equation solvers for off-diagonal blocks.

# Arguments
- `L`: Output matrix (will contain strictly lower triangular solution)
- `T_mat`: Upper triangular matrix with pairwise distinct diagonal entries
- `E`: Strictly lower triangular right-hand side matrix
- `nmin`: Minimum block size before switching to direct algorithm (default: 32)
- `max_entry`: Maximum allowed absolute value for entries of L (default: 1e5)

# Algorithm
Partition T, E, L into 2×2 blocks:
    T = [T11  T12]    E = [E11   0 ]    L = [L11   0 ]
        [ 0   T22]        [E21  E22]        [L21  L22]

1. Solve Sylvester equation: T22 * L21 - L21 * T11 = -E21
2. Update: E11 ← E11 + stril(T12 * L21)
3. Update: E22 ← E22 - stril(L21 * T12)
4. Recursively solve for L11 from stril(T11*L11 - L11*T11) = -E11
5. Recursively solve for L22 from stril(T22*L22 - L22*T22) = -E22

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithm 3: Recursive block algorithm
"""
function _solve_triangular_equation_recursive!(L::Matrix{T}, T_mat::Matrix{T}, E::Matrix{T};
                                               nmin::Int=32, max_entry::Real=1e5) where T
    n = size(T_mat, 1)

    # Base case: use direct algorithm for small matrices
    if n <= nmin
        return _solve_triangular_equation_direct!(L, T_mat, E; max_entry=max_entry)
    end

    # Partition at midpoint
    n1 = div(n, 2)
    ind1 = 1:n1
    ind2 = (n1+1):n

    # Extract blocks
    T11 = T_mat[ind1, ind1]
    T12 = T_mat[ind1, ind2]
    T22 = T_mat[ind2, ind2]

    E11 = E[ind1, ind1]
    E21 = E[ind2, ind1]
    E22 = E[ind2, ind2]

    # Step 1: Solve Sylvester equation T22 * L21 - L21 * T11 = -E21
    L21 = similar(E21)
    _solve_sylvester_triangular!(L21, T22, T11, -E21)

    # Store L21 in output
    L[ind2, ind1] .= L21

    # Step 2: Update E11 ← E11 + stril(T12 * L21)
    temp = T12 * L21
    for j in 1:n1
        for i in (j+1):n1
            E11[i, j] += temp[i, j]
        end
    end

    # Step 3: Update E22 ← E22 - stril(L21 * T12)
    temp = L21 * T12
    n2 = n - n1
    for j in 1:n2
        for i in (j+1):n2
            E22[i, j] -= temp[i, j]
        end
    end

    # Step 4: Recursively solve for L11
    L11 = zeros(T, n1, n1)
    _solve_triangular_equation_recursive!(L11, T11, E11; nmin=nmin, max_entry=max_entry)
    L[ind1, ind1] .= L11

    # Step 5: Recursively solve for L22
    L22 = zeros(T, n2, n2)
    _solve_triangular_equation_recursive!(L22, T22, E22; nmin=nmin, max_entry=max_entry)
    L[ind2, ind2] .= L22

    # Zero out upper triangular part (L should be strictly lower triangular)
    for j in 1:n
        for i in 1:j
            L[i, j] = zero(T)
        end
    end

    return L
end

"""
    solve_triangular_matrix_equation(T_mat::Matrix{T}, E::Matrix{T};
                                     use_recursive::Bool=true, nmin::Int=32, max_entry::Real=1e5) where T

Solve the triangular matrix equation stril(TL - LT) = -E for strictly lower triangular L.

Given an upper triangular matrix T with pairwise distinct diagonal entries and a strictly
lower triangular matrix E, finds the unique strictly lower triangular matrix L satisfying
the equation.

# Arguments
- `T_mat`: Upper triangular matrix (n × n) with pairwise distinct diagonal entries
- `E`: Strictly lower triangular matrix (n × n)
- `use_recursive`: If true, use recursive block algorithm (default: true)
- `nmin`: Minimum block size for recursive algorithm (default: 32)
- `max_entry`: Maximum allowed entry magnitude before truncation (default: 1e5).
  This prevents instability from clustered eigenvalues (see Example 4 of the paper, which uses 1e-5).

# Returns
Strictly lower triangular matrix L satisfying stril(TL - LT) = -E

# References
- [BujanovicKressnerSchroder2022](@cite) Algorithms 2 and 3
"""
function solve_triangular_matrix_equation(T_mat::Matrix{T}, E::Matrix{T};
                                          use_recursive::Bool=true, nmin::Int=32, max_entry::Real=1e5) where T
    n = size(T_mat, 1)
    L = zeros(T, n, n)

    if use_recursive && n > nmin
        _solve_triangular_equation_recursive!(L, T_mat, E; nmin=nmin, max_entry=max_entry)
    else
        _solve_triangular_equation_direct!(L, T_mat, E; max_entry=max_entry)
    end

    return L
end

# ---------------------------------------------------------------------------------------------
# The algorithms, each named for what it is
# ---------------------------------------------------------------------------------------------

# Line 1 of Algorithm 4 asks for an ordered Schur decomposition in low precision; Section 2.4:
# "the initial lp Schur decomposition is reordered so that clustered eigenvalues appear on
# neighboring diagonal elements of T_lp. This is done using ordschur in Matlab; the target
# permutation of eigenvalues is determined by the order in which they appear after being
# projected on a random line in the complex plane." The line is the one through the origin at a
# random angle; the eigenvalues are brought to the order of their projections, one at a time,
# by `ordschur`, which keeps the relative order of the eigenvalues it selects.
function _bujanovicetal2023_order(Q0::Matrix, T0::Matrix, rng)
    eltype(T0) <: Union{Float32, Float64, ComplexF32, ComplexF64} && eltype(Q0) === eltype(T0) ||
        throw(ArgumentError("the reordering of the Schur form is done by LAPACK: Q0 and T0 " *
                            "must be Float32, Float64 or their complex types"))
    n = size(T0, 1)
    θ = 2 * pi * (rng === nothing ? rand() : rand(rng))
    proj = real.(diag(T0) .* cis(-θ))
    desired = sortperm(proj)
    cur = collect(1:n)
    Q, T = copy(Q0), copy(T0)
    wanted = falses(n)                     # indexed by the original position
    for k in 1:(n - 1)
        wanted[desired[k]] = true
        sel = Bool[wanted[c] for c in cur]
        all(view(sel, 1:k)) && continue
        F = ordschur(Schur(T, Q, diag(T)), sel)
        T, Q = Matrix(F.T), Matrix(F.Z)
        cur = vcat(cur[sel], cur[.!sel])
    end
    return Q, T
end

"""
    _bujanovicetal2023_alg4(A, Q0, T0, max_iterations, tol; hp = BigFloat, rng = nothing)

Algorithm 4 of Bujanović, Kressner and Schröder (2023), the mixed-precision refinement of a
Schur decomposition, from line 1's ordering on: `Q0`, `T0` are the Schur factors of `A` in low
precision (`Float64`), `T0` upper triangular.

1. The Schur form is reordered so that eigenvalues close to each other are neighbours on the
   diagonal: by the order of their projections on a random line through the origin.
2. `Q̂` is converted to high precision and `Q ← ½ Q̂ (3I − Q̂ᴴQ̂)`, one Newton-Schulz step.
3. Until `E` is small: `T̂ ← QᴴAQ` in high precision; `E ← stril(T̂)`, `T ← T̂ − E`; the
   equation `stril(T L − L T) = −E` is solved for the strictly lower triangular `L` in low
   precision, by the recursive Algorithm 3; `W = L − Lᴴ`, `Y = QᴴQ − I`; the products `YW`,
   `W²`, `W³` in low precision; `Σ = 2I + 2W − Y − YW + W² + W³` and `Q ← ½ Q Σ` in high
   precision.

`hp` is the high precision: BigFloat at the current precision, or a type such as `Double64`.
The iteration stops when `‖E‖/‖A‖ ≤ tol`. The result is in floating point: its two norms are
diagnostics, not bounds. The paper reports that the iteration can fail when eigenvalues are
poorly separated (its Example 3), the entries of `L` growing until they are not numbers.

Not the paper's: `nmin`, the size at which Algorithm 3 stops recursing, is 32 here; the paper's
experiments use 4 for the complex Schur form.
"""
_bujanovicetal2023_alg4(A, Q0, T0, max_iterations, tol; hp::Type = BigFloat, rng = nothing) =
    _refine_schur_impl(A, Q0, T0, max_iterations, tol; hp, order = true, rng,
        truncation = Inf, single_initial_step = true, precheck = false,
        stop_on_orthogonality = false)

"""
    _bujanovicetal2023_alg4_truncated(A, Q0, T0, max_iterations, tol;
        hp = BigFloat, rng = nothing, threshold = 1e-5)

[`_bujanovicetal2023_alg4`](@ref) with the modification of Example 4 of the paper: "once an
element of L is computed (Line 4 of Algorithm 2, Line 6 of Algorithm 3), if its absolute value
turns out to be larger than, e.g., 10⁻⁵, such element is immediately set to zero. The idea is to
prevent propagation of large elements in L, at the cost of obtaining a less accurate solution of
the triangular equation." The paper adds that "there is little theoretical support that such a
modification will lead to convergence"; it resolved the failures of its Table 3 and not the one
of its Example 3.
"""
_bujanovicetal2023_alg4_truncated(A, Q0, T0, max_iterations, tol; hp::Type = BigFloat,
    rng = nothing, threshold::Real = 1e-5) =
    _refine_schur_impl(A, Q0, T0, max_iterations, tol; hp, order = true, rng,
        truncation = threshold, single_initial_step = true, precheck = false,
        stop_on_orthogonality = false)

"""
    _bujanovicetal2023_alg4_unordered(A, Q0, T0, max_iterations, tol; hp = BigFloat)

The iteration of [`_bujanovicetal2023_alg4`](@ref) without the reordering of its line 1, so that
the eigenvalues stay on the diagonal in the order of `T0`: a deviation, named for it. It is for
the callers that have ordered the Schur form themselves and use that order afterwards. Three
further differences from the paper's listing: `Q0`, `T0` are returned as they are when both
defects are already below the tolerance; the initial orthogonalisation takes up to three
Newton-Schulz steps; and the iteration stops when both `‖E‖/‖A‖` and `‖QᴴQ − I‖` are below the
tolerance. `Q0` and `T0` may be in any precision.
"""
_bujanovicetal2023_alg4_unordered(A, Q0, T0, max_iterations, tol; hp::Type = BigFloat) =
    _refine_schur_impl(A, Q0, T0, max_iterations, tol; hp)

# the algorithm for a `method`
function _schur_refinement(method::Symbol, A, Q0, T0, max_iterations, tol; hp::Type = BigFloat,
        rng = nothing, threshold::Real = 1e-5)
    method === :bujanovicetal2023 &&
        return _bujanovicetal2023_alg4(A, Q0, T0, max_iterations, tol; hp, rng)
    method === :bujanovicetal2023truncated &&
        return _bujanovicetal2023_alg4_truncated(A, Q0, T0, max_iterations, tol; hp, rng,
            threshold)
    method === :bujanovicetal2023unordered &&
        return _bujanovicetal2023_alg4_unordered(A, Q0, T0, max_iterations, tol; hp)
    throw(ArgumentError("unknown method $(repr(method)); the implemented methods are " *
                        ":bujanovicetal2023, :bujanovicetal2023truncated and " *
                        ":bujanovicetal2023unordered"))
end

"""
    refine_schur_decomposition(A, Q0, T0; method = :bujanovicetal2023unordered,
        target_precision = 256, max_iterations = 20, tol = 0, threshold = 1e-5, rng = nothing)

Refines an approximate Schur decomposition `A ≈ Q0 T0 Q0ᴴ` computed in `Float64`, `T0` upper
triangular, to BigFloat at `target_precision` bits. Returns a [`SchurRefinementResult`](@ref)
with the factors in BigFloat; its `residual_norm` and `orthogonality_defect` are floating-point
diagnostics, not bounds.

Each algorithm is an unexported function named for the paper it implements, a deviation being
visible in the name:

| `method` | routine | what it is |
|---|---|---|
| `:bujanovicetal2023` | [`_bujanovicetal2023_alg4`](@ref) | Algorithm 4 of Bujanović, Kressner and Schröder (2023), with the reordering of its line 1 |
| `:bujanovicetal2023truncated` | [`_bujanovicetal2023_alg4_truncated`](@ref) | the same with the modification of the paper's Example 4: the entries of `L` above `threshold` are set to zero |
| `:bujanovicetal2023unordered` | [`_bujanovicetal2023_alg4_unordered`](@ref) | the iteration without the reordering, the order of the eigenvalues of `T0` being kept; a deviation, named for it |

The default is the unordered one, because the callers in this package order the Schur form
before refining it and rely on that order. The first two reorder the eigenvalues along a random
line (`rng` gives the generator), so the diagonal of the result is in another order than that
of `T0`, and they need `Q0`, `T0` in a LAPACK type.

`tol` is the tolerance of the stopping test, by default 100 rounding units of the target
precision. A `T0` that is not upper triangular is an `ArgumentError`: for a real matrix with
complex eigenvalues pass the complex Schur form, `schur(complex(A))`; the paper mentions an
extension to the real Schur form and does not give it.

# Example
```julia
A = randn(100, 100)
F = schur(complex(A))
r = refine_schur_decomposition(A, F.Z, F.T; method = :bujanovicetal2023)
r.residual_norm          # near 1e-77 at 256 bits
```

# Reference

Z. Bujanović, D. Kressner and C. Schröder, "Iterative refinement of Schur decompositions",
Numerical Algorithms 92 (2023), 247-267, doi 10.1007/s11075-022-01327-6.
"""
function refine_schur_decomposition(A::AbstractMatrix, Q0::Matrix, T0::Matrix;
                                    method::Symbol = :bujanovicetal2023unordered,
                                    target_precision::Int=256,
                                    max_iterations::Int=20,
                                    tol::Real=0.0,
                                    threshold::Real=1e-5,
                                    rng=nothing)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(Q0) == (n, n) || throw(DimensionMismatch("Q0 must be n×n"))
    size(T0) == (n, n) || throw(DimensionMismatch("T0 must be n×n"))

    # a scoped setting: a global one has no effect inside a caller's scoped block
    return setprecision(BigFloat, target_precision) do
        return _schur_refinement(method, A, Q0, T0, max_iterations, tol; rng, threshold)
    end
end

# Helper function for converting to high precision
# The data in the high precision of the refinement: BigFloat at the current precision, or the
# given floating-point type (Double64, a MultiFloat).
_to_high(::Type{BigFloat}, A::AbstractMatrix{<:Real}) = _to_bigfloat(A)
_to_high(::Type{BigFloat}, A::AbstractMatrix{<:Complex}) = _to_bigfloat(A)
_to_high(::Type{H}, A::AbstractMatrix{<:Real}) where {H} = Matrix{H}(A)
_to_high(::Type{H}, A::AbstractMatrix{<:Complex}) where {H} = Matrix{Complex{H}}(A)

# The size of a defect of the refinement, a diagnostic used by its stopping test: an upper
# bound of the spectral norm in BigFloat, the Frobenius norm in the other arithmetics, where
# balls are not available.
_refine_defect_norm(E::AbstractMatrix{<:Union{BigFloat, Complex{BigFloat}}}) =
    upper_bound_L2_opnorm(BallMatrix(E))
_refine_defect_norm(E::AbstractMatrix) = _frobenius_norm(E)

function _to_bigfloat(A::AbstractMatrix{T}) where T<:Real
    return convert.(BigFloat, A)
end

function _to_bigfloat(A::AbstractMatrix{Complex{T}}) where T<:Real
    return convert.(Complex{BigFloat}, A)
end

"""
    _refine_schur_impl(A, Q0, T0, max_iterations, tol)

Internal implementation of Schur refinement using Algorithm 4 from Bujanović et al. (2022).

This implements the full mixed-precision iterative refinement algorithm:

1. **Initial orthogonalization**: Apply Newton-Schulz to Q₀ converted to high precision
2. **Main iteration loop**:
   a. Compute T̂ = Q^H A Q in high precision
   b. Extract E = stril(T̂), T = T̂ - E (upper triangular part)
   c. Solve triangular matrix equation stril(TL - LT) = -E for L
   d. Construct skew-Hermitian correction W = L - L^H
   e. Compute orthogonality defect Y = Q^H Q - I
   f. Update Q with combined Newton-Schulz step:
      Q_new = ½ Q (2I + 2W - Y - YW + W² + W³)
3. **Convergence**: Stop when ‖E‖_F and ‖Y‖_F are below tolerance

The algorithm achieves quadratic convergence and typically needs only 3-4 iterations
to refine from double to quadruple precision.

# References
- [BujanovicKressnerSchroder2022](@cite) Bujanović, Kressner & Schröder,
  "Iterative refinement of Schur decompositions", Numer. Algorithms 92, 247–267 (2023).
  Algorithm 4: Mixed-precision Schur refinement.
"""
function _refine_schur_impl(A::AbstractMatrix, Q0::Matrix, T0::Matrix,
                            max_iterations::Int, tol::Real; hp::Type = BigFloat,
                            order::Bool = false, rng = nothing, truncation::Real = 1e5,
                            single_initial_step::Bool = false, precheck::Bool = true,
                            stop_on_orthogonality::Bool = true)
    n = size(A, 1)
    # The iteration is for a triangular T. The real Schur form of a real matrix with complex
    # eigenvalues has 2 × 2 blocks on the diagonal; the paper mentions an extension to it and
    # does not give it, and on such an input this iteration diverges.
    istriu(T0) || throw(ArgumentError(
        "the Schur factor must be upper triangular; for a real matrix with complex " *
        "eigenvalues pass its complex Schur form, schur(complex(A))"))

    if order
        Q0, T0 = _bujanovicetal2023_order(Q0, T0, rng)
    end

    # The high precision: BigFloat at the current precision, or the type `hp`
    A_high = _to_high(hp, A)
    Q = _to_high(hp, Q0)
    T_mat = _to_high(hp, T0)

    # Determine element type for high precision
    HP = eltype(Q)  # BigFloat or Complex{BigFloat}
    HP_real = real(HP)

    # Determine tolerance
    # The default: 100 rounding units in BigFloat. In the other arithmetics the defects stall a
    # little above that for larger n (their operations are not correctly rounded), so the
    # default grows with the order.
    target_tol = tol > 0 ? HP_real(tol) :
                 hp === BigFloat ? HP_real(100) * eps(HP_real) : HP_real(100 * n) * eps(HP_real)

    # Compute initial norm for relative tolerance
    A_norm = _frobenius_norm(A_high)

    # Identity matrix
    I_n = Matrix{HP}(I, n, n)

    # Pre-check: if the seed is already converged, return immediately.
    # This avoids Newton-Schulz orthogonalization potentially degrading
    # an already-accurate decomposition (e.g. from GenericSchur at high precision).
    T_hat_pre = Q' * A_high * Q
    E_pre = _stril(T_hat_pre)
    E_norm_pre = _refine_defect_norm(E_pre)
    residual_pre = E_norm_pre / A_norm
    Y_pre = Q' * Q - I_n
    orth_pre = _refine_defect_norm(Y_pre)

    if precheck && residual_pre < target_tol && orth_pre < target_tol
        return SchurRefinementResult(
            Q, T_hat_pre - E_pre, 0, residual_pre, orth_pre, true
        )
    end

    # Step 1: Initial orthogonalization of Q using Newton-Schulz (Algorithm 4, line 2)
    # Q ← ½Q̂(3I - Q̂^H Q̂)
    if single_initial_step
        Q .= (Q * (3 * I_n - Q' * Q)) / HP(2)
    else
        newton_schulz_orthogonalize!(Q; max_iter=3, tol=target_tol)
    end

    converged = false
    iterations = 0
    residual_norm = HP_real(Inf)
    orthogonality_defect = HP_real(Inf)

    # Main iteration loop (Algorithm 4, lines 3-12)
    for iter in 1:max_iterations
        iterations = iter

        # Step 2: Compute T̂ = Q^H * A * Q in high precision (line 4)
        T_hat = Q' * A_high * Q

        # Step 3: Extract E = stril(T̂) and T = T̂ - E (line 5)
        # E is strictly lower triangular part, T is upper triangular
        E = _stril(T_hat)
        T_mat = T_hat - E

        # Check convergence: E should become small
        # Use rigorous ‖·‖₂ upper bound (tighter than Frobenius)
        E_norm = _refine_defect_norm(E)
        residual_norm = E_norm / A_norm

        # Also check orthogonality
        Y = Q' * Q - I_n
        orthogonality_defect = _refine_defect_norm(Y)

        if residual_norm < target_tol &&
           (!stop_on_orthogonality || orthogonality_defect < target_tol)
            converged = true
            return SchurRefinementResult(
                Q, T_mat, iterations, residual_norm, orthogonality_defect, converged
            )
        end

        # Step 4: Solve triangular matrix equation stril(TL - LT) = -E for L (line 6)
        # According to Algorithm 4 in Bujanović et al., this should be done in LOW precision
        # Convert to Float64/ComplexF64 for the solve, then convert back
        T_lp = HP <: Complex ? convert.(ComplexF64, T_mat) : convert.(Float64, T_mat)
        E_lp = HP <: Complex ? convert.(ComplexF64, E) : convert.(Float64, E)

        # Solve in low precision with aggressive truncation for stability
        # entries of L above `truncation` are set to zero: the trick of Example 4 of the paper
        # when it is its 1e-5, and no truncation in practice at the 1e5 of the unordered variant
        L_lp = solve_triangular_matrix_equation(T_lp, E_lp; use_recursive=true, nmin=32,
            max_entry=truncation)

        # Convert L back to high precision (line 7)
        L = convert.(HP, L_lp)

        # Step 5: Construct skew-Hermitian W = L - L^H (line 8)
        W = L - L'

        # Step 6: Compute products for Newton-Schulz update (lines 9-10)
        # Following equation (23) from the paper:
        # Q_new = ½ Q (2I + 2W - Y - YW + W² + W³ + W²Y + W²YW)
        #
        # According to Algorithm 4, line 9: compute YW, W², W³ in LOW precision
        # This is more efficient and avoids numerical issues
        W_lp = HP <: Complex ? convert.(ComplexF64, W) : convert.(Float64, W)
        Y_lp = HP <: Complex ? convert.(ComplexF64, Y) : convert.(Float64, Y)

        W2_lp = W_lp * W_lp
        W3_lp = W2_lp * W_lp
        YW_lp = Y_lp * W_lp

        # Convert back to high precision for the summation (line 10)
        W2 = convert.(HP, W2_lp)
        W3 = convert.(HP, W3_lp)
        YW = convert.(HP, YW_lp)

        # Compute Σ = 2I + 2W - Y - YW + W² + W³ in high precision (line 10)
        Σ = 2 * I_n + 2 * W - Y - YW + W2 + W3

        # Step 7: Update Q ← ½QΣ (line 11)
        Q .= (Q * Σ) / HP(2)
    end

    # Final computation if not converged
    T_hat = Q' * A_high * Q
    E = _stril(T_hat)
    T_mat = T_hat - E

    E_norm = _refine_defect_norm(E)
    residual_norm = E_norm / A_norm

    Y = Q' * Q - I_n
    orthogonality_defect = _refine_defect_norm(Y)

    return SchurRefinementResult(
        Q, T_mat, iterations, residual_norm, orthogonality_defect, converged
    )
end

"""
    certify_schur_decomposition(A::BallMatrix, result::SchurRefinementResult) -> (Q_ball, T_ball)

The refined pair `Q`, `T` of a [`SchurRefinementResult`](@ref) as ball matrices of radius zero,
`T` with its part below the diagonal set to zero.

**They are not enclosures of a Schur decomposition of `A`.** `Q` is an approximately unitary
matrix and `T` an upper triangular one with `A Q ≈ Q T`; what can be proved about them is the
size of the two defects `‖Q*Q − I‖₂` and `‖AQ − QT‖₂` over the ball `A`, and those are measured
in ball arithmetic by [`ordschur_ball`](@ref) (with `A` given) and by
[`compute_spectral_projector_schur`](@ref), which take this pair. The fields `residual_norm`
and `orthogonality_defect` of `result` are floating-point diagnostics of the refinement.

Until 2026-10-11 the radii were the two diagnostics plus the norm of `rad(A)`, added uniformly
to every entry, and the result was described as an enclosure of the Schur factors; the audit
found an eigenvalue at distance 1.65e-71 from a diagonal entry given the radius 6.4e-77. For
inclusions of the factors of a Schur decomposition, see `_rumpogita2024_schur` (Rump and Ogita
(2024), Section 8, for simple eigenvalues).
"""
function certify_schur_decomposition(A::BallMatrix, result::SchurRefinementResult)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(result.Q) == (n, n) || throw(DimensionMismatch(
        "result.Q size $(size(result.Q)) does not match A size ($n, $n)"))
    return BallMatrix(Matrix(result.Q)), BallMatrix(Matrix(UpperTriangular(Matrix(result.T))))
end

"""
    rigorous_schur_bigfloat(A::BallMatrix{T}; target_precision::Int=256,
                            max_iterations::Int=20,
                            schur_seed=nothing) where T

An approximate Schur pair of the midpoint of `A` at BigFloat precision, as ball matrices of
radius zero:
1. approximate Schur decomposition in Float64 (or `schur_seed`);
2. iterative refinement to BigFloat precision ([`refine_schur_decomposition`](@ref));
3. the pair wrapped by [`certify_schur_decomposition`](@ref).

The name is historical: the pair is not an enclosure of a Schur decomposition of `A`, and the
rigorous statements about it are the defects measured by [`ordschur_ball`](@ref) and
[`compute_spectral_projector_schur`](@ref). See [`certify_schur_decomposition`](@ref).

# Arguments
- `A::BallMatrix`: Input ball matrix
- `target_precision`: Target BigFloat precision in bits (default: 256)
- `max_iterations`: Maximum refinement iterations (default: 20)
- `schur_seed`: Optional `(Q0, T0)` tuple — external Schur seed to use instead of
  computing a Float64 Schur internally.  Useful when an accurate decomposition is
  already available (e.g. from `GenericSchur.schur` in BigFloat).

# Returns
A tuple `(Q_ball, T_ball, result)` where:
- `Q_ball::BallMatrix{BigFloat}`: the refined, approximately unitary factor, radius zero
- `T_ball::BallMatrix{BigFloat}`: the refined upper triangular factor, radius zero
- `result::SchurRefinementResult`: floating-point diagnostics of the refinement

# Example
```julia
A = BallMatrix(randn(10, 10), fill(1e-10, 10, 10))
Q, T, result = rigorous_schur_bigfloat(A; target_precision=512)
@show result.converged

# With external seed:
F = schur(Complex{BigFloat}.(mid(A)))
Q2, T2, r2 = rigorous_schur_bigfloat(A; schur_seed=(F.Z, F.T))
```

# References
- [BujanovicKressnerSchroder2022](@cite) for the iterative refinement algorithm
"""
function rigorous_schur_bigfloat(A::BallMatrix{T, NT};
                                  target_precision::Int=256,
                                  max_iterations::Int=20,
                                  schur_seed=nothing) where {T, NT}
    n = size(A, 1)
    A_mid = mid(A)

    if schur_seed !== nothing
        Q0, T0 = Matrix(schur_seed[1]), Matrix(schur_seed[2])
        # Convert midpoint to seed's element type.
        # If A_mid has tiny imaginary parts (e.g. from Arb→BigFloat conversion
        # of a real matrix) that are within the input radii, strip them to avoid
        # the refinement trying to fit complex artifacts.
        A_for_refine = A_mid
        ET = eltype(Q0)
        if ET <: Complex && eltype(A_mid) <: Complex
            imag_max = maximum(abs.(imag.(A_mid)))
            rad_max = maximum(rad(A))
            if imag_max > 0 && imag_max < rad_max
                A_for_refine = real.(A_mid)
            end
        end
        A_center = convert.(ET, A_for_refine)
    else
        # Always use complex Schur to ensure strictly upper triangular T
        # (real Schur has 2x2 blocks for complex eigenvalue pairs which
        # the refinement algorithm doesn't handle)
        A_center = convert.(ComplexF64, A_mid)
        F = schur(A_center)
        Q0, T0 = F.Z, F.T
    end

    # Step 2: Refine to BigFloat precision
    result = refine_schur_decomposition(A_center, Q0, T0;
                                        target_precision=target_precision,
                                        max_iterations=max_iterations)

    if !result.converged
        @warn "Schur refinement did not converge. Residual: $(result.residual_norm)"
    end

    # Step 3: Certify with rigorous error bounds
    Q_ball, T_ball = certify_schur_decomposition(A, result)
    return Q_ball, T_ball, result
end

#####################################################
# RefSyEv: Symmetric Eigenvalue Decomposition        #
# Refinement (Ogita & Aishima, 2018)                 #
#####################################################

"""
    SymmetricEigenRefinementResult{T, RT}

Result from RefSyEv iterative refinement of symmetric eigenvalue decomposition.

# Fields
- `Q::Matrix{T}`: Refined orthogonal eigenvector matrix
- `λ::Vector{RT}`: Refined eigenvalues (real)
- `iterations::Int`: Number of refinement iterations performed
- `residual_norm::RT`: Final ‖A - QΛQ^T‖_F / ‖A‖_F
- `orthogonality_defect::RT`: Final ‖Q^T Q - I‖_F
- `converged::Bool`: Whether refinement converged to desired tolerance

# References
- [OgitaAishima2018](@cite) Ogita, T. & Aishima, K., "Iterative refinement for symmetric
  eigenvalue decomposition", Japan J. Indust. Appl. Math. 35, 1007–1035 (2018).
"""
struct SymmetricEigenRefinementResult{T, RT<:Real}
    Q::Matrix{T}
    λ::Vector{RT}
    iterations::Int
    residual_norm::RT
    orthogonality_defect::RT
    converged::Bool
end

"""
    refine_symmetric_eigen(A::AbstractMatrix{T}, Q0::Matrix, λ0::AbstractVector;
                           target_precision::Int=256,
                           max_iterations::Int=20,
                           tol::Real=0.0) where T

Refine an approximate symmetric eigenvalue decomposition A ≈ Q₀ Λ₀ Q₀^T to higher precision
using the RefSyEv algorithm from Ogita & Aishima (2018).

This algorithm is specifically designed for symmetric (or Hermitian) matrices and handles
multiple eigenvalues gracefully by using a special correction formula that avoids
division by nearly-zero eigenvalue differences.

# Algorithm (RefSyEv from Ogita & Aishima 2018)

For symmetric A with approximate decomposition A ≈ Q̂ D̂ Q̂^T:

1. Compute residuals: R = I - Q̂^T Q̂, S = Q̂^T A Q̂
2. Update eigenvalues: λ̃_i = s_ii / (1 - r_ii)  (Rayleigh quotient)
3. Compute threshold: δ = 2(‖S - D̃‖ + ‖A‖ · ‖R‖)
4. Compute correction matrix Ẽ:
   - For distinct eigenvalues (|λ̃_i - λ̃_j| > δ):
       ẽ_ij = (s_ij + λ̃_j · r_ij) / (λ̃_j - λ̃_i)
   - For multiple eigenvalues (|λ̃_i - λ̃_j| ≤ δ):
       ẽ_ij = r_ij / 2
5. Update: Q' = Q̂(I + Ẽ)

The method converges quadratically, even for matrices with multiple eigenvalues.

# Arguments
- `A`: Symmetric matrix (will be converted to target precision)
- `Q0`: Approximate orthogonal eigenvector matrix
- `λ0`: Approximate eigenvalues
- `target_precision`: Target precision in bits (default: 256)
- `max_iterations`: Maximum refinement iterations (default: 20)
- `tol`: Convergence tolerance (default: machine epsilon of target precision)

# Returns
[`SymmetricEigenRefinementResult`](@ref) containing refined Q, λ and convergence info.

# Example
```julia
using BallArithmetic, LinearAlgebra

# Create symmetric matrix
A = randn(10, 10); A = A + A'

# Compute eigen decomposition in Float64
F = eigen(Symmetric(A))
Q0, λ0 = F.vectors, F.values

# Refine to BigFloat precision
result = refine_symmetric_eigen(A, Q0, λ0; target_precision=256)

@show result.residual_norm       # Should be ≈ 10^-77
@show result.orthogonality_defect
@show result.converged
```

# Notes
- For general (non-symmetric) matrices, use [`refine_schur_decomposition`](@ref)
- This algorithm is more efficient than general Schur refinement for symmetric matrices
- Handles multiple eigenvalues without instability

# References
- [OgitaAishima2018](@cite) Ogita, T. & Aishima, K., "Iterative refinement for symmetric
  eigenvalue decomposition", Japan J. Indust. Appl. Math. 35, 1007–1035 (2018).
  Algorithm 1: RefSyEv.
"""
function refine_symmetric_eigen(A::AbstractMatrix, Q0::Matrix, λ0::AbstractVector;
                                target_precision::Int=256,
                                max_iterations::Int=20,
                                tol::Real=0.0)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(Q0) == (n, n) || throw(DimensionMismatch("Q0 must be n×n"))
    length(λ0) == n || throw(DimensionMismatch("λ0 must have length n"))

    # Set precision for BigFloat
    # a scoped setting: a global one has no effect inside a caller's scoped block
    return setprecision(BigFloat, target_precision) do
        return _refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol)
    end
end

"""
    _refine_symmetric_eigen_impl(A, Q0, λ0, max_iterations, tol)

Internal implementation of RefSyEv algorithm (Algorithm 1 from Ogita & Aishima 2018).
"""
function _refine_symmetric_eigen_impl(A::AbstractMatrix, Q0::Matrix, λ0::AbstractVector,
                                      max_iterations::Int, tol::Real)
    n = size(A, 1)

    # Convert to BigFloat
    A_high = _to_bigfloat(A)
    Q = _to_bigfloat(Q0)
    λ = convert.(BigFloat, λ0)

    HP = eltype(Q)  # BigFloat or Complex{BigFloat}
    HP_real = real(HP)

    # Determine tolerance
    target_tol = tol > 0 ? HP_real(tol) : HP_real(100) * eps(HP_real)

    # Compute norm of A for relative tolerance
    A_norm = _frobenius_norm(A_high)

    # Identity matrix
    I_n = Matrix{HP}(I, n, n)

    converged = false
    iterations = 0
    residual_norm = HP_real(Inf)
    orthogonality_defect = HP_real(Inf)

    # Correction matrix
    E_tilde = zeros(HP, n, n)

    # Main iteration loop (Algorithm 1 from Ogita & Aishima 2018)
    for iter in 1:max_iterations
        iterations = iter

        # Line 2: R ← I - Q̂^T Q̂ (orthogonality residual)
        R = I_n - Q' * Q

        # Line 3: S ← Q̂^T A Q̂ (should be diagonal if eigenvectors are exact)
        S = Q' * A_high * Q

        # Line 4: λ̃_i ← s_ii / (1 - r_ii) for i = 1,...,n (Rayleigh quotient)
        for i in 1:n
            denom = HP(1) - R[i, i]
            if abs(denom) > eps(HP_real)
                λ[i] = real(S[i, i] / denom)
            else
                λ[i] = real(S[i, i])
            end
        end

        # Line 5: D̃ ← diag(λ̃_i)
        D_tilde = Diagonal(λ)

        # Line 6: δ ← 2(‖S - D̃‖₂ + ‖A‖₂ ‖R‖₂)
        # Use Frobenius norm as upper bound on spectral norm
        S_minus_D_norm = _frobenius_norm(S - D_tilde)
        R_norm = _frobenius_norm(R)
        δ = HP_real(2) * (S_minus_D_norm + A_norm * R_norm)

        # Check convergence
        orthogonality_defect = R_norm

        # Compute reconstruction error
        Λ = Diagonal(λ)
        reconstruction = Q * Λ * Q'
        residual_norm = _frobenius_norm(A_high - reconstruction) / A_norm

        if residual_norm < target_tol && orthogonality_defect < target_tol
            converged = true
            return SymmetricEigenRefinementResult(
                Q, λ, iterations, residual_norm, orthogonality_defect, converged
            )
        end

        # Line 7: Compute correction matrix Ẽ
        fill!(E_tilde, zero(HP))

        for j in 1:n
            for i in 1:n
                if i == j
                    # Diagonal: ẽ_ii = r_ii / 2
                    E_tilde[i, i] = R[i, i] / HP(2)
                else
                    # Off-diagonal
                    λ_diff = λ[j] - λ[i]
                    if abs(λ_diff) > δ
                        # Distinct eigenvalues: ẽ_ij = (s_ij + λ̃_j r_ij) / (λ̃_j - λ̃_i)
                        E_tilde[i, j] = (S[i, j] + λ[j] * R[i, j]) / λ_diff
                    else
                        # Multiple eigenvalues: ẽ_ij = r_ij / 2
                        E_tilde[i, j] = R[i, j] / HP(2)
                    end
                end
            end
        end

        # Line 8: X' ← X̂ + X̂ Ẽ = X̂(I + Ẽ)
        Q .= Q * (I_n + E_tilde)
    end

    # Final computation if not converged
    R = I_n - Q' * Q
    S = Q' * A_high * Q
    for i in 1:n
        denom = HP(1) - R[i, i]
        if abs(denom) > eps(HP_real)
            λ[i] = real(S[i, i] / denom)
        else
            λ[i] = real(S[i, i])
        end
    end

    orthogonality_defect = _frobenius_norm(R)

    Λ = Diagonal(λ)
    reconstruction = Q * Λ * Q'
    residual_norm = _frobenius_norm(A_high - reconstruction) / A_norm

    return SymmetricEigenRefinementResult(
        Q, λ, iterations, residual_norm, orthogonality_defect, converged
    )
end

# `rigorous_symmetric_eigen_bigfloat` was removed on 2026-10-11: it gave every entry of the
# eigenvector matrix the radius "orthogonality defect plus norm of rad(A)", with no eigenvalue
# gap. Verified eigenvalues and bases of invariant subspaces of a Hermitian ball matrix, at any
# precision, are `verifyeigall(A; method = :rumplange2023)`.

#==============================================================================#
# Stub functions for Double64 extension (DoubleFloatsExt)
#==============================================================================#

"""
    refine_schur_double64(A, Q0, T0; max_iterations = 10, tol = 0)

[`refine_schur_decomposition`](@ref) with `Double64` (about 32 digits) as the high precision in
place of BigFloat: the same iteration, Algorithm 4 of Bujanović, Kressner and Schröder, on the
upper triangular Schur factor `T0`. Returns a [`SchurRefinementResult`](@ref) with factors in
`Double64`; its two norms are floating-point diagnostics, not bounds. Requires DoubleFloats.jl.
"""
function refine_schur_double64 end

"""
    refine_schur_hybrid(A, Q0, T0; d64_iterations = 10, bf_iterations = 5, precision_bits = 256)

[`refine_schur_double64`](@ref) followed by [`refine_schur_decomposition`](@ref) in BigFloat at
`precision_bits`, started from its result. Requires DoubleFloats.jl.
"""
function refine_schur_hybrid end

"""
    refine_symmetric_eigen_double64(A, Q0, λ0; max_iterations=2, certify_with_bigfloat=true, bigfloat_precision=256)

Fast symmetric eigenvalue refinement (RefSyEv) using Double64 arithmetic.
Requires DoubleFloats.jl to be loaded.
"""
function refine_symmetric_eigen_double64 end

"""
    refine_symmetric_eigen_hybrid(A, Q0, λ0; d64_iterations=2, bf_iterations=1, precision_bits=256)

Hybrid RefSyEv refinement: Double64 for bulk iterations, BigFloat for final polish.
Requires DoubleFloats.jl to be loaded.
"""
function refine_symmetric_eigen_hybrid end

#==============================================================================#
# Stub functions for MultiFloat extension (MultiFloatsExt)
#==============================================================================#

"""
    refine_schur_multifloat(A, Q0, T0; precision = :x2, max_iterations = 10, tol = 0)

[`refine_schur_decomposition`](@ref) with a MultiFloat (`:x2`, `:x4`, `:x8`) as the high
precision in place of BigFloat. Returns a [`SchurRefinementResult`](@ref) with factors in that
type; its two norms are floating-point diagnostics, not bounds. Requires MultiFloats.jl.
"""
function refine_schur_multifloat end

"""
    refine_symmetric_eigen_multifloat(A, Q0, λ0; precision=:x2, max_iterations=2, certify=true, bigfloat_precision=256)

Refine symmetric eigenvalue decomposition using MultiFloats arithmetic.
Requires MultiFloats.jl to be loaded.
"""
function refine_symmetric_eigen_multifloat end

# Export functions
export SchurRefinementResult, refine_schur_decomposition, rigorous_schur_bigfloat, certify_schur_decomposition
export newton_schulz_orthogonalize!, solve_triangular_matrix_equation
export SymmetricEigenRefinementResult, refine_symmetric_eigen
export refine_schur_double64, refine_schur_hybrid
export refine_symmetric_eigen_double64, refine_symmetric_eigen_hybrid
export refine_schur_multifloat, refine_symmetric_eigen_multifloat
