# Spectral projectors and spectral coefficients from an approximate Schur pair.
#
# The scheme: an ordered Schur pair (Q, T̃) of the midpoint of A, computed in floating point; its
# defects against A measured in ball arithmetic; the projector of the triangular T̃ enclosed
# through a Sylvester equation; and the distance to the projector of A bounded from the defects
# and a bound of the resolvent of A on a contour, supplied by the caller.

"""
    SchurSpectralProjectorResult{T, NT}

Returned by [`compute_spectral_projector_schur`](@ref) and
[`compute_spectral_projector_hermitian`](@ref).

# Fields
- `projector`: when `projector_error_bound` is finite, an enclosure of the spectral projector `P_A`
  of every matrix of the ball `A`; otherwise an enclosure of the computed
  `P_c = Q P_T̃ Q*` only, which is not `P_A`
- `schur_projector`: enclosure of `P_T̃ = [I Y; 0 0]`, the spectral projector of `T̃` for its first
  `k` diagonal entries
- `coupling_matrix`: enclosure of `Y`, the solution of `T₁₁Y − YT₂₂ = T₁₂`
- `eigenvalue_separation`: `min |t_ii − t_jj|` over `i ≤ k < j`, in floating point (a diagnostic)
- `projector_norm`: upper bound of `‖P‖₂` over the returned ball
- `idempotency_defect`: upper bound of `‖P² − P‖₂` over the returned ball (zero when not asked)
- `schur_basis`, `schur_form`: the pair `Q`, `T̃` used, the latter exactly upper triangular
- `cluster_indices`: `1:k`, the positions of the selected eigenvalues in `T̃`
- `orth_defect`: upper bound of `‖Q*Q − I‖₂`
- `fact_defect`: upper bound of `‖AQ − QT̃‖₂` over the ball `A`
- `projector_error_bound`: upper bound of `‖P_A − P_c‖₂` from
  [`spectral_projector_error_bound`](@ref); `Inf` when no resolvent bound was given
"""
struct SchurSpectralProjectorResult{T, NT}
    projector::BallMatrix{T, NT}
    schur_projector::BallMatrix{T, NT}
    coupling_matrix::Union{BallMatrix{T, NT}, Nothing}
    eigenvalue_separation::T
    projector_norm::T
    idempotency_defect::T
    schur_basis::Matrix{NT}
    schur_form::Matrix{NT}
    cluster_indices::UnitRange{Int}
    orth_defect::T
    fact_defect::T
    projector_error_bound::T
end

# x as a float of type RT, not below x
function _float_up(::Type{RT}, x::Real) where {RT <: AbstractFloat}
    y = RT(x)
    return y < x ? nextfloat(y) : y
end

# The ordered pair used for the cluster `indices`: (Q, T̃, k) with the selected diagonal entries
# first and T̃ exactly upper triangular. Everything here is floating point; what the pair
# satisfies is measured by `_schur_pair_defects`.
function _ordered_schur_pair(A::BallMatrix, indices, schur_data, ordered_schur_data)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("A must be square"))
    k = length(indices)
    k >= 1 || throw(ArgumentError("Cluster must contain at least one index"))
    k < n || throw(ArgumentError("Cluster cannot contain all indices (would be identity)"))
    all(i -> 1 <= i <= n, indices) || throw(ArgumentError("Indices must be in 1:$n"))
    allunique(indices) || throw(ArgumentError("Indices must be distinct"))

    point(M) = Matrix(M isa BallMatrix ? mid(M) : M)
    if ordered_schur_data !== nothing
        Q, Tm, k_ord = ordered_schur_data
        k_ord == k ||
            throw(ArgumentError("ordered_schur_data k=$k_ord does not match cluster size $k"))
        Q, Tm = point(Q), point(Tm)
    else
        if schur_data !== nothing
            Q, Tm = point(schur_data[1]), point(schur_data[2])
        else
            F = schur(mid(A))
            Q, Tm = Matrix(F.Z), Matrix(F.T)
        end
        istriu(Tm) || throw(ArgumentError(
            "the Schur form is not upper triangular (a real matrix with complex " *
            "eigenvalues): pass the matrix, or schur_data, in complex form"))
        if collect(indices) != collect(1:k)
            select = falses(n)
            select[collect(indices)] .= true
            Q, Tm, _ = ordschur_bigfloat(Tm, Q, select)
            Q, Tm = Matrix(Q), Matrix(Tm)
        end
    end
    size(Q) == (n, n) && size(Tm) == (n, n) ||
        throw(DimensionMismatch("the Schur factors must be $n×$n"))
    # what the floating-point reordering left below the diagonal goes into the defect
    return Q, Matrix(UpperTriangular(Tm)), k
end

# upper bounds of ‖Q*Q − I‖₂ and of ‖AQ − QT̃‖₂ over the ball A
function _schur_pair_defects(A::BallMatrix, Q::AbstractMatrix, T̃::AbstractMatrix)
    bQ = BallMatrix(Q)
    bA = _ball_as(eltype(Q), A)
    return upper_bound_L2_opnorm(bQ' * bQ - I), upper_bound_L2_opnorm(bA * bQ - bQ * BallMatrix(T̃))
end

# enclosure of Y with T₁₁Y − YT₂₂ = T₁₂ for the exactly triangular T̃, and the gap diagnostic.
# `triangular_sylvester_miyajima_enclosure` encloses X with T₂₂*X − XT₁₁* = T₁₂*, and Y = −X*.
function _coupling_enclosure(T̃::AbstractMatrix, k::Int, sylvester_fallback::Symbol)
    n = size(T̃, 1)
    separation = minimum(abs(T̃[i, i] - T̃[j, j]) for i in 1:k for j in (k + 1):n)
    X = triangular_sylvester_miyajima_enclosure(T̃, k; sylvester_fallback)
    return BallMatrix(-Matrix(adjoint(mid(X))), Matrix(transpose(rad(X)))), separation
end

# ‖P_A − P_c‖₂ from the resolvent bound, or Inf when none is given
function _projector_error(::Type{RT}, resolvent_bound, contour_radius, δ, ε) where {RT}
    resolvent_bound === nothing && contour_radius === nothing && return RT(Inf)
    (resolvent_bound === nothing || contour_radius === nothing) && throw(ArgumentError(
        "resolvent_bound and contour_radius must be given together"))
    return _float_up(RT, spectral_projector_error_bound(; resolvent_bound_A = resolvent_bound,
        contour_radius, orth_defect = δ, fact_defect = ε))
end

"""
    compute_spectral_projector_schur(A::BallMatrix, cluster_indices;
        resolvent_bound = nothing, contour_radius = nothing, verify_idempotency = true,
        schur_data = nothing, ordered_schur_data = nothing, sylvester_fallback = :direct)

The spectral projector of `A` for the eigenvalues at the positions `cluster_indices` (a range or
a vector of distinct indices) of the diagonal of its Schur form.

# What is computed, and what is proved

1. In floating point: a Schur pair of the midpoint of `A` (or `schur_data = (Q, T)`), reordered
   so that the selected eigenvalues come first (or `ordered_schur_data = (Q, T, k)`, already in
   that order). `T̃` is its upper triangle.
2. In ball arithmetic: the defects `δ ≥ ‖Q*Q − I‖₂` and `ε ≥ ‖AQ − QT̃‖₂`, the second over the
   whole ball `A`.
3. `P_T̃ = [I Y; 0 0]`, the spectral projector of the triangular `T̃` for its first `k` diagonal
   entries, with `Y` the solution of `T₁₁Y − YT₂₂ = T₁₂`, enclosed by
   [`triangular_sylvester_miyajima_enclosure`](@ref); and `P_c = Q P_T̃ Q*` in ball arithmetic.
4. With `resolvent_bound ≥ ‖(zI − A)⁻¹‖₂` on a contour of length at most `2π·contour_radius` that
   has the selected eigenvalues of `A` inside and the others outside (for instance from
   `CertifScripts.run_certification`), [`spectral_projector_error_bound`](@ref) gives
   `b ≥ ‖P_A − P_c‖₂`, and the radius of every entry of the returned projector is increased by
   `b`: the result is then an enclosure of `P_A` for every matrix of the ball `A`.

Without the two keywords step 4 is not done, `projector_error_bound` is `Inf`, and the returned
ball encloses `P_c` only. `P_c` is the projector of a nearby matrix; its distance from `P_A` is
exactly what step 4 bounds, and ball arithmetic on `Q` and `T̃` alone says nothing about it.

That the contour separates the spectrum as required is the caller's hypothesis; the bound
`γ < 1` inside `spectral_projector_error_bound` shows that `T̃` has no eigenvalue on it.

`verify_idempotency` adds the bound of `‖P² − P‖₂` over the returned ball.
"""
function compute_spectral_projector_schur(A::BallMatrix{T, NT},
        cluster_indices::Union{UnitRange{Int}, AbstractVector{Int}};
        resolvent_bound = nothing, contour_radius = nothing,
        verify_idempotency::Bool = true, schur_data = nothing, ordered_schur_data = nothing,
        sylvester_fallback::Symbol = :direct) where {T, NT}
    Q, T̃, k = _ordered_schur_pair(A, cluster_indices, schur_data, ordered_schur_data)
    return _spectral_projector_core(A, Q, T̃, k, verify_idempotency, sylvester_fallback,
        resolvent_bound, contour_radius)
end

function _spectral_projector_core(A::BallMatrix{T}, Q::AbstractMatrix, T̃::AbstractMatrix,
        k::Int, verify_idempotency::Bool, sylvester_fallback::Symbol, resolvent_bound,
        contour_radius) where {T}
    n = size(A, 1)
    CT = promote_type(eltype(Q), eltype(T̃))
    Q, T̃ = Matrix{CT}(Q), Matrix{CT}(T̃)
    δ, ε = _schur_pair_defects(A, Q, T̃)
    Y, separation = _coupling_enclosure(T̃, k, sylvester_fallback)

    P_schur_c = zeros(CT, n, n)
    P_schur_r = zeros(T, n, n)
    P_schur_c[1:k, 1:k] .= Matrix{CT}(I, k, k)
    P_schur_c[1:k, (k + 1):n] .= mid(Y)
    P_schur_r[1:k, (k + 1):n] .= rad(Y)
    P_schur = BallMatrix(P_schur_c, P_schur_r)

    bQ = BallMatrix(Q)
    P = bQ * P_schur * bQ'
    err = _projector_error(T, resolvent_bound, contour_radius, δ, ε)
    if isfinite(err)
        P = BallMatrix(Matrix(mid(P)), add_up.(rad(P), err))
    end

    projector_norm = upper_bound_L2_opnorm(P)
    idempotency_defect = verify_idempotency ? upper_bound_L2_opnorm(P * P - P) : zero(T)
    return SchurSpectralProjectorResult(P, P_schur, Y, T(separation), projector_norm,
        idempotency_defect, Q, T̃, 1:k, δ, ε, err)
end

"""
    compute_spectral_projector_hermitian(A::BallMatrix, cluster_indices::UnitRange{Int};
        resolvent_bound = nothing, contour_radius = nothing)

[`compute_spectral_projector_schur`](@ref) with the eigenvector matrix of the Hermitian midpoint
of `A` as `Q` and the diagonal matrix of its eigenvalues (in increasing order) as `T̃`;
`cluster_indices` are positions in that order. The coupling `Y` is zero, and the statement and
the role of the two keywords are those of the general routine.
"""
function compute_spectral_projector_hermitian(A::BallMatrix{T, NT},
        cluster_indices::UnitRange{Int}; resolvent_bound = nothing,
        contour_radius = nothing) where {T, NT}
    n = size(A, 1)
    F = eigen(Hermitian(Matrix(mid(A))))
    order = vcat(collect(cluster_indices), setdiff(1:n, cluster_indices))
    Q = Matrix{NT}(F.vectors[:, order])
    T̃ = Matrix{NT}(Diagonal(F.values[order]))
    k = length(cluster_indices)
    1 <= k < n || throw(ArgumentError("the cluster must be a proper nonempty subset"))
    return _spectral_projector_core(A, Q, T̃, k, true, :direct, resolvent_bound, contour_radius)
end

"""
    project_vector_spectral(v, result::SchurSpectralProjectorResult)

The product of the returned projector ball with `v` (a `BallVector`, or a vector with exact
entries), in ball arithmetic. It encloses `P_A v` when `result.projector_error_bound` is finite,
and `P_c v` otherwise.
"""
project_vector_spectral(v::BallVector, result::SchurSpectralProjectorResult) =
    result.projector * v
project_vector_spectral(v::AbstractVector, result::SchurSpectralProjectorResult) =
    result.projector * BallVector(v)

"""
    verify_spectral_projector_properties(result, A::BallMatrix; tol = 1e-10,
                                         check_commutation = false)

Whether `result.idempotency_defect < tol`, `result.projector_norm` is finite,
`result.eigenvalue_separation` is positive and, when `check_commutation` is set, the upper bound
of `‖AP − PA‖₂` over the two balls is below `tol`. These are consistency checks of the returned
ball; they do not replace `projector_error_bound`.
"""
function verify_spectral_projector_properties(result::SchurSpectralProjectorResult,
        A::BallMatrix; tol::Real = 1e-10, check_commutation::Bool = false)
    ok = result.idempotency_defect < tol && isfinite(result.projector_norm) &&
         result.eigenvalue_separation > 0
    if ok && check_commutation
        ok = upper_bound_L2_opnorm(A * result.projector - result.projector * A) < tol
    end
    return ok
end

"""
    SpectralCoefficientResult{T, NT}

Returned by [`compute_spectral_coefficient`](@ref).

# Fields
- `coefficients`: when `coefficient_error_bound` is finite, an enclosure of `Q₁* P_A v`, with `Q₁`
  the first `k` columns of `schur_basis`, for every matrix of the ball `A`; otherwise an enclosure
  of the computed `[I Y] Q* v` only
- `left_eigenvector_schur`: enclosure of the `k×n` block `[I Y]`
- `coupling_matrix`: enclosure of `Y`
- `eigenvalue_separation`: `min |t_ii − t_jj|` over `i ≤ k < j`, in floating point
- `schur_basis`, `schur_form`: the pair `Q`, `T̃` used
- `orth_defect`, `fact_defect`: upper bounds of `‖Q*Q − I‖₂` and `‖AQ − QT̃‖₂`
- `coefficient_error_bound`: what was added to the radius of every coefficient; `Inf` when no
  resolvent bound was given
"""
struct SpectralCoefficientResult{T, NT}
    coefficients::BallVector{T, NT}
    left_eigenvector_schur::BallMatrix{T, NT}
    coupling_matrix::Union{BallMatrix{T, NT}, Nothing}
    eigenvalue_separation::T
    schur_basis::Matrix{NT}
    schur_form::Matrix{NT}
    orth_defect::T
    fact_defect::T
    coefficient_error_bound::T
end

"""
    compute_spectral_coefficient(A::BallMatrix, v, cluster_indices;
        resolvent_bound = nothing, contour_radius = nothing,
        schur_data = nothing, ordered_schur_data = nothing, sylvester_fallback = :direct)

The coordinates of the spectral projection of `v` in the first `k` Schur vectors, without forming
the `n×n` projector: with the ordered pair `(Q, T̃)` and the coupling `Y` of
[`compute_spectral_projector_schur`](@ref),

    c = [I Y] Q* v,        so that        P_c v = Q₁ c,

`Q₁` being the first `k` columns of `Q`. `v` is a `BallVector` or a vector with exact entries.

With `resolvent_bound` and `contour_radius` (as in `compute_spectral_projector_schur`),
`b ≥ ‖P_A − P_c‖₂` and

    ‖Q₁* P_A v − c‖₂ ≤ ‖Q₁‖₂ ‖(P_A − P_c) v‖₂ + ‖(Q₁*Q₁ − I) c‖₂ ≤ √(1 + δ) · b · ‖v‖₂ + δ ‖c‖₂,

which is added to the radius of every coefficient: the result then encloses `Q₁* P_A v`. Without
them `coefficient_error_bound` is `Inf` and the ball encloses the computed `c` only.
"""
function compute_spectral_coefficient(A::BallMatrix{T, NT}, v::AbstractVector,
        cluster_indices; resolvent_bound = nothing, contour_radius = nothing,
        schur_data = nothing, ordered_schur_data = nothing,
        sylvester_fallback::Symbol = :direct) where {T, NT}
    n = size(A, 1)
    length(v) == n || throw(DimensionMismatch("v must have length $n"))
    Q, T̃, k = _ordered_schur_pair(A, cluster_indices, schur_data, ordered_schur_data)
    CT = promote_type(eltype(Q), eltype(T̃), v isa BallVector ? eltype(mid(v)) : eltype(v))
    Q, T̃ = Matrix{CT}(Q), Matrix{CT}(T̃)
    δ, ε = _schur_pair_defects(A, Q, T̃)
    Y, separation = _coupling_enclosure(T̃, k, sylvester_fallback)

    IY_c = zeros(CT, k, n)
    IY_r = zeros(T, k, n)
    IY_c[1:k, 1:k] .= Matrix{CT}(I, k, k)
    IY_c[1:k, (k + 1):n] .= mid(Y)
    IY_r[1:k, (k + 1):n] .= rad(Y)
    IY = BallMatrix(IY_c, IY_r)

    bv = v isa BallVector ? BallVector(Vector{CT}(mid(v)), Vector(rad(v))) :
         BallVector(Vector{CT}(v))
    c = IY * (BallMatrix(Q)' * bv)

    b = _projector_error(T, resolvent_bound, contour_radius, δ, ε)
    err = T(Inf)
    if isfinite(b)
        err = add_up(mul_up(mul_up(sqrt_up(add_up(one(T), δ)), b), upper_bound_norm(bv, 2)),
            mul_up(δ, upper_bound_norm(c, 2)))
        c = BallVector(Vector(mid(c)), add_up.(rad(c), err))
    end
    return SpectralCoefficientResult(c, IY, Y, T(separation), Q, T̃, δ, ε, err)
end
