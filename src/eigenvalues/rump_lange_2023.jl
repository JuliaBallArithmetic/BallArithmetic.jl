# Error bounds for all eigenpairs of a Hermitian matrix and all singular pairs of a rectangular
# matrix, with clusters, after
#
#   S. M. Rump and M. Lange, Fast computation of error bounds for all eigenpairs of a Hermitian
#   and all singular pairs of a rectangular matrix with emphasis on eigen- and singular value
#   clusters, J. Comput. Appl. Math. 434 (2023) 115332, doi 10.1016/j.cam.2023.115332.
#
# Each routine is named for the algorithm or the section of the paper it implements. The paper
# calls its eigenvalue algorithm `verifyeigall`, as Rump (2022) does for general matrices; here it
# is reached through `verifyeigall(A; method = :rumplange2023)`.

export VerifySvdAllResult, verifysvdall

# ---------------------------------------------------------------------------------------------
# Section 3: norms and the smallest singular value
# ---------------------------------------------------------------------------------------------

# `singmin` of Section 3: a lower bound of σ_min(X) for X with nearly orthonormal columns, by
# (3.2): ‖X*X − I‖ ≤ α < 1 gives σ_min(X) ≥ √(1 − α). Zero when α ≥ 1. The norm of the Hermitian
# matrix is bounded by the Collatz quotient of its modulus, (3.1), which is `NormBnd(·, true)`.
function _rumplange2023_singmin(X::AbstractMatrix{S}) where {S}
    T = real(S)
    Xb = BallMatrix(Matrix(X))
    α = collatz_upper_bound(Xb' * Xb - I)
    return (isfinite(α) && α < 1) ? sqrt_down(sub_down(one(T), T(α))) : zero(T)
end

# an upper bound of ‖I − X*X‖₂, the α of Lemma 3.1
function _rumplange2023_alpha(X::AbstractMatrix)
    Xb = BallMatrix(Matrix(X))
    return collatz_upper_bound(Xb' * Xb - I)
end

# upper bounds of the squared 2-norms of the columns of a ball matrix
function _column_norms2_up(E::BallMatrix{T}) where {T}
    Em, Er = mid(E), rad(E)
    out = zeros(T, size(Em, 2))
    for j in axes(Em, 2), i in axes(Em, 1)
        a = add_up(abs_up(Em[i, j]), Er[i, j])
        out[j] = add_up(out[j], mul_up(a, a))
    end
    return out
end

# The numbers x_j* (Y_1 + Y_2 + …)_j, one for each column j, as balls: the routines `norm_X2` and
# `norm_xAx` of (5.2), "using some increased precision". The sum over the rows is accumulated
# with error-free transformations and bounded as in `_accuracy_term`; `R[i, j]` is a bound of
# the entrywise error of Y_1 + Y_2 + …, which adds (|X|ᵀR)_jj.
function _rumplange2023_coldots(X::Matrix{S}, Ys::Tuple, R::AbstractMatrix{T}) where {S, T}
    p = size(X, 2)
    Xa = Matrix{S}(X')
    terms = [(Xa, Matrix{S}(Y), one(T)) for Y in Ys]
    re, im_ = _real_pairs(terms, T)
    absX = _modulus_up(X)
    absY = [_modulus_up(Matrix{S}(Y)) for Y in Ys]
    N = (S <: Complex ? 2 : 1) * size(X, 1) * length(Ys)
    m, r = Vector{S}(undef, p), Vector{T}(undef, p)
    for j in 1:p
        s, e = compensated_terms2(re, j, j)
        v = S <: Complex ? complex(s + e, sum(compensated_terms2(im_, j, j))) : s + e
        absprod, extra = zero(T), zero(T)
        for i in axes(X, 1)
            for aY in absY
                absprod = add_up(absprod, mul_up(absX[i, j], aY[i, j]))
            end
            extra = add_up(extra, mul_up(absX[i, j], R[i, j]))
        end
        acc = _accuracy_term(reshape(T[abs_up(v)], 1, 1), reshape(T[absprod], 1, 1), N, T,
            S <: Complex)[1]
        m[j], r[j] = v, add_up(acc, extra)
    end
    return m, r
end

# connected components of a symmetric relation given as a Boolean matrix, sorted by first index
function _rumplange2023_components(adj::AbstractMatrix{Bool})
    n = size(adj, 1)
    parent = collect(1:n)
    find(x) = (while parent[x] != x
        x = parent[x]
    end; x)
    for j in 1:n, i in 1:(j - 1)
        if adj[i, j] || adj[j, i]
            a, b = find(i), find(j)
            a == b || (parent[max(a, b)] = min(a, b))
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, find(i), Int[]), i)
    end
    return sort(collect(values(groups)); by = first)
end

# the distance, rounded down, from the point c to the interval [lo, hi]; negative inside it
_point_interval_gap(c::T, lo::T, hi::T) where {T} = max(sub_down(lo, c), sub_down(c, hi))

# the gap, rounded down, between two intervals; zero when they meet
_interval_gap(lo1::T, hi1::T, lo2::T, hi2::T) where {T} =
    max(zero(T), sub_down(lo2, hi1), sub_down(lo1, hi2))

# ---------------------------------------------------------------------------------------------
# Sections 4 to 6: the Hermitian eigenproblem
# ---------------------------------------------------------------------------------------------

"""
    _rumplange2023_eig(A::BallMatrix) -> (; lo, hi, clusters, vectors, basis, centers, radii)

Algorithm `verifyeigall` of Rump and Lange (2023) for a Hermitian matrix, with the refinement
`refineeig` of its Section 5 and the eigenvector bounds of its Section 6. For every Hermitian
matrix `Ã` of the ball `A`:

- **Theorem 4.2.** There is a numbering `λ_1, …, λ_n` of the eigenvalues of `Ã` with
  `λ_j ∈ [lo[j], hi[j]]`. `clusters` is a partition of `1:n`; the unions of the intervals of the
  clusters are mutually disjoint, and the union for a cluster holds exactly as many eigenvalues
  as the cluster has indices.
- **Theorem 6.2.** For each cluster `μ`, the ball matrix `vectors[:, μ]` contains an orthonormal
  basis of the invariant subspace of the eigenvalues in the intervals of `μ`.

`basis` is the floating-point eigenvector matrix `X̃`, `centers` the approximate eigenvalues `λ̃`
(midpoints of the Rayleigh quotients) and `radii` the `δ` of the algorithm before refinement.

# The algorithm

1. `X̃` are the eigenvectors of `(mid(A)* + mid(A))/2`; the Rayleigh quotients
   `ϱ_j = x̃_j*Ãx̃_j / x̃_j*x̃_j` are enclosed with the products accumulated to twice the working
   precision, and `λ̃_j` is the midpoint of the enclosure (Section 5).
2. `E ∋ ÃX̃ − X̃Λ̃`, and `δ_j = ‖E e_j‖/‖x̃_j‖`, so that `λ̃_j ± δ_j` holds an eigenvalue
   (Theorem 4.1 with `k = 1`).
3. The intervals are grouped into the connected components of "these two intersect". For a
   component `μ` with more than one index, `δ_j` is replaced for `j ∈ μ` by
   `‖E(:, μ)‖/σ_min(X̃(:, μ))` (Theorem 4.1 with `k = |μ|`), the intervals keeping their own
   midpoints, and the grouping is repeated until the partition no longer changes.
4. For a cluster of one index, with `e_j` the gap of its interval to the others,
   `|λ_j − ϱ_j| ≤ ‖Ãx̃_j − ϱ_j x̃_j‖²/(e_j‖x̃_j‖²)` (Theorem 5.1), and the interval is
   intersected with that one.
5. For a cluster `μ`, with `ε` a lower bound of the distance from the `λ̃_j`, `j ∈ μ`, to the
   intervals outside `μ`, there is `Y` with columns in the invariant subspace and
   `‖X̃(:, μ) − Y‖ ≤ τ := ‖E(:, μ)‖/ε` (Lemma 6.1), and then an orthonormal basis `Q` of that
   subspace with `‖Q − X̃(:, μ)‖ ≤ ‖I − X̃(:, μ)*X̃(:, μ)‖ + √2 τ` (Lemma 3.1), which bounds
   every entry.

# Where the code departs from the paper's listing

Three places, each toward the statement of the lemma used:
- the loop of step 3 stops when the partition is the one for which the `δ` were computed, where
  the listing compares the number of components;
- in step 5 the distance is measured from `λ̃_j`, the value in the residual `E`, where the
  listing takes the midpoint of the refined interval; and one `ε` is used for the whole cluster,
  as (6.3) and the sentence after (6.4) say, where the listing divides column by column;
- when all eigenvalues form one cluster the listing returns radius zero for `X̃`; here the radius
  is `‖I − X̃*X̃‖`, Lemma 3.1 with `Y = X̃`, since `X̃` is orthonormal only up to rounding.

# Reference

S. M. Rump and M. Lange, *Fast computation of error bounds for all eigenpairs of a Hermitian and
all singular pairs of a rectangular matrix with emphasis on eigen- and singular value clusters*,
J. Comput. Appl. Math. **434** (2023) 115332, doi 10.1016/j.cam.2023.115332, Sections 3 to 6.
"""
function _rumplange2023_eig(A::BallMatrix{T}) where {T}
    n = size(A, 1)
    n == size(A, 2) || throw(ArgumentError("_rumplange2023_eig: A must be square"))
    Am, Ar = Matrix(mid(A)), rad(A)
    S = eltype(Am)
    Xs = Matrix{S}(eigen(Hermitian((Am' + Am) / 2)).vectors)
    absX = _modulus_up(Xs)
    up(f) = setrounding(f, T, RoundUp)

    # ‖x̃_j‖² and the Rayleigh quotients, (5.2)
    n2m, n2r = _rumplange2023_coldots(Xs, (Xs,), zeros(T, n, n))
    n2lo = T[sub_down(real(n2m[j]), n2r[j]) for j in 1:n]
    P1, P2, RP = _two_term_product_sum(((Am, Xs, one(T)),))
    RP = up(() -> RP .+ Ar * absX)
    qm, qr = _rumplange2023_coldots(Xs, (P1, P2), RP)
    all(>(0), n2lo) || return _rumplange2023_eig_failed(Xs, T)
    rho = [Ball(real(qm[j]), qr[j]) / Ball(real(n2m[j]), n2r[j]) for j in 1:n]
    lam = T[mid(r) for r in rho]
    rhor = T[rad(r) for r in rho]
    all(isfinite, lam) || return _rumplange2023_eig_failed(Xs, T)

    # E ∋ ÃX̃ − X̃Λ̃
    Eb = _accurate_product_sum(((Am, Xs, one(T)), (Xs, Matrix{S}(Diagonal(lam)), -one(T))))
    E = BallMatrix(mid(Eb), up(() -> rad(Eb) .+ Ar * absX))
    e2 = _column_norms2_up(E)
    normE = sqrt_up.(e2)
    delta = T[div_up(normE[j], sqrt_down(n2lo[j])) for j in 1:n]

    # the clusters, Algorithm verifyeigall of Section 4
    clusters = [[j] for j in 1:n]
    lo, hi = sub_down.(lam, delta), add_up.(lam, delta)
    for _ in 1:(n + 1)
        lo, hi = sub_down.(lam, delta), add_up.(lam, delta)
        new = _rumplange2023_components(Bool[lo[i] <= hi[j] && lo[j] <= hi[i] for i in 1:n, j in 1:n])
        new == clusters && break
        clusters = new
        for v in clusters
            length(v) > 1 || continue
            s = _rumplange2023_singmin(Xs[:, v])
            nE = collatz_upper_bound_L2_opnorm(E[:, v])
            normE[v] .= nE
            delta[v] .= s > 0 ? div_up(T(nE), s) : T(Inf)
        end
    end
    lo, hi = sub_down.(lam, delta), add_up.(lam, delta)
    final = _rumplange2023_components(Bool[lo[i] <= hi[j] && lo[j] <= hi[i] for i in 1:n, j in 1:n])
    if final != clusters                  # the grouping did not settle: one cluster of everything
        clusters = [collect(1:n)]
        s = _rumplange2023_singmin(Xs)
        nE = collatz_upper_bound_L2_opnorm(E)
        normE .= nE
        delta .= s > 0 ? div_up(T(nE), s) : T(Inf)
        lo, hi = sub_down.(lam, delta), add_up.(lam, delta)
    end
    radii = copy(delta)

    # refineeig, Section 5
    for v in clusters
        length(v) == 1 || continue
        j = v[1]
        gap = minimum((_interval_gap(lo[j], hi[j], lo[i], hi[i]) for i in 1:n if i != j);
            init = T(Inf))
        (gap > 0 && isfinite(gap)) || continue
        res = add_up(sqrt_up(div_up(e2[j], n2lo[j])), rhor[j])
        rr = add_up(div_up(mul_up(res, res), gap), rhor[j])
        lo[j] = max(lo[j], sub_down(lam[j], rr))
        hi[j] = min(hi[j], add_up(lam[j], rr))
    end

    # the eigenvectors, Section 6
    rX = Vector{T}(undef, n)
    sqrt2 = sqrt_up(T(2))
    for v in clusters
        ε = T(Inf)
        for j in v, i in 1:n
            i in v && continue
            ε = min(ε, _point_interval_gap(lam[j], lo[i], hi[i]))
        end
        τ = length(v) == n ? (all(isfinite, delta) ? zero(T) : T(Inf)) :
            (ε > 0 ? div_up(T(normE[v[1]]), ε) : T(Inf))
        α = if length(v) == 1
            j = v[1]
            max(sub_up(one(T), n2lo[j]), sub_up(add_up(real(n2m[j]), n2r[j]), one(T)))
        else
            T(_rumplange2023_alpha(Xs[:, v]))
        end
        rX[v] .= add_up(α, mul_up(sqrt2, τ))
    end
    vectors = BallMatrix(Xs, repeat(reshape(rX, 1, n), n, 1))
    return (; lo, hi, clusters, vectors, basis = Xs, centers = lam, radii)
end

# nothing is proved: every interval is the whole line
_rumplange2023_eig_failed(Xs::Matrix{S}, ::Type{T}) where {S, T} = (n = size(Xs, 1);
(; lo = fill(T(-Inf), n), hi = fill(T(Inf), n), clusters = [collect(1:n)],
    vectors = BallMatrix(Xs, fill(T(Inf), n, n)), basis = Xs, centers = zeros(T, n),
    radii = fill(T(Inf), n)))

"""
    _rumplange2023(A::BallMatrix) -> VerifyEigAllResult

[`_rumplange2023_eig`](@ref) with its result in the fields of a [`VerifyEigAllResult`](@ref);
reached through [`verifyeigall`](@ref) with `method = :rumplange2023`. The statements are for
the Hermitian matrices of the ball `A`.

- `clusters` is the partition of Theorem 4.2; `centers[i]` and `radii[i]` describe the smallest
  interval containing the intervals of cluster `i`, which holds exactly `length(clusters[i])`
  eigenvalues; `certified` is true where that interval is bounded, and `spectrum_covered` when
  all are.
- `gershgorin_centers` and `gershgorin_radii` hold the `n` intervals `λ_j` of the theorem, one
  for each eigenvalue: every eigenvalue lies in their union and a union of `k` of them that is
  disjoint from the others holds `k` eigenvalues, which is the contract of those fields.
- `subspaces[i]` contains an orthonormal basis `Q_i` of the invariant subspace of cluster `i`
  (Theorem 6.2), and `blocks[i]` the Hermitian matrix `Q_i*AQ_i`, with `A Q_i = Q_i blocks[i]`.
- `similarity` contains the unitary matrix `Q` made of those bases, and `transformed` the matrix
  `Q*AQ`: block diagonal on the clusters, exact zeros elsewhere.
"""
function _rumplange2023(A::BallMatrix{T}) where {T}
    r = _rumplange2023_eig(A)
    n = size(A, 1)
    CT = complex(T)
    m = length(r.clusters)
    centers, radii = Vector{CT}(undef, m), Vector{T}(undef, m)
    subspaces, blocks = Vector{BallMatrix{T, CT}}(undef, m), Vector{BallMatrix{T, CT}}(undef, m)
    Ac = BallMatrix(Matrix{CT}(mid(A)), rad(A))
    X = BallMatrix(Matrix{CT}(mid(r.vectors)), rad(r.vectors))
    Tm, Tr = zeros(CT, n, n), zeros(T, n, n)
    for (i, v) in enumerate(r.clusters)
        a, b = minimum(r.lo[v]), maximum(r.hi[v])
        c = (a + b) / 2
        centers[i] = c
        radii[i] = isfinite(c) ? max(sub_up(c, a), sub_up(b, c)) : T(Inf)
        Q = X[:, v]
        subspaces[i] = Q
        M = _ball_adjoint(Q) * Ac * Q
        blocks[i] = M
        Tm[v, v] .= mid(M)
        Tr[v, v] .= rad(M)
    end
    certified = isfinite.(radii)
    gc = CT[(r.lo[j] + r.hi[j]) / 2 for j in 1:n]
    gr = T[isfinite(real(gc[j])) ? max(sub_up(real(gc[j]), r.lo[j]), sub_up(r.hi[j], real(gc[j]))) :
           T(Inf) for j in 1:n]
    return VerifyEigAllResult(r.clusters, collect(certified), centers, radii, subspaces, blocks,
        Matrix{CT}(r.basis), all(certified), 0, maximum(r.radii; init = zero(T)), gc, gr, X,
        BallMatrix(Tm, Tr))
end

# ---------------------------------------------------------------------------------------------
# Section 7: singular values and singular vectors
# ---------------------------------------------------------------------------------------------

"""
    VerifySvdAllResult{T, NT}

Outcome of [`verifysvdall`](@ref) for an `m × n` matrix, `p = min(m, n)`.

- `values::Vector{Ball{T, T}}`: `p` intervals; there is a numbering of the singular values with
  `σ_j ∈ values[j]`.
- `clusters::Vector{Vector{Int}}`: a partition of `1:p`; the unions of the intervals of the
  clusters are mutually disjoint and the union for a cluster holds exactly as many singular
  values as the cluster has indices.
- `U::BallMatrix` (`m × p`), `V::BallMatrix` (`n × p`): for each cluster `μ`, `U[:, μ]` and
  `V[:, μ]` contain orthonormal bases of the left and of the right singular subspace of the
  singular values of `μ`. A radius is `Inf` where this was not proved.
"""
struct VerifySvdAllResult{T, NT}
    values::Vector{Ball{T, T}}
    clusters::Vector{Vector{Int}}
    U::BallMatrix{T, NT}
    V::BallMatrix{T, NT}
end

"""
    verifysvdall(A::BallMatrix; kappa = 0) -> VerifySvdAllResult

Verified inclusions of all singular values and of the left and right singular subspaces of an
`m × n` matrix: Algorithm `verifysvdall` of Rump and Lange (2023), with its refinement
`refinesvd` and the singular vector bounds of its Section 7. The statements, Theorem 7.3 there,
are those of [`VerifySvdAllResult`](@ref) and hold for every matrix of the ball `A`. `kappa ≥ 0`
is the paper's threshold: singular value inclusions closer than `kappa` relative to their size
are put in one cluster, which widens their intervals and tightens their subspaces.

# The algorithm

For `m ≥ n`, with `X̃ Σ̃ Ỹ*` a floating-point economy-size decomposition of the midpoint and
`B = [0 A*; A 0]`, whose eigenvalues are `±σ_j` and `m − n` zeros:

1. `σ̃_j` is the midpoint of an enclosure of the Rayleigh quotient of `B` at `[ỹ_j; x̃_j]`,
   `2 Re(x̃_j*Aỹ_j)/(‖x̃_j‖² + ‖ỹ_j‖²)`, with the products in twice the working precision.
2. `E ∋ AỸ − X̃Σ̃`, `F ∋ A*X̃ − ỸΣ̃`, and `δ_j = √(‖E e_j‖² + ‖F e_j‖²)/‖ỹ_j‖`: the interval
   `max(0, σ̃_j ± δ_j)` holds a singular value (Theorem 7.1).
3. The intervals are grouped, with `kappa`, into connected components, and for a component `μ`
   of more than one index `δ_j` becomes `√(‖E(:, μ)‖² + ‖F(:, μ)‖²)/σ_min(Ỹ(:, μ))`, until the
   partition no longer changes.
4. For a cluster of one index the interval is intersected with the one of Theorem 5.1 applied to
   `B`, the gap including the distance to the eigenvalues `−σ_i` and, for `m > n`, to zero.
5. For a cluster `μ` with gap `ε` to the other intervals, the left and the right singular
   subspaces are within `τ = √(‖E(:, μ)‖² + ‖F(:, μ)‖²)/ε` of `X̃(:, μ)` and `Ỹ(:, μ)`
   (Lemma 6.1 applied to `B`; Lemma 7.2 for the cluster of the smallest singular values), and
   Lemma 3.1 gives orthonormal bases. For `m > n` the gap for the left subspace of the cluster
   of the smallest singular values also includes its distance to zero.

For `m < n` the algorithm is applied to `A*` and the two sides are exchanged.

The departures from the paper's listing are those of [`_rumplange2023_eig`](@ref): the stopping
test of step 3, the centres and the single `ε` of a cluster in step 5, and the cluster of all
singular values, for which the listing returns radius zero. The treatment of the left null
space for `m > n`, which the paper describes and leaves out of its listing, is not implemented:
`U` has `min(m, n)` columns.

# Reference

S. M. Rump and M. Lange, *Fast computation of error bounds for all eigenpairs of a Hermitian and
all singular pairs of a rectangular matrix with emphasis on eigen- and singular value clusters*,
J. Comput. Appl. Math. **434** (2023) 115332, doi 10.1016/j.cam.2023.115332, Section 7.
"""
function verifysvdall(A::BallMatrix{T}; kappa::Real = 0) where {T}
    kappa >= 0 || throw(ArgumentError("verifysvdall: kappa must be nonnegative"))
    m, n = size(A)
    if m < n
        r = verifysvdall(_ball_adjoint(A); kappa)
        return VerifySvdAllResult(r.values, r.clusters, r.V, r.U)
    end
    Am, Ar = Matrix(mid(A)), rad(A)
    S = eltype(Am)
    κ = T(kappa)
    Fs = svd(Am)
    Xs, Ys = Matrix{S}(Fs.U), Matrix{S}(Fs.V)
    absX, absY = _modulus_up(Xs), _modulus_up(Ys)
    up(f) = setrounding(f, T, RoundUp)
    failed() = VerifySvdAllResult(fill(Ball(zero(T), T(Inf)), n), [collect(1:n)],
        BallMatrix(Xs, fill(T(Inf), m, n)), BallMatrix(Ys, fill(T(Inf), n, n)))

    # ‖x̃_j‖², ‖ỹ_j‖² and the Rayleigh quotients of B
    x2m, x2r = _rumplange2023_coldots(Xs, (Xs,), zeros(T, m, n))
    y2m, y2r = _rumplange2023_coldots(Ys, (Ys,), zeros(T, n, n))
    y2lo = T[sub_down(real(y2m[j]), y2r[j]) for j in 1:n]
    all(>(0), y2lo) || return failed()
    Q1, Q2, RQ = _two_term_product_sum(((Am, Ys, one(T)),))
    RQ = up(() -> RQ .+ Ar * absY)
    qm, qr = _rumplange2023_coldots(Xs, (Q1, Q2), RQ)
    z2 = [Ball(real(x2m[j]), x2r[j]) + Ball(real(y2m[j]), y2r[j]) for j in 1:n]
    z2lo = T[sub_down(mid(z), rad(z)) for z in z2]
    rho = [Ball(2 * real(qm[j]), mul_up(T(2), qr[j])) / z2[j] for j in 1:n]
    sig = T[mid(r) for r in rho]
    rhor = T[rad(r) for r in rho]
    all(isfinite, sig) || return failed()

    Σ = Matrix{S}(Diagonal(sig))
    Eb = _accurate_product_sum(((Am, Ys, one(T)), (Xs, Σ, -one(T))))
    E = BallMatrix(mid(Eb), up(() -> rad(Eb) .+ Ar * absY))
    Fb = _accurate_product_sum(((Matrix{S}(Am'), Xs, one(T)), (Ys, Σ, -one(T))))
    F = BallMatrix(mid(Fb), up(() -> rad(Fb) .+ transpose(Ar) * absX))
    g2 = add_up.(_column_norms2_up(E), _column_norms2_up(F))
    normG = sqrt_up.(g2)
    delta = T[div_up(normG[j], sqrt_down(y2lo[j])) for j in 1:n]

    bounds() = (max.(zero(T), sub_down.(sig, delta)), add_up.(sig, delta))
    function relation(lo, hi)
        near(i, j) = sub_down(lo[i], mul_up(κ, abs(lo[i]))) <= lo[j] &&
                     add_up(hi[i], mul_up(κ, abs(hi[i]))) >= lo[j]
        return Bool[near(i, j) || near(j, i) for i in 1:n, j in 1:n]
    end
    function cluster_delta!(v)
        s = _rumplange2023_singmin(Ys[:, v])
        a, b = T(collatz_upper_bound_L2_opnorm(E[:, v])), T(collatz_upper_bound_L2_opnorm(F[:, v]))
        nG = sqrt_up(add_up(mul_up(a, a), mul_up(b, b)))
        normG[v] .= nG
        delta[v] .= s > 0 ? div_up(nG, s) : T(Inf)
    end
    clusters = [[j] for j in 1:n]
    for _ in 1:(n + 1)
        new = _rumplange2023_components(relation(bounds()...))
        new == clusters && break
        clusters = new
        for v in clusters
            length(v) > 1 && cluster_delta!(v)
        end
    end
    if _rumplange2023_components(relation(bounds()...)) != clusters
        clusters = [collect(1:n)]
        cluster_delta!(clusters[1])
    end
    lo, hi = bounds()

    # refinesvd
    for v in clusters
        length(v) == 1 || continue
        j = v[1]
        gap = minimum((_interval_gap(lo[j], hi[j], lo[i], hi[i]) for i in 1:n if i != j);
            init = T(Inf))
        gap = min(gap, m > n ? lo[j] : mul_down(T(2), lo[j]))
        (gap > 0 && isfinite(gap)) || continue
        res = add_up(sqrt_up(div_up(g2[j], z2lo[j])), rhor[j])
        rr = add_up(div_up(mul_up(res, res), gap), rhor[j])
        lo[j] = max(lo[j], sub_down(sig[j], rr))
        hi[j] = min(hi[j], add_up(sig[j], rr))
    end

    # the singular vectors
    rU, rV = Vector{T}(undef, n), Vector{T}(undef, n)
    sqrt2 = sqrt_up(T(2))
    smallest = argmin(sig)
    for v in clusters
        ε = T(Inf)
        for j in v, i in 1:n
            i in v && continue
            ε = min(ε, _point_interval_gap(sig[j], lo[i], hi[i]))
        end
        whole = length(v) == n
        ok = all(isfinite, delta[v])
        τV = !ok ? T(Inf) : whole ? zero(T) : (ε > 0 ? div_up(normG[v[1]], ε) : T(Inf))
        τU = τV
        if m > n && smallest in v
            εU = min(ε, minimum(lo[v]))
            τU = (ok && εU > 0) ? div_up(normG[v[1]], εU) : T(Inf)
        end
        rV[v] .= add_up(T(_rumplange2023_alpha(Ys[:, v])), mul_up(sqrt2, τV))
        rU[v] .= add_up(T(_rumplange2023_alpha(Xs[:, v])), mul_up(sqrt2, τU))
    end
    values = Ball{T, T}[]
    for j in 1:n
        c = (lo[j] + hi[j]) / 2
        push!(values, isfinite(c) ? Ball(c, max(sub_up(c, lo[j]), sub_up(hi[j], c))) :
                      Ball(zero(T), T(Inf)))
    end
    return VerifySvdAllResult(values, clusters, BallMatrix(Xs, repeat(reshape(rU, 1, n), m, 1)),
        BallMatrix(Ys, repeat(reshape(rV, 1, n), n, 1)))
end
