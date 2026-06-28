# Schur + Newton verified block diagonalisation (VBD)
#
# Replaces the O(n⁴) NJD (RDEFL staircase) route with the O(n³) guarded-`trevc`
# architecture of the reference `RigPseudospectra.jl` (`src/vbd.jl::vbd_solve` +
# `src/miyajima_rump.jl::_certify`):
#
#   1. Schur `B = QTQ'` (unitary frame; GenericSchur for BigFloat);
#   2. cluster the eigenvalues at a fixed separation level (distance only);
#   3. column-norm merge: a column whose decoupling transform is O(1) is merged
#      into its dominant coupling partner (the near-defective tail coalesces);
#   4. between-cluster Newton steps `Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ)`, `A ← (I+X)⁻¹A(I+X)`,
#      `W ← W(I+X)` (within-cluster coupling kept);
#   5. per-block QR ⇒ a block-orthonormal basis `W` (orthonormal within each
#      invariant subspace; κ(W) governed by inter-block angles, benign).
#
# CERTIFICATION (the rigorous part — Miyajima two-residual, in ball arithmetic):
# with `Y = inv(W)` an APPROXIMATE inverse, form the residual `R₂ = YW − I` and
# require `‖R₂‖_∞ < 1` (else REFUSE — `W` not certifiably nonsingular). The
# collapsed `Ã = Y·A·W` (a ball product against the INPUT ball `A`, so input
# radii propagate) is enclosed rigorously; the systematic `Y ≠ W⁻¹` error is
# charged through the slack `β = ‖R₂‖_∞‖Ã‖_∞/(1−‖R₂‖_∞)` added to every
# Gershgorin disc radius. The β-inflated discs of `Ã` then rigorously enclose
# `σ(M)` for every `M ∈ A`. This β is exactly what the old NJD path omitted.

using LinearAlgebra

"""
    SchurNewtonVBDResult

Container returned by [`schur_newton_vbd`](@ref).  Field names are duck-type
compatible with [`MiyajimaVBDResult`](@ref) so the same downstream consumers
(block Schur, spectral projectors) work unchanged, with three extra
certification scalars (`nrmR2`, `beta`, `kappa`).

The basis `W` is **block-orthonormal**, not globally unitary, so projector /
similarity consumers must use `inv(W)` (not `adjoint(W)`); see
[`_vbd_unitary_basis`](@ref).
"""
struct SchurNewtonVBDResult{MT, BT, IT, RT, ET}
    """Block-orthonormal basis `W` that block-diagonalises `mid(A)` (NOT unitary)."""
    basis::MT
    """Collapsed enclosure `Ã = inv(W)·A·W` (ball matrix)."""
    transformed::BT
    """Block-diagonal truncation preserving the verified clusters."""
    block_diagonal::BT
    """Rigorous remainder satisfying `transformed = block_diagonal + remainder`."""
    remainder::BT
    """Index ranges identifying each spectral cluster (contiguous after permutation)."""
    clusters::Vector{UnitRange{Int}}
    """β-inflated Gershgorin discs of `Ã` enclosing `σ(A)` (one per diagonal entry)."""
    cluster_intervals::Vector{IT}
    """Rigorous upper bound on `‖remainder‖₂`."""
    remainder_norm::RT
    """Eigenvalue estimates `diag(mid(transformed))`."""
    eigenvalues::Vector{ET}
    """Certified `‖R₂‖_∞` with `R₂ = inv(W)·W − I` (proved `< 1`)."""
    nrmR2::RT
    """Certification slack `β = ‖R₂‖_∞‖Ã‖_∞/(1−‖R₂‖_∞)` (max over rows of the per-row βᵢ)."""
    beta::RT
    """Condition diagnostic `κ₂ = ‖W‖₂‖W⁻¹‖₂` (`:cheap` Collatz or `:svdbox`)."""
    kappa::RT
    """Per-block coupling radii `rᵢ = ‖Ñ[Cᵢ,:]‖₂ + β₂` (block-Gershgorin, one per cluster)."""
    block_coupling::Vector{RT}
    """Per-block barycentres `cᵢ = mean(diag Pᵢ)` (the within-block recentring)."""
    block_centers::Vector{ET}
    """Per-block within-block non-normality `nᵢ = ‖Pᵢ − cᵢI‖₂`."""
    block_nonnormality::Vector{RT}
    """Block-residual slack `β_Λ = ‖R₁‖₂/(1−‖R₂‖₂)`, `R₁ = Y(AW − WΛ)` (certifies `Λ`)."""
    block_residual_norm::RT
end

# block-orthonormal, NOT globally unitary ⇒ consumers must use inv(basis).
_vbd_unitary_basis(::SchurNewtonVBDResult) = false

# ── Phase 1: Schur + Newton block-orthonormal basis (point matrix, O(n³)) ──

"""
    _vbd_solve(Bc::Matrix{CT}; sep = -1, maxsteps = 6) -> (W, clusters)

Guarded-`trevc` solve on the complexified point matrix `Bc`: Schur frame,
fixed-separation distance clustering, column-norm merge, between-cluster
Newton refinement, per-block QR ⇒ block-orthonormal `W`.  `clusters` is a
`Vector{Vector{Int}}` of (generally non-contiguous) index sets.
"""
function _vbd_solve(Bc::Matrix{CT}; sep::Real = -1, maxsteps::Integer = 6) where {CT}
    n = size(Bc, 1)
    Tr = real(CT)
    # 1. Schur (orthonormal frame); BigFloat via GenericSchur, Float64 fallback.
    Q, A = try
        F = schur(Bc)
        Matrix(F.vectors), Matrix(F.Schur)
    catch
        F = schur(ComplexF64.(Bc))
        CT.(Matrix(F.vectors)), CT.(Matrix(F.Schur))
    end
    d = diag(A)
    # 2. initial distance clustering — |dᵢ−dⱼ| < τ, connected components (union–find).
    τ = sep > 0 ? Tr(sep) : sqrt(eps(Tr)) * opnorm(Bc, Inf)
    parent = collect(1:n)
    rt(x) = parent[x] == x ? x : (parent[x] = rt(parent[x]))
    uni(a, b) = (ra = rt(a); rb = rt(b); ra == rb ? false : (parent[ra] = rb; true))
    @inbounds for j in 1:n, i in 1:(j - 1)
        abs(d[i] - d[j]) < τ && uni(i, j)
    end
    W = copy(Q)
    # Newton transform decoupling current cross-cluster pairs: Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ).
    function newtonX()
        cof = Int[rt(i) for i in 1:n]
        X = zeros(CT, n, n)
        @inbounds for j in 1:n, i in 1:n
            cof[i] != cof[j] && (X[i, j] = -A[i, j] / (d[i] - d[j]))
        end
        return X
    end
    # 3. column-norm merge: a column whose decoupling transform has ‖X[:,j]‖₂ ≥ 1
    # "fails" (the rigorous fixed point would not contract) ⇒ merge it into its
    # dominant coupling partner and re-cluster.  Merge-only: clusters grow to a
    # fixpoint, never split.
    Xcap = one(Tr)
    for _ in 1:n
        X = newtonX()
        fail = findall(j -> norm(view(X, :, j)) > Xcap, 1:n)
        isempty(fail) && break
        merged = false
        for j in fail
            uni(argmax(abs.(view(X, :, j))), j) && (merged = true)
        end
        merged || break
    end
    # 4. Newton refinement of the (now separable) between-cluster structure.
    for _ in 1:maxsteps
        X = newtonX()
        iszero(X) && break
        IpX = I + X
        A = IpX \ (A * IpX)
        W = W * IpX
        d = diag(A)
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, rt(i), Int[]), i)
    end
    clusters = collect(values(groups))
    # 5. orthogonalize each block ⇒ block-orthonormal W.
    for c in clusters
        W[:, c] = Matrix(qr(W[:, c]).Q)
    end
    return W, clusters
end

# ── Phase 2: Miyajima two-residual certification against the INPUT ball ──

"""
    _certify_ball(A::BallMatrix, W; kappa_mode = :cheap)
        -> (transformed, discs, nrmR2, beta, kappa)

Certify the candidate basis `W` of the ball matrix `A`.  Returns the collapsed
ball enclosure `Ã = inv(W)·A·W`, the β-inflated Gershgorin discs, and the
certification record.  Throws if the Neumann condition `‖R₂‖_∞ < 1` fails.
"""
function _certify_ball(A::BallMatrix{T}, W::AbstractMatrix;
        kappa_mode::Symbol = :cheap) where {T}
    Wc = W
    Y = inv(Wc)
    WB = BallMatrix(Wc)
    YB = BallMatrix(Y)

    R2 = YB * WB - I                                # single residual (Y is "free")
    transformed = YB * A * WB                       # rigorous enclosure of inv(W)·A·W

    # per-row certification slack βᵢ (charges the Y≠W⁻¹ error to each disc using the
    # actual residual rows, not the global ‖R₂‖_∞‖Ã‖_∞); throws if ‖R₂‖_∞ ≥ 1.
    beta_rows, nrmR2 = _vbd_beta_rows(R2, transformed, T)

    # β-inflated discs: reuse the proven Gershgorin loop, then add βᵢ to each radius.
    base = _vbd_gershgorin_intervals(transformed; hermitian = false)
    discs = _inflate_intervals(base, beta_rows)
    beta = maximum(beta_rows)                        # scalar summary for the record

    kappa = _vbd_kappa(WB, YB, R2, kappa_mode, T)
    return transformed, discs, nrmR2, beta, kappa
end

# κ₂ = ‖W‖₂‖W⁻¹‖₂ — a reported diagnostic, NOT used in the eigenvalue enclosure.
# `:cheap` (default) keeps the whole pipeline O(n³ matmul); `:svdbox` does one
# verified SVD of W for a tight κ₂.
function _vbd_kappa(WB, YB, R2, mode::Symbol, ::Type{T}) where {T}
    if mode === :cheap
        r2₂ = upper_bound_L2_opnorm(R2)
        return r2₂ < 1 ?
               upper_bound_L2_opnorm(WB) * upper_bound_L2_opnorm(YB) / (1 - r2₂) :
               T(Inf)
    else
        sv = svdbox(WB)
        σmax = maximum(mid(s) + rad(s) for s in sv)
        σmin = minimum(mid(s) - rad(s) for s in sv)
        invW = if σmin > 0
            inv(σmin)
        else
            r2₂ = upper_bound_L2_opnorm(R2)
            r2₂ < 1 ? svd_bound_L2_opnorm(YB) / (1 - r2₂) : T(Inf)
        end
        return σmax * invW
    end
end

# ── Driver ──

"""
    schur_newton_vbd(A::BallMatrix; sep = -1, maxsteps = 6, kappa_mode = :cheap)

Verified block diagonalisation of the square ball matrix `A` via the O(n³)
Schur + Newton route, with a rigorous Miyajima two-residual certification.

The midpoint is reduced to a block-orthonormal basis `W` (Schur frame, distance
clustering, between-cluster Newton decoupling, per-block QR).  The enclosure is
transported to that basis and the β-inflated Gershgorin discs of `Ã = inv(W)·A·W`
rigorously enclose `σ(M)` for every `M ∈ A`.  The discs are re-clustered by
overlap (the rigorous arbiter — an over-optimistic Newton split whose discs
still overlap is re-merged) so that each cluster is a contiguous range holding
exactly its eigenvalue count.

Unlike the removed NJD route this is rigorous (it charges the `Y ≠ W⁻¹` error
through `β`) and **refuses** (throws) when the basis fails the Neumann condition
`‖R₂‖_∞ < 1`, i.e. when `mid(A)` is numerically defective at the working
precision.  `kappa_mode` selects the `κ₂` diagnostic: `:cheap` (Collatz, keeps
O(n³)) or `:svdbox` (one verified SVD of `W`).
"""
function schur_newton_vbd(A::BallMatrix{T, NT}; sep::Real = -1,
        maxsteps::Integer = 6, kappa_mode::Symbol = :cheap) where {T, NT}
    m, n = size(A)
    m == n || throw(ArgumentError("schur_newton_vbd expects a square matrix"))

    CT = Complex{T}
    Bc = CT.(mid(A))
    # complexify the input ball once so the certification products are unambiguously
    # complex (mid(A) may be real); radii are preserved.
    Acx = BallMatrix(Bc, rad(A))
    W, cl = _vbd_solve(Bc; sep, maxsteps)

    # permute columns so the Newton blocks are contiguous
    order = isempty(cl) ? collect(1:n) : vcat(cl...)
    W = W[:, order]

    identity_order = collect(1:n)
    local transformed, discs, nrmR2, beta, kappa, clusters
    attempts = 0
    while true
        transformed, discs, nrmR2, beta, kappa = _certify_ball(Acx, W; kappa_mode)
        clusters, ord = _interval_clusters(discs)
        ord == identity_order && break
        W = W[:, ord]
        attempts += 1
        attempts > n &&
            throw(ArgumentError("failed to permute β-Gershgorin clusters into contiguous blocks"))
    end

    # Reorthogonalize each FINAL cluster.  `_vbd_solve` QR-orthogonalizes per Newton
    # cluster, but the disc-overlap re-clustering above can MERGE Newton blocks; a merged
    # block is the concatenation of separately-orthonormalized sub-blocks, so it is not
    # block-orthonormal as a unit.  Re-QR'ing each final cluster restores that (it preserves
    # the cluster's invariant subspace, so the clustering is unchanged); re-certify against
    # the cleaned basis.  Rigor never depended on this — it tightens κ₂ and the block σ_min.
    if any(cl -> length(cl) > 1, clusters)
        for cl in clusters
            length(cl) > 1 && (W[:, cl] = Matrix(qr(W[:, cl]).Q))
        end
        transformed, discs, nrmR2, beta, kappa = _certify_ball(Acx, W; kappa_mode)
    end

    block = _block_diagonal_part(transformed, clusters)
    remainder = transformed - block
    remainder_norm = upper_bound_L2_opnorm(remainder)
    midT = mid(transformed)
    eigenvalues = [midT[i, i] for i in 1:n]

    block_coupling, block_centers, block_nonnormality, block_residual_norm = _vbd_block_data(
        Acx, BallMatrix(W), BallMatrix(inv(W)), transformed, block, remainder, clusters, T)

    return SchurNewtonVBDResult(W, transformed, block, remainder, clusters,
        discs, remainder_norm, eigenvalues, nrmR2, beta, kappa, block_coupling,
        block_centers, block_nonnormality, block_residual_norm)
end
