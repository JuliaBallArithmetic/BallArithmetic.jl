# A variant of Miyajima's verified block diagonalisation: HIS certification over a
# Schur-Newton candidate frame instead of his Schur-plus-Sylvester one.
#
#   @Article{Miyajima2014a,
#     author  = {Miyajima, Shinya},
#     title   = {Fast Enclosure for All Eigenvalues and Invariant Subspaces in Generalized
#                Eigenvalue Problems},
#     journal = {SIAM Journal on Matrix Analysis and Applications},
#     year    = {2014},
#     volume  = {35},
#     number  = {3},
#     pages   = {1205--1225},
#     doi     = {10.1137/140953150},
#   }
#
# WHAT IS MIYAJIMA'S. The certification is the two-residual structure of his Theorem 3.1: with Y an
# approximate inverse of the frame, R2 := I - Y B W and R1 := Y(A W - B W Lambda), the bound carries
# the Neumann slack ||R2||/(1 - ||R2||) and refuses unless ||R2||_inf < 1, which proves B, W and Y
# nonsingular at once rather than assuming it. Theorem 3.1 states it as
# rhat = |R| 1 + (||R||_inf/(1 - ||S||_inf)) |S| 1; here the same slack is charged per row to
# Gershgorin discs of the collapsed matrix, which is the refinement that lets each disc pay for its
# own row of R2 instead of the worst one.
#
# WHAT IS NOT. His section 4 builds the frame by a Schur form plus Sylvester equations and certifies
# each block by Brouwer's theorem on a Newton operator (Theorems 4.5 and 4.10, the diagonal replaced
# by its mean). Here the frame comes from a Schur form with between-cluster Newton decoupling and a
# per-block QR, and there is no fixed-point test at all: the discs are Gershgorin's, inflated by the
# slack, with the counting theorem for the multiplicities. So the eigenvalue enclosure is his and the
# route to it is not, which is why the name carries both.
#
# The candidate frame needs no verification of its own. Every claim rests on `_certify_ball`, so the
# clustering, the merge rule and the Newton steps can be changed freely without touching rigour.
#
# Replaces the O(n^4) NJD (RDEFL staircase) route with the O(n^3) guarded-`trevc` architecture of
# the reference `RigPseudospectra.jl` (`src/vbd.jl::vbd_solve` + `src/miyajima_rump.jl::_certify`):
#
#   1. Schur `B = QTQ'` (unitary frame; GenericSchur for BigFloat);
#   2. cluster the eigenvalues at a fixed separation level (distance only);
#   3. column-norm merge: a column whose decoupling transform is O(1) is merged
#      into its dominant coupling partner (the near-defective tail coalesces);
#   4. between-cluster Newton steps `Xij = -Aij/(di-dj)`, `A <- (I+X)^-1 A(I+X)`,
#      `W <- W(I+X)` (within-cluster coupling kept);
#   5. per-block QR => a block-orthonormal basis `W` (orthonormal within each
#      invariant subspace; kappa(W) governed by inter-block angles, benign).
#
using LinearAlgebra

"""
    Miyajima2014aSchurNewtonResult

Container returned by [`miyajima2014a_schurnewton`](@ref).  Field names are duck-type
compatible with [`SchurGershgorinResult`](@ref) so the same downstream consumers
(block Schur, spectral projectors) work unchanged, with three extra
certification scalars (`nrmR2`, `beta`, `kappa`).

The basis `W` is **block-orthonormal**, not globally unitary, so projector /
similarity consumers must use `inv(W)` (not `adjoint(W)`); see
[`_vbd_unitary_basis`](@ref).
"""
struct Miyajima2014aSchurNewtonResult{MT, BT, IT, RT, ET}
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
    """Whether the result is for a pencil `Ax = λBx` (the frame inverted is then `B·W`)."""
    pencil::Bool
end

# block-orthonormal, NOT globally unitary ⇒ consumers must use inv(basis).
_vbd_unitary_basis(::Miyajima2014aSchurNewtonResult) = false

# ── Phase 1: Schur + Newton block-orthonormal basis (point matrix, O(n³)) ──

"""
    _vbd_solve(Bc::Matrix{CT}; sep = -1, maxsteps = 6) -> (W, clusters)

Guarded-`trevc` solve on the complexified point matrix `Bc`: Schur frame,
fixed-separation distance clustering, column-norm merge, between-cluster
Newton refinement, per-block QR ⇒ block-orthonormal `W`.  `clusters` is a
`Vector{Vector{Int}}` of (generally non-contiguous) index sets.
"""
function _vbd_solve(Bc::Matrix{CT}; sep::Real = -1, maxsteps::Integer = 6,
        mode::Symbol = :entrywise) where {CT}
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
    # Newton transform decoupling current cross-cluster pairs: Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ), but only for
    # pairs where that division is CONTRACTING. This per-pair gate is Miyajima's (2014a, `bdg.m`:
    # `if abs(D(i,i)-D(j,j)) > tol % elimination possible`) — a pair whose off-diagonal is not
    # small against its gap is left alone rather than divided by that gap, which is precisely the
    # division that makes the sweep diverge on a strongly non-normal band. A skipped pair is NOT
    # merged; it simply stays coupled, its residual lands in the off-block part, and the
    # disc-overlap arbiter downstream re-merges it if the enclosures really do overlap.
    θcontract = Tr(9) / 10
    function newtonX()
        cof = Int[rt(i) for i in 1:n]
        X = zeros(CT, n, n)
        @inbounds for j in 1:n, i in 1:n

            if cof[i] != cof[j]
                g = abs(d[i] - d[j])
                abs(A[i, j]) < θcontract * g && (X[i, j] = -A[i, j] / (d[i] - d[j]))
            end
        end
        return X
    end
    # off-block mass — the quantity the refinement exists to drive down (it feeds β_Λ).
    function offmass()
        cof = Int[rt(i) for i in 1:n]
        s = zero(Tr)
        @inbounds for j in 1:n, i in 1:n

            cof[i] != cof[j] && (s += abs2(A[i, j]))
        end
        return sqrt(s)
    end
    # one between-cluster Newton sweep (apply): Xᵢⱼ = −Aᵢⱼ/(dᵢ−dⱼ), A ← (I+X)⁻¹A(I+X).
    function newtonstep!()
        X = newtonX()
        iszero(X) && return false
        IpX = I + X
        A = IpX \ (A * IpX)
        W = W * IpX
        d = diag(A)
        return true
    end
    # 3. Newton refinement of the between-cluster structure, MONOTONE. A sweep is kept only if it
    # actually decreases the off-block mass; otherwise it is reverted and we stop. Refinement must
    # never inflate the enclosure, and without this guard it did: on Grcar (n=100) the certified
    # radii grew 5.17 → 8.32 → 9.04 as maxsteps went 0 → 6 → 12, because the unguarded sweeps ran
    # while every index was still a singleton and diverged before any clustering had happened.
    #
    # NOTE (why there is no coupling-based merge here any more). The clustering is decided in step 2
    # by DIAGONAL DISTANCE alone, which is exactly Rump's rule (2022, `verifyeigall` step 2:
    # `dist = mig(d-d.') <= 1e-14*normA; conncomp(graph(dist))`), and Miyajima's (2014a Alg. 3:
    # `|λ̃ᵢ-λ̃ⱼ| <= tol`). Neither author ever forms a block from the size of a decoupling transform:
    # Rump does not solve a Sylvester equation at all, and inside a block he simply bypasses the
    # division. The previous code merged column j whenever ‖X[:,j]‖₂ ≥ 1, which is not in either
    # paper — it appears to come from reading Rump's Remark 2.3 ("the columns µᵢ in Z, where
    # |µᵢ| > 1 for a cluster") as a column NORM when |µᵢ| is the CARDINALITY of the index set.
    # Because that test aggregates Σᵢ|Aᵢⱼ/(dᵢ−dⱼ)| over a whole column, on any strongly non-normal
    # band (O(1) Schur off-diagonals over O(1) gaps) it exceeded 1 for every column and collapsed
    # ALL n eigenvalues into one block, even where they are plainly separable. The rigorous arbiter
    # is, as in Miyajima Alg. 1, the overlap of the certified discs, applied downstream.
    if mode === :entrywise
        for _ in 1:maxsteps
            A0 = copy(A)
            W0 = copy(W)
            d0 = copy(d)
            m0 = offmass()
            newtonstep!() || break
            if !(offmass() < m0)
                A = A0
                W = W0
                d = d0
                break
            end
        end
    end
    groups = Dict{Int, Vector{Int}}()
    for i in 1:n
        push!(get!(groups, rt(i), Int[]), i)
    end
    clusters = collect(values(groups))

    # 4b. EXACT block-Sylvester elimination (Miyajima 2014a `bdg.m`). Where the
    # entrywise sweep above is a first-order step that needs several iterations
    # and can stall, this decouples each block from everything after it in ONE
    # pass and exactly: with A = [A₁₁ A₁₂; 0 A₂₂] block upper triangular and
    # S = [I V; 0 I], the similarity S⁻¹AS sends A₁₂ ↦ A₁₁V − VA₂₂ + A₁₂, so
    # solving the Sylvester equation A₁₁V − VA₂₂ = −A₁₂ annihilates the strip.
    # The basis follows as W ← WS, i.e. W₂ ← W₁V + W₂ — exactly bdg.m's
    # `X(:,idx+ctr+1:end) = X(:,idx:idx+ctr)*V + X(:,idx+ctr+1:end)`.
    # `bdg.m` gates this on |dᵢ−dⱼ| > tol, which is precisely "different
    # cluster", so the block boundaries below already encode it.
    if mode === :block
        order = vcat(sort(clusters; by = first)...)
        A = A[order, order]
        W = W[:, order]
        sizes = [length(c) for c in sort(clusters; by = first)]
        # The transform is W <- W[I V; 0 I], whose conditioning is
        # kappa = ((||V||_2 + sqrt(||V||_2^2+4))/2)^2 ~ ||V||_2^2, and whose rounding enters the
        # certification slack beta_Lambda at about u*||V||_2*||A||. Applying an unbounded V
        # therefore buys a block-diagonal candidate at the price of a basis the Neumann test
        # ||R2|| < 1 then rejects, losing the candidate entirely: on a defective matrix this
        # built kappa(W) = 1.8e15 and failed with ||R2||_inf = 1.46. Two near-coincident
        # clusters are better merged than separated, so a V above the threshold merges the
        # current block with the next and solves again. The norm is the spectral one: ||V||_F
        # overestimates it by up to sqrt(min(size(V)...)), which merges earlier than needed.
        xmax = sqrt(one(real(CT)) / eps(real(CT)))
        k = 1
        pos = 1
        while k <= length(sizes)
            lo, hi = pos, pos + sizes[k] - 1
            if hi < n
                # `solve_sylvester_oracle` and not `LinearAlgebra.sylvester`: the latter has no
                # method for BigFloat and recursed until StackOverflowError, which the bare
                # `catch` here took for an ill-posed pair, so that for BigFloat input no block was
                # ever decoupled. The oracle solves in Float64 for other types, which is enough
                # for a candidate.
                V = try
                    solve_sylvester_oracle(A[lo:hi, lo:hi], A[lo:hi, (hi + 1):n],
                        A[(hi + 1):n, (hi + 1):n])
                catch err
                    err isa Union{LAPACKException, SingularException} || rethrow()
                    nothing            # ill-posed pair: merge below, or leave the strip coupled
                end
                if V === nothing || !all(isfinite, V) || opnorm(V) > xmax
                    if k < length(sizes)
                        sizes[k] += sizes[k + 1]
                        deleteat!(sizes, k + 1)
                        continue       # same position, one bigger block, solve again
                    end
                    break              # nothing left to merge: leave the strip coupled
                end
                W[:, (hi + 1):n] = W[:, lo:hi] * V + W[:, (hi + 1):n]
                A[lo:hi, (hi + 1):n] .= zero(CT)
            end
            pos = hi + 1
            k += 1
        end
        # relabel the clusters to the new contiguous positions
        clusters = Vector{Int}[]
        pos = 1
        for s_k in sizes
            push!(clusters, collect(pos:(pos + s_k - 1)))
            pos += s_k
        end
    end

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
        B::Union{Nothing, BallMatrix} = nothing,
        kappa_mode::Symbol = :cheap) where {T}
    WB = BallMatrix(W)

    # Frame that Y approximately inverts. For the pencil `Ax = λBx` the change of
    # variable `x = Wy` gives `(YAW)y = λ(YBW)y`, so with `R₂ = YBW − I` the
    # collapsed matrix is again `M = (I+R₂)⁻¹Ã`, `Ã = YAW` — the same structure as
    # the standard problem, which is why every step below is unchanged.
    FB = B === nothing ? WB : B * WB
    Y = inv(mid(FB))                                # Y is "free": any float inverse
    YB = BallMatrix(Y)

    R2 = YB * FB - I                                # single residual (Y is "free")
    transformed = YB * A * WB                       # rigorous enclosure of Ã = Y·A·W

    # per-row certification slack βᵢ (charges the Y≠W⁻¹ error to each disc using the
    # actual residual rows, not the global ‖R₂‖_∞‖Ã‖_∞); throws if ‖R₂‖_∞ ≥ 1.
    beta_rows, nrmR2 = _vbd_beta_rows(R2, transformed, T)

    # β-inflated discs: reuse the proven Gershgorin loop, then add βᵢ to each radius.
    base = _vbd_gershgorin_intervals(transformed; hermitian = false)
    discs = _inflate_intervals(base, beta_rows)
    beta = maximum(beta_rows)                        # scalar summary for the record

    # κ₂ of the frame Y inverts: κ₂(W) for the standard problem, κ₂(B·W) for the
    # pencil. Reported only; never used in the enclosure.
    kappa = _vbd_kappa(FB, YB, R2, kappa_mode, T)
    return transformed, discs, nrmR2, beta, kappa, FB
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
    miyajima2014a_schurnewton(A::BallMatrix; sep = -1, maxsteps = 6, kappa_mode = :cheap)

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
function miyajima2014a_schurnewton(A::BallMatrix{T, NT}; sep::Real = -1,
        maxsteps::Integer = 6, kappa_mode::Symbol = :cheap,
        refine::Symbol = :auto) where {T, NT}
    return _miyajima2014a_schurnewton(A, nothing; sep, maxsteps, kappa_mode, refine)
end

"""
    miyajima2014a_schurnewton(A::BallMatrix, B::BallMatrix; sep = -1, maxsteps = 6,
                     kappa_mode = :cheap)

Verified block diagonalisation of the **pencil** `Ax = λBx`, by the same Schur +
Newton route as the one-argument method.

The change of variable `x = Wy` turns the pencil into `(YAW)y = λ(YBW)y`, so
with an approximate inverse `Y ≈ (BW)⁻¹` and the residual `R₂ = YBW − I` the
collapsed matrix is again `M = (I+R₂)⁻¹Ã` with `Ã = YAW`. That is *structurally
identical* to the standard problem, so the per-row certification slack, the
β-inflated Gershgorin discs, the overlap re-clustering and the block enclosure
carry over unchanged — the only substitution is `W ⟶ B·W` wherever the frame is
paired with `Y` or with the candidate `Λ` (`R₁ = Y(AW − BWΛ)`).

`B` is **not** assumed Hermitian or positive definite; its nonsingularity is not
assumed either but *proved*, by `‖R₂‖_∞ < 1` (which certifies `B`, `W` and `Y`
nonsingular at once). This follows Miyajima (2014), *Fast enclosure for all
eigenvalues and invariant subspaces in generalized eigenvalue problems*,
SIAM J. Matrix Anal. Appl. 35(3), 1205–1225, whose block-diagonalisation
algorithm (§4) is obtained here through the Schur–Newton frame rather than the
Kronecker fixed point of that paper.

The candidate basis is produced in floating point from `mid(B) \\ mid(A)`; being
a candidate it needs no verification of its own, all correctness resting on the
final certification.
"""
function miyajima2014a_schurnewton(A::BallMatrix{T, NT}, B::BallMatrix; sep::Real = -1,
        maxsteps::Integer = 6, kappa_mode::Symbol = :cheap,
        refine::Symbol = :auto) where {T, NT}
    size(A) == size(B) ||
        throw(DimensionMismatch("A and B must have the same size"))
    return _miyajima2014a_schurnewton(A, B; sep, maxsteps, kappa_mode, refine)
end

function _miyajima2014a_schurnewton(A::BallMatrix{T, NT}, B::Union{Nothing, BallMatrix};
        sep::Real = -1, maxsteps::Integer = 6, kappa_mode::Symbol = :cheap,
        refine::Symbol = :auto) where {T, NT}
    refine in (:auto, :none, :entrywise, :block) ||
        throw(ArgumentError("refine must be :auto, :none, :entrywise or :block"))
    m, n = size(A)
    m == n || throw(ArgumentError("miyajima2014a_schurnewton expects a square matrix"))

    CT = Complex{T}
    # complexify the input ball once so the certification products are unambiguously
    # complex (mid(A) may be real); radii are preserved.
    Acx = BallMatrix(CT.(mid(A)), rad(A))
    Bcx = B === nothing ? nothing : BallMatrix(CT.(mid(B)), rad(B))

    # Candidate frame: the collapsed midpoint matrix. Float only — a candidate
    # needs no verification, correctness rests on the certification below.
    Bc = if B === nothing
        CT.(mid(A))
    else
        try
            CT.(mid(B)) \ CT.(mid(A))
        catch
            throw(ArgumentError("mid(B) is numerically singular: no candidate frame"))
        end
    end
    modes = refine === :auto ? (:none, :entrywise, :block) : (refine,)
    best = nothing
    best_score = nothing
    failures = String[]

    for m in modes
        cand = try
            W, cl = _vbd_solve(Bc; sep, maxsteps, mode = m)
            _vbd_finish(Acx, Bcx, W, cl, n, kappa_mode, T)
        catch e
            push!(failures, "$m: " * first(sprint(showerror, e), 80))
            nothing
        end
        cand === nothing && continue
        # Score on CERTIFIED quantities only: the worst disc radius (the primary
        # product), then the off-block remainder (which drives β_Λ and the
        # deflating-subspace residual). Every candidate has already passed the
        # same certification, so picking between them cannot affect rigour — the
        # basis is a candidate and correctness rests on `_certify_ball`.
        score = (maximum(rad, cand.cluster_intervals), cand.remainder_norm)
        if best_score === nothing || score < best_score
            best, best_score = cand, score
        end
    end

    best === nothing &&
        throw(ArgumentError("no VBD candidate could be certified (" *
                            join(failures, "; ") * ")"))
    return best
end

# Certify a candidate basis and assemble the result: disc-overlap re-clustering,
# per-cluster reorthogonalisation, block data.
function _vbd_finish(Acx::BallMatrix{T}, Bcx, W, cl, n::Integer,
        kappa_mode::Symbol, ::Type{T}) where {T}
    # permute columns so the blocks are contiguous
    order = isempty(cl) ? collect(1:n) : vcat(cl...)
    W = W[:, order]

    identity_order = collect(1:n)
    local transformed, discs, nrmR2, beta, kappa, clusters, FB
    # The returned `clusters` must be the connected components of the returned `discs`: that is
    # what Gershgorin's counting theorem needs before a cluster can be said to hold exactly its
    # own number of eigenvalues, and it is what `block_enclosure` reports as `mult`.
    #
    # Reorthogonalising a merged block changes the frame, so it changes the discs, so it can change
    # the components. Doing it after the loop and certifying once more left `clusters` describing
    # the PREVIOUS discs: measured over 14 matrices, 6 came back inconsistent, a defective
    # triangular at n = 12 returning clusters [6,1,1,1,1,1,1] whose own discs merge into
    # [9,1,1,1], so three claimed singletons each asserted one eigenvalue where the counting
    # theorem licenses only nine for the component. The reorthogonalisation therefore happens
    # INSIDE the loop, and the loop exits only when the clustering is contiguous, consistent with
    # the discs in hand, and already orthonormalised.
    orthonormalised_for = Vector{Int}[]
    attempts = 0
    while true
        transformed, discs, nrmR2, beta, kappa, FB = _certify_ball(Acx, W; B = Bcx,
            kappa_mode)
        clusters, ord = _interval_clusters(discs)
        if ord != identity_order
            W = W[:, ord]
            empty!(orthonormalised_for)          # a permuted frame invalidates the QRs
        elseif clusters != orthonormalised_for && any(cl -> length(cl) > 1, clusters)
            # a merged block is the concatenation of separately orthonormalised sub-blocks, so it
            # is not block-orthonormal as a unit; re-QR preserves each invariant subspace
            for cl in clusters
                length(cl) > 1 && (W[:, cl] = Matrix(qr(W[:, cl]).Q))
            end
            orthonormalised_for = copy(clusters)
        else
            break
        end
        attempts += 1
        attempts > 2 * n &&
            throw(ArgumentError("failed to reach a β-Gershgorin clustering that is contiguous, " *
                                "consistent with its own discs and block-orthonormalised"))
    end

    block = _block_diagonal_part(transformed, clusters)
    remainder = transformed - block
    remainder_norm = upper_bound_L2_opnorm(remainder)
    midT = mid(transformed)
    eigenvalues = [midT[i, i] for i in 1:n]

    block_coupling, block_centers, block_nonnormality,
    block_residual_norm = _vbd_block_data(
        Acx, BallMatrix(W), BallMatrix(inv(mid(FB))), transformed, block, remainder,
        clusters, T; FB = FB)

    return Miyajima2014aSchurNewtonResult(W, transformed, block, remainder, clusters,
        discs, remainder_norm, eigenvalues, nrmR2, beta, kappa, block_coupling,
        block_centers, block_nonnormality, block_residual_norm, Bcx !== nothing)
end
