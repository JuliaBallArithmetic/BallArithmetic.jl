"""
    SchurGershgorinResult

Container returned by [`schur_gershgorin_enclosure`](@ref). The fields hold the
unitary basis (Schur, or eigenvectors when the input is Hermitian), the enclosure
of `A` transported to it, the block-diagonal part of that enclosure, the rigorous
remainder, and the clusters read off the inflated Gershgorin discs.

The basis is **not** a block-diagonalising one: it is the unitary basis of the
midpoint matrix, permuted so the clusters are contiguous, so `remainder` is the
off-block part of `Z*AZ` and is not small. Field names are duck-type compatible
with [`Miyajima2014aSchurNewtonResult`](@ref), whose basis does decouple the blocks, so the
same downstream consumers accept either.
"""
struct SchurGershgorinResult{MT, BT, IT, RT, ET}
    """Unitary basis of `mid(A)`, column-permuted so the clusters are contiguous."""
    basis::MT
    """Interval matrix expressed in the chosen basis."""
    transformed::BT
    """Block-diagonal truncation preserving the verified clusters."""
    block_diagonal::BT
    """Rigorous remainder satisfying `transformed = block_diagonal + remainder`."""
    remainder::BT
    """Index ranges identifying each spectral cluster."""
    clusters::Vector{UnitRange{Int}}
    """Gershgorin-type discs enclosing each diagonal entry."""
    cluster_intervals::Vector{IT}
    """Rigorous upper bound on `‖remainder‖₂` (best of the Collatz and ‖·‖₁‖·‖∞ bounds)."""
    remainder_norm::RT
    """Recentred (Rayleigh-quotient) eigenvalue estimates `diag(mid(transformed))`."""
    eigenvalues::Vector{ET}
    """Per-block coupling radii `rᵢ = ‖Ñ[Cᵢ,:]‖₂ + β₂` (block-Gershgorin, one per cluster)."""
    block_coupling::Vector{RT}
    """Per-block barycentres `cᵢ = mean(diag Pᵢ)` (the within-block recentring)."""
    block_centers::Vector{ET}
    """Per-block within-block non-normality `nᵢ = ‖Pᵢ − cᵢI‖₂`."""
    block_nonnormality::Vector{RT}
    """Block-residual slack `β_Λ = ‖R₁‖₂/(1−‖R₂‖₂)`, `R₁ = Y(AW − WΛ)` (certifies `Λ`)."""
    block_residual_norm::RT
end

"""
    _vbd_unitary_basis(result) -> Bool

Trait distinguishing VBD results whose `basis` is genuinely unitary (so that
`adjoint(basis)` is its inverse) from those whose basis is only block-orthonormal
(so that consumers must use `inv(basis)`).  `SchurGershgorinResult` (NSD / Schur /
Hermitian eigenvectors) is unitary; `Miyajima2014aSchurNewtonResult` is not.
"""
_vbd_unitary_basis(::SchurGershgorinResult) = true

# PER-ROW certification slack charging the approximate-inverse error `Y ≈ W⁻¹` to each
# Gershgorin disc.  With `Ã = Y·A·W` (a rigorous ball enclosure) and `R₂ = Y·W − I`, the
# true collapsed matrix is `M₀ = W⁻¹AW = (I+R₂)⁻¹Ã`, so disc `i` (centre `Ãᵢᵢ`, radius
# `Σ_{j≠i}|Ãᵢⱼ|`) must be widened by `‖rowᵢ(M₀ − Ã)‖₁`.  Since `M₀ − Ã = −K·Ã` with
# `K = R₂(I+R₂)⁻¹`:
#     ‖rowᵢ(M₀−Ã)‖₁ = ‖rowᵢ(K)·Ã‖₁ ≤ ‖rowᵢ(R₂)‖₁ · ‖Ã‖_∞ / (1−‖R₂‖_∞)  =:  βᵢ.
# The global operator norm `‖R₂‖_∞` (= max row sum) is replaced by the INDIVIDUAL row
# sum `‖rowᵢ(R₂)‖₁`, so each disc is as tight as its own row of the residual — `βᵢ ≤` the
# uniform bound `‖R₂‖_∞‖Ã‖_∞/(1−‖R₂‖_∞)` for every `i`, with no extra matrix product.
# Refuses (throws) when `‖R₂‖_∞ ≥ 1`.  Returns `(β::Vector, ‖R₂‖_∞)`.  For a genuinely
# unitary frame (`Y = Z'`, Z Hermitian-eigvecs / Schur vectors) the `βᵢ` are tiny but
# nonzero, and must still be accounted for.
function _vbd_beta_rows(R2::BallMatrix, transformed::BallMatrix, ::Type{T}) where {T}
    n = size(R2, 1)
    nrmR2 = upper_bound_L_inf_opnorm(R2)
    nrmR2 < 1 ||
        error("VBD: Neumann condition ‖R₂‖_∞ = $nrmR2 ≥ 1; basis not certifiably " *
              "nonsingular at this precision")
    normT = upper_bound_L_inf_opnorm(transformed)     # ‖Ã‖_∞
    denom = setrounding(T, RoundDown) do
        one(T) - nrmR2
    end
    g = setrounding(T, RoundUp) do
        normT / denom                                 # ‖Ã‖_∞/(1−‖R₂‖_∞)
    end
    absR2 = upper_abs(R2)
    beta = Vector{T}(undef, n)
    for i in 1:n
        beta[i] = setrounding(T, RoundUp) do
            rowR2 = zero(T)
            for j in 1:n
                rowR2 += absR2[i, j]                   # ‖rowᵢ(R₂)‖₁
            end
            rowR2 * g
        end
    end
    return beta, nrmR2
end

# add the per-row certification slack `β[i]` to each Gershgorin disc radius (RoundUp)
function _inflate_intervals(intervals::AbstractVector{Ball{T, CT}},
        beta::AbstractVector{T}) where {T, CT}
    return [Ball(mid(intervals[i]), setrounding(T, RoundUp) do
                rad(intervals[i]) + beta[i]
            end) for i in eachindex(intervals)]
end

# PER-BLOCK enclosure data (draft Thm. enclosure + Prop. blockgersg, with within-block
# barycentre recentring).  The block-diagonal candidate `Λ = blockdiag(Pᵢ)` may be formed
# freely (it is a candidate); rigor comes from the two residuals computed here:
#   * `R₂ = YW − I`  (basis residual, Neumann `‖R₂‖₂ < 1`);
#   * `R₁ = Y(AW − WΛ)`  (BLOCK residual) ⇒ `M − Λ = (I+R₂)⁻¹R₁`, so the SINGLE slack
#       `β_Λ = ‖R₁‖₂/(1−‖R₂‖₂)` bounds the full deviation `‖M − Λ‖₂` (inter-block leakage
#       included — no separate coupling term); `spec(A) ⊆ ⋃ᵢ{z: σ_min(Pᵢ − zI) ≤ β_Λ}`.
# For each block `Cᵢ` (`Pᵢ = M̃[Cᵢ,Cᵢ]`) we also record:
#   * barycentre  `cᵢ = mean(diag Pᵢ)`  and within-block non-normality `nᵢ = ‖Pᵢ − cᵢI‖₂`,
#       so the barycentre floor `σ_min(Pᵢ − zI) ≥ |z − cᵢ| − nᵢ` gives the explicit disc
#       `D(cᵢ, nᵢ + ρᵢ)` carrying `|Cᵢ|` eigenvalues;
#   * the localized coupling `rᵢ` (Prop. blockgersg, `Ñ = M̃ − Λ` = `remainder`,
#       `β₂ = ‖R₂‖₂/(1−‖R₂‖₂)·‖M̃‖₂`): each block charged only its OWN off-block row, never
#       the worst block's.  Two forms of that row bound are available and BOTH are needed,
#       because neither dominates the other; `rᵢ` is their minimum.  Writing the eigenvector
#       in blocks and taking `i` with `‖xᵢ‖` largest, block row `i` of `(M̃ − zI)x = 0` gives
#           σ_min(Pᵢ − zI)‖xᵢ‖ ≤ ‖Ñ[Cᵢ,:] x̂‖,      x̂ = x with the `Cᵢ` block zeroed,
#       and the right-hand side is bounded either by `Σ_{j≠i} ‖Ñ[Cᵢ,Cⱼ]‖₂ ‖xᵢ‖`, the
#       Feingold–Varga row sum, or by `‖Ñ[Cᵢ,:]‖₂ ‖x̂‖ ≤ √(q−1) ‖Ñ[Cᵢ,:]‖₂ ‖xᵢ‖`, since `x̂`
#       has at most `q−1` nonzero blocks each of norm at most `‖xᵢ‖`.  The bare strip norm
#       `‖Ñ[Cᵢ,:]‖₂`, without the `√(q−1)`, bounds NEITHER and is not rigorous: see
#       `test/test_decompositions/test_svd/test_vbd_block_coupling.jl`, where an eigenvalue
#       of a 10×10 matrix falls outside the union of discs it produces.
# Returns `(coupling, centres, nonnormality, block_slack)` with `block_slack = β_Λ`.
function _vbd_block_data(A::BallMatrix, WB::BallMatrix, YB::BallMatrix,
        transformed::BallMatrix, block::BallMatrix, remainder::BallMatrix,
        clusters::Vector{UnitRange{Int}}, ::Type{T}; FB::BallMatrix = WB) where {T}
    # `FB` is the frame that `Y` approximately inverts: `W` for the standard
    # problem `Ax = λx`, and `B·W` for the pencil `Ax = λBx`. It appears exactly
    # where the frame is paired with `Y` or with the candidate `Λ`; `A·W` is
    # untouched. That single substitution is the whole generalisation.
    R2 = YB * FB - I
    nrmR2 = upper_bound_L2_opnorm(R2)
    nrmR2 < 1 ||
        error("VBD: Neumann condition ‖R₂‖₂ = $nrmR2 ≥ 1 while forming block data")
    denom = setrounding(T, RoundDown) do
        one(T) - nrmR2
    end
    # R₁ = Y(AW − BWΛ) certifies the candidate Λ = block: β_Λ = ‖R₁‖₂/(1−‖R₂‖₂).
    # (B = I recovers the standard R₁ = Y(AW − WΛ).)
    R1 = YB * (A * WB - FB * block)
    block_slack = setrounding(T, RoundUp) do
        upper_bound_L2_opnorm(R1) / denom
    end
    beta2 = setrounding(T, RoundUp) do
        nrmR2 * upper_bound_L2_opnorm(transformed) / denom
    end
    midT = mid(transformed)
    CT = eltype(midT)
    coupling = Vector{T}(undef, length(clusters))
    centres = Vector{CT}(undef, length(clusters))
    nonnormality = Vector{T}(undef, length(clusters))
    q = length(clusters)
    # √(q−1), rounded up, for the strip form of the row bound
    sqrtqm1 = setrounding(T, RoundUp) do
        sqrt(T(max(q - 1, 0)))
    end
    for (k, cl) in enumerate(clusters)
        cj = sum(midT[i, i] for i in cl) / length(cl)        # block barycentre
        centres[k] = cj
        nonnormality[k] = upper_bound_L2_opnorm(transformed[cl, cl] - cj * I)  # ‖Pⱼ − cⱼI‖₂
        rowsum = setrounding(T, RoundUp) do
            acc = zero(T)
            for (j, cj2) in enumerate(clusters)
                j == k && continue
                acc += upper_bound_L2_opnorm(remainder[cl, cj2])
            end
            acc
        end
        strip = setrounding(T, RoundUp) do
            sqrtqm1 * upper_bound_L2_opnorm(remainder[cl, :])
        end
        coupling[k] = setrounding(T, RoundUp) do
            min(rowsum, strip) + beta2
        end
    end
    return coupling, centres, nonnormality, block_slack
end

"""
    schur_gershgorin_enclosure(A::BallMatrix; hermitian = false)

Enclose `σ(M)` for every `M` in the square ball matrix `A` by inflated Gershgorin
discs in a unitary basis of `mid(A)`, grouped into blocks.

The midpoint matrix is reduced by a unitary Schur form, or by an eigenvalue
decomposition when `hermitian = true`; the enclosure is transported to that basis
as `Ã = Z*AZ`, a ball product against the input ball so input radii propagate. `Z`
is unitary only up to rounding, so the residual `R₂ = Z*Z − I` is formed and
`‖R₂‖_∞ < 1` required, and the per-row slack `βᵢ = ‖row_i(R₂)‖₁‖Ã‖_∞/(1−‖R₂‖_∞)`
is added to disc `i` before anything is read off it. The inflated discs are then
clustered by their connectivity graph and the basis permuted so each cluster is
contiguous; the block-diagonal part of `Ã` and its rigorous remainder follow, with
`remainder_norm` a rigorous bound on `‖Ã − block‖₂` (the best of the Collatz and
`‖·‖₁‖·‖∞` estimates).

Each cluster `Cᵢ` also carries a disc `D(cᵢ, nᵢ + rᵢ)` holding `|Cᵢ|` eigenvalues
counted with multiplicity, with barycentre `cᵢ = mean(diag Pᵢ)`, within-block
non-normality `nᵢ = ‖Pᵢ − cᵢI‖₂`, and coupling `rᵢ = min(Σ_{j≠i}‖Ñ[Cᵢ,Cⱼ]‖₂,
√(q−1)‖Ñ[Cᵢ,:]‖₂) + β₂`; that is the block Gershgorin bound of Feingold and Varga
(1962), Theorem 2, in the form used by `prop:blockgersg` of the companion draft.
The bare strip norm `‖Ñ[Cᵢ,:]‖₂` alone is **not** a valid radius and was corrected
in PR #68.

When `hermitian = true` the input is expected Hermitian and the eigenvalues and
intervals are real; otherwise the clusters are discs in the complex plane.

This is **not** either algorithm of Miyajima (2014), whose name the routine
formerly carried. That paper certifies clusters by a Brouwer fixed-point test on a
Newton operator, over an approximate eigendecomposition (§3, radii `u + ⟨u⟩_t t`
from infinity-norm row sums) or a genuine block diagonalisation of `B⁻¹A` (§4);
neither the fixed-point test nor a diagonalising transformation appears here, and
the basis is only permuted. For the block-diagonalising route see
[`miyajima2014a_schurnewton`](@ref).
"""

function schur_gershgorin_enclosure(A::BallMatrix{T, NT}; hermitian::Bool = false) where {T, NT}
    m, n = size(A)
    m == n || throw(ArgumentError("schur_gershgorin_enclosure expects a square matrix"))

    basis, _ = hermitian ? _hermitian_diagonalisation(mid(A)) :
               _schur_diagonalisation(mid(A))
    identity_order = collect(1:n)

    current_basis = basis

    intervals = nothing
    clusters = UnitRange{Int}[]
    transformed = nothing

    order = identity_order
    attempts = 0
    while true
        basis_ball = BallMatrix(current_basis)
        basis_adjoint = BallMatrix(adjoint(current_basis))
        transformed = basis_adjoint * A * basis_ball
        # per-row β (tied to the current row order, hence recomputed after a permutation);
        # discs are inflated BEFORE clustering so the overlap-clustering is itself rigorous.
        R2 = basis_adjoint * basis_ball - I
        beta, _ = _vbd_beta_rows(R2, transformed, T)
        intervals = _inflate_intervals(_vbd_gershgorin_intervals(transformed; hermitian), beta)
        clusters, order = _interval_clusters(intervals)
        order == identity_order && break

        current_basis = current_basis[:, order]
        attempts += 1
        attempts > n &&
            throw(ArgumentError("failed to permute Gershgorin clusters into contiguous blocks"))
    end

    basis = current_basis

    transformed = Base.something(transformed)
    intervals = Base.something(intervals)

    # Recentred (Rayleigh-quotient) eigenvalue estimates: the diagonal of the collapsed
    # `M̃ = Z'AZ`, NOT the candidate `D` from eigen/schur — the candidate drops out (draft
    # Remark "the candidate eigenvalues drop out"; the disc centres are these same `M̃ᵢᵢ`).
    midT = mid(transformed)
    eigenvalues = hermitian ? [real(midT[i, i]) for i in 1:n] : [midT[i, i] for i in 1:n]

    block = _block_diagonal_part(transformed, clusters)
    remainder = transformed - block

    # Rigorous upper bound on ‖remainder‖₂.  We must NOT fold in the
    # block-separation estimate `r2_infty_bound_by_blocks` here: that quantity
    # bounds the Sylvester / invariant-subspace correction ‖X‖ ≈ ‖offdiag‖/sep,
    # which is a *different* and generally *smaller* value than ‖remainder‖₂ for
    # well-separated clusters.  Taking `min(collatz, block_bound)` therefore
    # underestimated ‖remainder‖₂ and was not rigorous.  `upper_bound_L2_opnorm`
    # already returns the best (still rigorous) of the Collatz and ‖·‖₁‖·‖∞ bounds.
    remainder_norm = upper_bound_L2_opnorm(remainder)

    block_coupling, block_centers, block_nonnormality,
    block_residual_norm = _vbd_block_data(
        A, BallMatrix(basis), BallMatrix(adjoint(basis)),
        transformed, block, remainder, clusters, T)
    hermitian && (block_centers = real.(block_centers))

    return SchurGershgorinResult(basis, transformed, block, remainder, clusters,
        intervals, remainder_norm, eigenvalues, block_coupling, block_centers,
        block_nonnormality, block_residual_norm)
end

"""
    block_enclosure(vbd) -> Vector{NamedTuple}

Block-disc enclosure of `σ(A)` from a VBD result (`SchurGershgorinResult` or
`Miyajima2014aSchurNewtonResult`): one disc per cluster, `(center = cᵢ, radius = nᵢ + rᵢ, mult = |Cᵢ|)`
with the barycentre `cᵢ` (`block_centers`), within-block non-normality `nᵢ`
(`block_nonnormality`) and localized coupling `rᵢ` (`block_coupling`).

From the barycentre floor `σ_min(Pᵢ − zI) ≥ |z − cᵢ| − nᵢ` and the block-Gershgorin region
`{z : σ_min(Pᵢ − zI) ≤ rᵢ}`, each disc rigorously contains exactly `|Cᵢ|` eigenvalues of `A`
(a connected component of overlapping discs holds the summed multiplicities).
"""
function block_enclosure(vbd)
    T = eltype(vbd.block_coupling)
    return [(center = vbd.block_centers[k],
                radius = setrounding(T, RoundUp) do
                    vbd.block_nonnormality[k] + vbd.block_coupling[k]
                end,
                mult = length(vbd.clusters[k])) for k in eachindex(vbd.clusters)]
end

function _hermitian_diagonalisation(H::AbstractMatrix{T}) where {T}
    # For Diagonal matrices, eigendecomposition is trivial
    if H isa Diagonal
        n = size(H, 1)
        return (Matrix{T}(I, n, n), diag(H))
    end

    # For BigFloat matrices, try without alg keyword (Julia compat issue)
    # Julia 1.12+ passes alg=RobustRepresentations which BigFloat doesn't support
    if T <: BigFloat || (T <: Complex && real(T) <: BigFloat)
        # Use explicit call without keyword arguments
        try
            eig = eigen(Hermitian(H))
            return (eig.vectors, eig.values)
        catch e
            if e isa MethodError
                # Fallback: convert to Float64, compute, convert back
                H_f64 = convert.(Complex{Float64}, H)
                eig = eigen(Hermitian(H_f64))
                vectors = convert.(T, eig.vectors)
                values = convert.(real(T), eig.values)
                return (vectors, values)
            end
            rethrow(e)
        end
    end

    eig = eigen(Hermitian(H))
    return (eig.vectors, eig.values)
end

function _schur_diagonalisation(A::AbstractMatrix{T}) where {T}
    sch = schur(A)
    return (Matrix(sch.Z), sch.values)
end

function _vbd_gershgorin_intervals(H::BallMatrix{T, NT}; hermitian::Bool) where {T, NT}
    n = size(H, 1)
    midH = mid(H)
    radH = rad(H)

    intervals = hermitian ? Vector{Ball{T, T}}(undef, n) :
                Vector{Ball{T, Complex{T}}}(undef, n)

    absH = upper_abs(H)
    for i in 1:n
        diag_entry = Ball(midH[i, i], radH[i, i])
        diag_ball = hermitian ? _real_interval(diag_entry) : diag_entry

        row_sum = setrounding(T, RoundUp) do
            s = zero(T)
            for j in 1:n
                if j != i
                    s += absH[i, j]
                end
            end
            s
        end

        radius = setrounding(T, RoundUp) do
            rad(diag_ball) + row_sum
        end
        intervals[i] = Ball(mid(diag_ball), radius)
    end

    return intervals
end

_real_interval(x::Ball{T, T}) where {T} = x

function _real_interval(x::Ball{T, Complex{T}}) where {T}
    radius = setrounding(T, RoundUp) do
        rad(x) + abs(imag(mid(x)))
    end
    return Ball(real(mid(x)), radius)
end

function _interval_clusters(intervals::Vector{Ball{T, T}}) where {T}
    return _interval_clusters_generic(intervals)
end

function _interval_clusters(intervals::Vector{Ball{T, Complex{T}}}) where {T}
    return _interval_clusters_generic(intervals)
end

function _interval_clusters_generic(intervals)
    components = overlap_components(intervals)
    order = isempty(components) ? Int[] : vcat(components...)
    clusters = UnitRange{Int}[]
    start = 1
    for comp in components
        len = length(comp)
        push!(clusters, start:(start + len - 1))
        start += len
    end
    return clusters, (isempty(order) ? Int[] : order)
end

_balls_overlap(a::Ball{T, T}, b::Ball{T, T}) where {T} = intersect_ball(a, b) !== nothing

function _balls_overlap(a::Ball{T, Complex{T}}, b::Ball{T, Complex{T}}) where {T}
    # For conservative overlap detection (no false negatives) we need a
    # rigorous lower bound on the distance between centres.  RoundDown on
    # abs values followed by RoundDown hypot gives distance_down ≤ true
    # distance, so distance_down ≤ threshold_up guarantees detection of
    # every genuine overlap.
    # `hypot` is not correctly rounded, so it is not a lower bound even under RoundDown; the
    # difference of the centres is bracketed componentwise and its modulus bounded with abs_down
    ca, cb = mid(a), mid(b)
    x = max(sub_down(real(ca), real(cb)), sub_down(real(cb), real(ca)), zero(T))
    y = max(sub_down(imag(ca), imag(cb)), sub_down(imag(cb), imag(ca)), zero(T))
    distance = abs_down(complex(x, y))
    threshold = add_up(rad(a), rad(b))
    return distance <= threshold
end

function _block_diagonal_part(H::BallMatrix, clusters::Vector{UnitRange{Int}})
    midH = mid(H)
    radH = rad(H)
    block_mid = zeros(eltype(midH), size(midH))
    block_rad = zeros(eltype(radH), size(radH))

    for cluster in clusters
        block_mid[cluster, cluster] .= midH[cluster, cluster]
        block_rad[cluster, cluster] .= radH[cluster, cluster]
    end

    return BallMatrix(block_mid, block_rad)
end

function overlap_components(balls::AbstractVector{Ball{T, CT}}) where {T, CT}
    n = length(balls)
    components = Vector{Vector{Int}}()
    n == 0 && return components

    adjacency = [Int[] for _ in 1:n]
    for i in 1:(n - 1)
        for j in (i + 1):n
            if _balls_overlap(balls[i], balls[j])
                push!(adjacency[i], j)
                push!(adjacency[j], i)
            end
        end
    end

    seen = falses(n)
    for s in 1:n
        seen[s] && continue
        stack = [s]
        seen[s] = true
        component = Int[s]
        while !isempty(stack)
            v = pop!(stack)
            for w in adjacency[v]
                if !seen[w]
                    seen[w] = true
                    push!(stack, w)
                    push!(component, w)
                end
            end
        end
        sort!(component)
        push!(components, component)
    end

    return components
end

function sep_clusters(ints, compA::UnitRange{Int}, compB::UnitRange{Int})
    hullA = reduce(ball_hull, ints[compA])
    hullB = reduce(ball_hull, ints[compB])
    T = radtype(typeof(hullA))

    cA, cB = mid(hullA), mid(hullB)
    rA, rB = rad(hullA), rad(hullB)

    # Rigorous lower bound on the distance between centres.
    distance_down = setrounding(T, RoundDown) do
        diff = cA - cB
        hypot(abs(real(diff)), abs(imag(diff)))
    end

    # Rigorous upper bound on the sum of the hull radii.
    radii_up = setrounding(T, RoundUp) do
        rA + rB
    end

    # Rigorous lower bound on the gap.
    gap = setrounding(T, RoundDown) do
        distance_down - radii_up
    end

    return gap <= zero(T) ? zero(T) : gap
end

function r2_infty_bound_by_blocks(H::BallMatrix{T}, intervals, clusters::Vector{UnitRange{Int}}) where {T}
    P = length(clusters)
    P == 0 && return zero(T)

    absH = upper_abs(H)
    row_sums = zeros(T, P)

    for p in 1:P
        for q in 1:P
            p == q && continue
            sep = sep_clusters(intervals, clusters[p], clusters[q])
            if sep == zero(T)
                return convert(T, Inf)
            end
            E∞ = block_infty_upper(absH, clusters[p], clusters[q])
            contribution = setrounding(T, RoundUp) do
                E∞ / sep
            end
            row_sums[p] = setrounding(T, RoundUp) do
                row_sums[p] + contribution
            end
        end
    end

    return maximum(row_sums)
end

function block_infty_upper(absH::AbstractMatrix{T}, rows, cols) where {T}
    smax = zero(T)
    for i in rows
        s = setrounding(T, RoundUp) do
            acc = zero(T)
            for j in cols
                acc += absH[i, j]
            end
            acc
        end
        smax = max(smax, s)
    end
    return smax
end
