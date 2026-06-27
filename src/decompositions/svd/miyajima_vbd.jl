"""
    MiyajimaVBDResult

Container returned by [`miyajima_vbd`](@ref) encapsulating the data
produced by the verified block diagonalisation (VBD) step. The fields
contain the basis that block diagonalises the midpoint matrix, the
transformed enclosure, its block-diagonal truncation, the rigorous
remainder, and the Gershgorin clusters that certify how the spectrum
groups together.
"""
struct MiyajimaVBDResult{MT, BT, IT, RT, ET}
    """Basis that block diagonalises `mid(A)`."""
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
(so that consumers must use `inv(basis)`).  `MiyajimaVBDResult` (NSD / Schur /
Hermitian eigenvectors) is unitary; `SchurNewtonVBDResult` is not.
"""
_vbd_unitary_basis(::MiyajimaVBDResult) = true

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
#   * the sharper localized coupling `rᵢ = ‖Ñ[Cᵢ,:]‖₂ + β₂` (Prop. blockgersg, `Ñ = M̃ − Λ`
#       = `remainder`, `β₂ = ‖R₂‖₂/(1−‖R₂‖₂)·‖M̃‖₂`): each block charged only its OWN
#       off-block-row coupling, never the worst block's (`rᵢ ≤ ‖Ñ‖₂ + β₂`, never looser).
# Returns `(coupling, centres, nonnormality, block_slack)` with `block_slack = β_Λ`.
function _vbd_block_data(A::BallMatrix, WB::BallMatrix, YB::BallMatrix,
        transformed::BallMatrix, block::BallMatrix, remainder::BallMatrix,
        clusters::Vector{UnitRange{Int}}, ::Type{T}) where {T}
    R2 = YB * WB - I
    nrmR2 = upper_bound_L2_opnorm(R2)
    nrmR2 < 1 ||
        error("VBD: Neumann condition ‖R₂‖₂ = $nrmR2 ≥ 1 while forming block data")
    denom = setrounding(T, RoundDown) do
        one(T) - nrmR2
    end
    # R₁ = Y(AW − WΛ) certifies the candidate Λ = block: β_Λ = ‖R₁‖₂/(1−‖R₂‖₂).
    R1 = YB * (A * WB - WB * block)
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
    for (k, cl) in enumerate(clusters)
        cj = sum(midT[i, i] for i in cl) / length(cl)        # block barycentre
        centres[k] = cj
        nonnormality[k] = upper_bound_L2_opnorm(transformed[cl, cl] - cj * I)  # ‖Pⱼ − cⱼI‖₂
        coupling[k] = setrounding(T, RoundUp) do
            upper_bound_L2_opnorm(remainder[cl, :]) + beta2
        end
    end
    return coupling, centres, nonnormality, block_slack
end

"""
    miyajima_vbd(A::BallMatrix; hermitian = false)

Perform Miyajima's verified block diagonalisation (VBD) on the square
ball matrix `A`.  The midpoint matrix is reduced either by an eigenvalue
decomposition (when `hermitian = true`) or by a unitary Schur form (for
the general case).  The enclosure is transported to that basis, the
Gershgorin discs are clustered, and a block-diagonal truncation together
with a rigorous remainder is produced.

Overlapping discs are grouped via their connectivity graph so that each
cluster becomes contiguous after a basis permutation.  The remainder bound
is a rigorous upper bound on `‖transformed - block_diagonal‖₂` (the best of
the Collatz and `‖·‖₁‖·‖∞` interpolation estimates).

When `hermitian = true` the routine expects `A` to be Hermitian and the
resulting eigenvalues and intervals are real.  Otherwise the Schur form
is used and the clusters are discs in the complex plane.
"""

function miyajima_vbd(A::BallMatrix{T, NT}; hermitian::Bool = false) where {T, NT}
    m, n = size(A)
    m == n || throw(ArgumentError("miyajima_vbd expects a square matrix"))

    basis, _ = hermitian ? _hermitian_diagonalisation(mid(A)) : _schur_diagonalisation(mid(A))
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
        attempts > n && throw(ArgumentError("failed to permute Gershgorin clusters into contiguous blocks"))
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

    block_coupling, block_centers, block_nonnormality, block_residual_norm = _vbd_block_data(
        A, BallMatrix(basis), BallMatrix(adjoint(basis)), transformed, block, remainder, clusters, T)
    hermitian && (block_centers = real.(block_centers))

    return MiyajimaVBDResult(basis, transformed, block, remainder, clusters,
        intervals, remainder_norm, eigenvalues, block_coupling, block_centers,
        block_nonnormality, block_residual_norm)
end

"""
    block_enclosure(vbd) -> Vector{NamedTuple}

Block-disc enclosure of `σ(A)` from a VBD result (`MiyajimaVBDResult` or
`SchurNewtonVBDResult`): one disc per cluster, `(center = cᵢ, radius = nᵢ + rᵢ, mult = |Cᵢ|)`
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

    intervals = hermitian ? Vector{Ball{T, T}}(undef, n) : Vector{Ball{T, Complex{T}}}(undef, n)

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
    distance = setrounding(T, RoundDown) do
        diff = mid(a) - mid(b)
        hypot(abs(real(diff)), abs(imag(diff)))
    end
    threshold = setrounding(T, RoundUp) do
        rad(a) + rad(b)
    end
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
    for i in 1:n-1
        for j in i+1:n
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

