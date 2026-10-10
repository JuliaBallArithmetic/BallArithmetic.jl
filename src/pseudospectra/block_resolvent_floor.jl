# A lower bound of σ_min(A − zI), cheap at each z, from one verified block diagonalisation of A.

"""
    BlockResolventFloor{T}

The quantities of [`block_resolvent_floor`](@ref) that do not depend on `z`: per block the
barycentre `centres[j]`, `nonnormality[j] ≥ ‖P_j − c_j I‖₂`, the coupling radius
`coupling[j] ≥ ‖(M − Λ)[C_j, :]‖₂` and the ball matrix `blocks[j]` containing `P_j`; the slack
`slack ≥ ‖M − Λ‖₂`; and `kappa ≥ ‖W‖₂‖W⁻¹‖₂`. Evaluate with [`sigma_min_floor`](@ref) or
[`resolvent_bound`](@ref).
"""
struct BlockResolventFloor{T <: AbstractFloat}
    centres::Vector{Complex{T}}
    nonnormality::Vector{T}
    coupling::Vector{T}
    blocks::Vector{BallMatrix}
    slack::T
    kappa::T
end

"""
    block_resolvent_floor(vbd) -> BlockResolventFloor

Prepare, from a verified block diagonalisation `vbd` of `A` (the result of
[`miyajima2014a_schurnewton`](@ref) or of [`schur_gershgorin_enclosure`](@ref), for the standard
eigenvalue problem; for a result of [`verifyeigall`](@ref) see the method below), a lower bound
of `σ_min(A − zI)` that costs `O(number of blocks)` at each `z`. For a ball matrix `A` the bound holds for every matrix of the ball.

# Setting

`W` is the basis of `vbd`, `M := W⁻¹AW`, and `Λ = blockdiag(P_j)` its block diagonal part, with
blocks on the index sets `C_j`. The result carries `β_Λ ≥ ‖M − Λ‖₂` (`block_residual_norm`, from
`M − Λ = (I + R₂)⁻¹R₁`, `R₁ = Y(AW − WΛ)`, `R₂ = YW − I`), the radii `r_j ≥ ‖(M − Λ)[C_j, :]‖₂`
(`block_coupling`), the barycentres `c_j` and `n_j ≥ ‖P_j − c_j I‖₂`.

# The two bounds

With `s_j(z) ≤ σ_min(P_j − zI)` and `s(z) = min_j s_j(z)`:

1. `σ_min(M − zI) ≥ s(z) − β_Λ`, since `σ_min(Λ − zI) = min_j σ_min(P_j − zI)` and `σ_min` is
   1-Lipschitz in the spectral norm (`|σ_min(X + E) − σ_min(X)| ≤ ‖E‖₂`).
2. If every `s_j(z) > 0` and `τ(z)² := Σ_j (r_j / s_j(z))² < 1`, then
   `σ_min(M − zI) ≥ s(z)(1 − τ(z))`: factor
   `M − zI = (Λ − zI)(I + G)`, `G = (Λ − zI)⁻¹(M − Λ)`; the block row `j` of `G` is
   `(P_j − zI)⁻¹(M − Λ)[C_j, :]`, of norm at most `r_j/s_j`, and a matrix whose block rows have
   norms `g_j` has norm at most `√(Σ g_j²)`. Each block is charged its own coupling, so this
   bound is the larger one where the coupling is uneven across the blocks.

Both are carried to `A` by `A − zI = W(M − zI)W⁻¹`:

    σ_min(A − zI) ≥ σ_min(M − zI) / κ₂(W).

`κ₂(W)` is bounded here, not taken from `vbd.kappa` (a diagnostic there): with `Y` the
floating-point inverse of `W` and `ρ ≥ ‖YW − I‖₂ < 1`, `‖W⁻¹‖₂ ≤ ‖Y‖₂/(1 − ρ)`.

`s_j(z)` is `|z − c_j| − n_j` (from `P_j − zI = (c_j − z)I + (P_j − c_j I)`), or, on request, the
larger of that and a certified smallest singular value of the ball `P_j − zI`.

The statements are Lemma "Block resolvent floor" and Proposition "Localized block-dominance
floor" of I. Nisoli, *A certified resolvent bound via verified block diagonalization*
(unpublished draft, 2026); the arguments are the ones above.
"""
function block_resolvent_floor(vbd)
    hasproperty(vbd, :pencil) && vbd.pencil && throw(ArgumentError(
        "block_resolvent_floor: the block diagonalisation is of a pencil; the bound is for the " *
        "standard problem"))
    T = eltype(vbd.block_coupling)
    blocks = BallMatrix[vbd.block_diagonal[cl, cl] for cl in vbd.clusters]
    return BlockResolventFloor{T}(Complex{T}.(vbd.block_centers),
        Vector{T}(vbd.block_nonnormality), Vector{T}(vbd.block_coupling), blocks,
        T(vbd.block_residual_norm), _frame_kappa(BallMatrix(Matrix(vbd.basis)), T))
end

"""
    block_resolvent_floor(r::VerifyEigAllResult) -> BlockResolventFloor

The same bound from a result of [`verifyeigall`](@ref) with one of the methods of Rump (2022),
with the frame `W := S`, the similarity the algorithm ended with, and the blocks on its clusters.

`S` is a product of floating-point matrices, contained in the ball matrix `r.similarity`, and
`r.transformed` is a ball matrix containing `M = S⁻¹AS`, so the quantities of the bound are read from it and no residual is formed:
`P_j` is its block on the cluster `C_j`, `c_j` the barycentre of the midpoints of the diagonal of
that block, `n_j ≥ ‖P_j − c_j I‖₂`, `r_j ≥ ‖(M − Λ)[C_j, :]‖₂` the norm of the block row with the
block itself set to zero, and `β_Λ ≥ ‖M − Λ‖₂` the norm of the ball matrix with every diagonal
block set to zero. `κ₂(S)` is bounded over the ball `r.similarity` as above.

The bound uses the enclosure of `S⁻¹AS` and a partition, and neither the self-mapping test (2.10)
nor `r.spectrum_covered`: it holds whether or not the clusters are certified. When the
transformation failed, `r.transformed` is `A` and the frame is the identity.

S. M. Rump, *Verified error bounds for all eigenvalues and eigenvectors of a matrix*,
SIAM J. Matrix Anal. Appl. **43**(4):1736-1754, 2022, doi 10.1137/21M1451440, for the enclosure of
`S⁻¹AS`; the bound is the one of the method above.
"""
function block_resolvent_floor(r::VerifyEigAllResult{T, CT}) where {T, CT}
    M = r.transformed
    n = size(M, 1)
    n > 0 || throw(ArgumentError(
        "block_resolvent_floor: the result carries no transformed matrix; it comes from a " *
        "method other than those of Rump (2022)"))
    Mm, Mr = mid(M), rad(M)
    Nm, Nr = copy(Mm), copy(Mr)
    for cl in r.clusters
        Nm[cl, cl] .= zero(CT)
        Nr[cl, cl] .= zero(T)
    end
    N = BallMatrix(Nm, Nr)
    p = length(r.clusters)
    centres = Vector{CT}(undef, p)
    nonnormality = Vector{T}(undef, p)
    coupling = Vector{T}(undef, p)
    blocks = Vector{BallMatrix}(undef, p)
    for (j, cl) in enumerate(r.clusters)
        P = BallMatrix(Mm[cl, cl], Mr[cl, cl])
        centres[j] = sum(Mm[i, i] for i in cl) / length(cl)
        nonnormality[j] = upper_bound_L2_opnorm(P - Ball(centres[j], zero(T)) * I)
        coupling[j] = upper_bound_L2_opnorm(BallMatrix(Nm[cl, :], Nr[cl, :]))
        blocks[j] = P
    end
    return BlockResolventFloor{T}(centres, nonnormality, coupling, blocks,
        T(upper_bound_L2_opnorm(N)), _frame_kappa(r.similarity, T))
end

# Upper bound of κ₂(W) over a ball matrix W: with Y the floating-point inverse of its midpoint and
# ρ ≥ ‖YW − I‖₂ < 1 over the ball, every W of it is invertible and ‖W⁻¹‖₂ ≤ ‖Y‖₂/(1 − ρ).
# `Inf` when ρ ≥ 1.
function _frame_kappa(bW::BallMatrix, ::Type{T}) where {T}
    bY = BallMatrix(inv(mid(bW)))
    ρ = upper_bound_L2_opnorm(bY * bW - I)
    ρ < 1 || return T(Inf)
    return mul_up(T(_frame_norm(bW)), div_up(T(_frame_norm(bY)), sub_down(one(T), T(ρ))))
end

# Upper bound of the spectral norm of a frame: the smaller of the bound from a verified singular
# value computation and of the cheap ones. For a frame with nearly orthonormal columns the cheap
# bounds (through |W|, or √(‖W‖₁‖W‖_∞)) exceed the norm by a factor that grows like √n.
function _frame_norm(M::BallMatrix)
    cheap = upper_bound_L2_opnorm(M)
    tight = try
        svd_bound_L2_opnorm(M)
    catch
        cheap
    end
    return (isfinite(tight) && tight > 0) ? min(cheap, tight) : cheap
end

# lower bound of σ_min(P − zI) over the ball P, by a verified singular value computation
# (a modulus for a 1×1 block)
function _block_sigma_min_lower(P::BallMatrix, z::Complex{T}) where {T}
    if size(P, 1) == 1
        d = P[1, 1] - Ball(z, zero(T))
        return sub_down(_abs_lo(mid(d)), rad(d))
    end
    σ = svdbox(P - Ball(z, zero(T)) * I)
    return minimum(sub_down(mid(s), rad(s)) for s in σ)
end

"""
    sigma_min_floor(f::BlockResolventFloor, z; near = false)

A lower bound of `σ_min(A − zI)`: the larger of the two bounds of
[`block_resolvent_floor`](@ref), or zero when neither is positive. With `near = false` the blocks
enter through `|z − c_j| − n_j` only, at `O(number of blocks)` operations. With `near = true` a
certified smallest singular value of `P_j − zI` is computed for each block of size larger than
one, which is what gives a positive bound inside the disc `|z − c_j| ≤ n_j` of a non-normal block.
Every operation is rounded toward a smaller result.
"""
function sigma_min_floor(f::BlockResolventFloor{T}, z::Number; near::Bool = false) where {T}
    isfinite(f.kappa) || return zero(T)
    zc = Complex{T}(z)
    p = length(f.centres)
    s = Vector{T}(undef, p)
    for j in 1:p
        s[j] = sub_down(dist_down(zc, f.centres[j]), f.nonnormality[j])
        near && (s[j] = max(s[j], _block_sigma_min_lower(f.blocks[j], zc)))
    end
    smin = minimum(s)
    smin > 0 || return zero(T)
    best = sub_down(smin, f.slack)                                  # bound 1
    τ2 = zero(T)
    for j in 1:p
        q = div_up(f.coupling[j], s[j])
        τ2 = add_up(τ2, mul_up(q, q))
    end
    τ = sqrt_up(τ2)
    τ < 1 && (best = max(best, mul_down(smin, sub_down(one(T), τ))))  # bound 2
    return best > 0 ? div_down(best, f.kappa) : zero(T)
end

"""
    resolvent_bound(f::BlockResolventFloor, z; near = false)

An upper bound of `‖(A − zI)⁻¹‖₂`: the reciprocal, rounded up, of
[`sigma_min_floor`](@ref); `Inf` where that is zero.
"""
function resolvent_bound(f::BlockResolventFloor{T}, z::Number; near::Bool = false) where {T}
    s = sigma_min_floor(f, z; near)
    return s > 0 ? div_up(one(T), s) : T(Inf)
end

export BlockResolventFloor, block_resolvent_floor, sigma_min_floor, resolvent_bound
