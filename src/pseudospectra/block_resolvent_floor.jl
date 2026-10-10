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
eigenvalue problem), a lower bound of `σ_min(A − zI)` that costs `O(number of blocks)` at each
`z`. For a ball matrix `A` the bound holds for every matrix of the ball.

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
    W = Matrix(vbd.basis)
    T = eltype(vbd.block_coupling)
    bW = BallMatrix(W)
    bY = BallMatrix(inv(W))
    ρ = upper_bound_L2_opnorm(bY * bW - I)
    kappa = ρ < 1 ? mul_up(T(_frame_norm(bW)), div_up(T(_frame_norm(bY)), sub_down(one(T), T(ρ)))) :
            T(Inf)
    blocks = BallMatrix[vbd.block_diagonal[cl, cl] for cl in vbd.clusters]
    return BlockResolventFloor{T}(Complex{T}.(vbd.block_centers),
        Vector{T}(vbd.block_nonnormality), Vector{T}(vbd.block_coupling), blocks,
        T(vbd.block_residual_norm), T(kappa))
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
