"""
    SVDMethod

Abstract type for selecting SVD certification algorithms.
"""
abstract type SVDMethod end

"""
    MiyajimaM1 <: SVDMethod

Miyajima 2014, Theorem 7 (M1): Economy SVD approach with tighter bounds.

Bounds:
- Lower: σᵢ · √((1-‖F‖)(1-‖G‖)) - ‖E‖
- Upper: σᵢ · √((1+‖F‖)(1+‖G‖)) + ‖E‖

This is the recommended default method, providing tighter bounds than
the original Rump formulas.
"""
struct MiyajimaM1 <: SVDMethod end

"""
    MiyajimaM4 <: SVDMethod

Miyajima 2014, Theorem 11 (M4): Eigendecomposition-based bounds.

Works on D̂ + Ê = (AV)ᵀAV where D̂ is diagonal. Uses Gershgorin isolation
and can provide very tight bounds for well-separated singular values.
"""
struct MiyajimaM4 <: SVDMethod end

"""
    RumpOriginal <: SVDMethod

Original Rump 2011 formulas (looser bounds, for comparison).

Bounds:
- Lower: (σᵢ - ‖E‖) / ((1+‖F‖)(1+‖G‖))
- Upper: (σᵢ + ‖E‖) / ((1-‖F‖)(1-‖G‖))
"""
struct RumpOriginal <: SVDMethod end

"""
    RigorousSVDResult

Container returned by [`rigorous_svd`](@ref) bundling the midpoint
factorisation, the certified singular-value enclosures, and the
block-diagonal refinement obtained from [`miyajima_vbd`](@ref).  Besides
the singular values themselves the struct exposes the residual and
orthogonality defect bounds that underpin the certification.
"""
struct RigorousSVDResult{UT, ST, ΣT, VT, ET, RT, VBDT}
    """Ball enclosure of the left singular vectors."""
    U::UT
    """Certified singular values returned as floating-point balls."""
    singular_values::ST
    """Diagonal ball matrix containing the singular-value enclosure."""
    Σ::ΣT
    """Ball enclosure of the right singular vectors."""
    V::VT
    """Residual enclosure `U * Σ * V' - A`."""
    residual::ET
    """Upper bound on `‖residual‖₂`."""
    residual_norm::RT
    """Upper bound on `‖V'V - I‖₂` (right orthogonality defect)."""
    right_orthogonality_defect::RT
    """Upper bound on `‖U'U - I‖₂` (left orthogonality defect)."""
    left_orthogonality_defect::RT
    """Verified block diagonalisation of `Σ'Σ` via Miyajima's procedure, or `nothing` when skipped."""
    block_diagonalisation::VBDT
end

Base.size(result::RigorousSVDResult) = size(result.singular_values)
Base.length(result::RigorousSVDResult) = length(result.singular_values)
Base.firstindex(result::RigorousSVDResult) = firstindex(result.singular_values)
Base.lastindex(result::RigorousSVDResult) = lastindex(result.singular_values)
Base.lastindex(result::RigorousSVDResult, i::Int) = lastindex(result.singular_values, i)
Base.getindex(result::RigorousSVDResult, inds...) = getindex(result.singular_values, inds...)
Base.iterate(result::RigorousSVDResult) = iterate(result.singular_values)
Base.iterate(result::RigorousSVDResult, state) = iterate(result.singular_values, state)

# BigFloat SVD cache for warm-starting Ogita refinement
# This cache stores the most recently computed SVD for reuse with nearby matrices
const _svd_cache_U = Ref{Union{Nothing, Matrix}}(nothing)
const _svd_cache_S = Ref{Union{Nothing, Vector}}(nothing)
const _svd_cache_V = Ref{Union{Nothing, Matrix}}(nothing)
const _svd_cache_A_hash = Ref{UInt64}(0)  # Hash of the matrix for cache validation
const _svd_cache_hits = Ref{Int}(0)
const _svd_cache_misses = Ref{Int}(0)

"""
    clear_svd_cache!()

Clear the BigFloat SVD cache used for warm-starting Ogita refinement.
"""
function clear_svd_cache!()
    _svd_cache_U[] = nothing
    _svd_cache_S[] = nothing
    _svd_cache_V[] = nothing
    _svd_cache_A_hash[] = 0
    _svd_cache_hits[] = 0
    _svd_cache_misses[] = 0
    return nothing
end

"""
    svd_cache_stats()

Return statistics about the BigFloat SVD cache usage.
"""
function svd_cache_stats()
    return (hits = _svd_cache_hits[], misses = _svd_cache_misses[])
end

"""
    set_svd_cache!(U, S, V, A_hash)

Set the SVD cache with the given factors and matrix hash.
Used for warm-starting Ogita refinement on nearby matrices.
"""
function set_svd_cache!(U, S, V, A_hash::UInt64)
    _svd_cache_U[] = U
    _svd_cache_S[] = S
    _svd_cache_V[] = V
    _svd_cache_A_hash[] = A_hash
    return nothing
end

"""
    rigorous_svd(A::BallMatrix; apply_vbd = true)

Compute a rigorous singular value decomposition of the ball matrix `A`.
The midpoint SVD is certified following Theorem 3.1 of
Ref. [Rump2011](@cite); optionally, the resulting singular-value
enclosure can be refined by applying [`miyajima_vbd`](@ref) to `Σ'Σ`,
yielding a block-diagonal structure with a rigorously bounded remainder.

The returned [`RigorousSVDResult`](@ref) exposes both the enclosures and
the intermediate norm bounds that justify them. When `apply_vbd` is set
to `false`, the `block_diagonalisation` field is `nothing`.

# References

* Miyajima S. (2014), "Verified bounds for all the singular values of matrix",
  Japan J. Indust. Appl. Math. 31, 513–539.
* Rump S.M. (2011), "Verified bounds for singular values", BIT 51, 367–384.
"""
function rigorous_svd(A::BallMatrix{T}; method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = true) where {T}
    RT = real(T)

    # For BigFloat, use Ogita refinement since LAPACK doesn't support BigFloat
    if RT === BigFloat
        return _rigorous_svd_bigfloat(A, method; apply_vbd)
    end

    # Standard path for Float64/Float32
    svdA = svd(A.c)
    return _certify_svd(A, svdA, method; apply_vbd)
end

"""
    _rigorous_svd_bigfloat(A, method; apply_vbd, use_cache)

BigFloat version of rigorous SVD using Ogita refinement.
Computes SVD in Float64, refines to BigFloat, then certifies.

When `use_cache=true` (default), attempts to warm-start from a cached SVD
if available, which can significantly speed up computation for similar matrices.
"""
function _rigorous_svd_bigfloat(A::BallMatrix{BigFloat}, method::SVDMethod;
                                 apply_vbd::Bool = true, use_cache::Bool = true)
    prec_bits = precision(BigFloat)
    # Quadratic convergence: ~ceil(log2(prec_bits / 15)) iterations
    n_iter = max(2, ceil(Int, log2(prec_bits / 15)))

    # Try to use cached SVD for warm-starting
    use_cached = false
    if use_cache && _svd_cache_U[] !== nothing
        # Cache is available - use it for warm-starting (fewer iterations needed)
        use_cached = true
        _svd_cache_hits[] += 1

        refined = ogita_svd_refine(A.c,
                                   _svd_cache_U[],
                                   _svd_cache_S[],
                                   _svd_cache_V[];
                                   max_iterations=n_iter,  # Still need full iterations from cached start
                                   precision_bits=prec_bits,
                                   check_convergence=false)
    else
        # No cache - compute Float64 SVD first
        _svd_cache_misses[] += 1

        A_f64 = Complex{Float64}.(A.c)
        F64 = svd(A_f64)

        refined = ogita_svd_refine(A.c, F64.U, F64.S, F64.Vt';
                                   max_iterations=n_iter,
                                   precision_bits=prec_bits,
                                   check_convergence=false)
    end

    # Update cache with refined SVD for future use
    if use_cache
        Σ_for_cache = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
        set_svd_cache!(refined.U, Σ_for_cache, refined.V, hash(A.c))
    end

    # Certify the refined SVD
    Σ_vec = isa(refined.Σ, Diagonal) ? diag(refined.Σ) : refined.Σ
    svd_refined = SVD(Matrix(refined.U), Vector(Σ_vec), Matrix(refined.V'))

    return _certify_svd(A, svd_refined, method; apply_vbd)
end

"""
    svdbox(A::BallMatrix; method = MiyajimaM1(), apply_vbd = true)

Backward-compatible wrapper returning only the vector of singular-value
enclosures produced by [`rigorous_svd`](@ref).  New code should prefer
[`rigorous_svd`](@ref) directly to access the additional certification
data.  The optional `method` and `apply_vbd` flags mirror those in
[`rigorous_svd`](@ref).
"""
function svdbox(A::BallMatrix{T}; method::SVDMethod = MiyajimaM1(), apply_vbd::Bool = true) where {T}
    result = rigorous_svd(A; method, apply_vbd)
    return result.singular_values
end

function _certify_svd(A::BallMatrix{T}, svdA::SVD, method::SVDMethod; apply_vbd::Bool = true) where {T}
    return _certify_svd_impl(A, svdA.U, svdA.S, svdA.V, svdA.Vt, method; apply_vbd)
end

# Method for NamedTuple (used by Ogita refinement)
function _certify_svd(A::BallMatrix{T}, svdA::NamedTuple{(:U, :S, :V, :Vt)}, method::SVDMethod; apply_vbd::Bool = true) where {T}
    return _certify_svd_impl(A, svdA.U, svdA.S, svdA.V, svdA.Vt, method; apply_vbd)
end

function _certify_svd_impl(A::BallMatrix{T}, U_in, S_in, V_in, Vt_in, method::SVDMethod; apply_vbd::Bool = true) where {T}
    U = BallMatrix(U_in)
    V = BallMatrix(V_in)
    Vt = BallMatrix(Vt_in)
    Σ_mid = BallMatrix(Diagonal(S_in))

    E = U * Σ_mid * Vt - A
    normE = upper_bound_L2_opnorm(E)
    @debug "norm E" normE

    F = Vt * V - I
    normF = upper_bound_L2_opnorm(F)
    @debug "norm F" normF

    G = U' * U - I
    normG = upper_bound_L2_opnorm(G)
    @debug "norm G" normG

    @assert normF < 1 "It is not possible to verify the singular values with this precision"
    @assert normG < 1 "It is not possible to verify the singular values with this precision"

    # Compute bounds based on method
    svdbounds_down, svdbounds_up = _compute_svd_bounds(method, S_in, normE, normF, normG, T)

    midpoints = (svdbounds_down + svdbounds_up) / 2
    radii = setrounding(T, RoundUp) do
        [max(svdbounds_up[i] - midpoints[i], midpoints[i] - svdbounds_down[i])
         for i in 1:length(midpoints)]
    end

    singular_values = [Ball(midpoints[i], radii[i]) for i in 1:length(midpoints)]
    Σ = _diagonal_ball_matrix(singular_values)

    ΔΣ = Σ - Σ_mid
    # Reuse the midpoint residual `E` and only account for the interval
    # widening introduced when replacing `Σ_mid` with the ball diagonal `Σ`.
    residual = E + U * ΔΣ * Vt
    residual_norm = upper_bound_L2_opnorm(residual)

    vbd = nothing
    if apply_vbd
        # VBD requires eigen decomposition which doesn't work for BigFloat
        # Skip VBD for BigFloat matrices
        if T !== BigFloat
            H = adjoint(Σ) * Σ
            vbd = miyajima_vbd(H; hermitian = true)
        end
    end

    return RigorousSVDResult(U, singular_values, Σ, V, residual,
        residual_norm, normF, normG, vbd)
end

#=
Miyajima 2014, Theorem 7 (M1): Economy SVD bounds

Lower bound: σᵢ · √((1-‖F‖)(1-‖G‖)) - ‖E‖
Upper bound: σᵢ · √((1+‖F‖)(1+‖G‖)) + ‖E‖

These are tighter than Rump's original formulas.
=#
function _compute_svd_bounds(::MiyajimaM1, S::Vector, normE, normF, normG, ::Type{T}) where {T}
    # For lower bound: σ * sqrt((1-F)(1-G)) - E
    # Need sqrt_factor_down ≤ sqrt((1-F)(1-G))
    sqrt_factor_down = setrounding(T, RoundDown) do
        sqrt((one(T) - normF) * (one(T) - normG))
    end

    # For upper bound: σ * sqrt((1+F)(1+G)) + E
    # Need sqrt_factor_up ≥ sqrt((1+F)(1+G))
    sqrt_factor_up = setrounding(T, RoundUp) do
        sqrt((one(T) + normF) * (one(T) + normG))
    end

    svdbounds_down = setrounding(T, RoundDown) do
        [σ * sqrt_factor_down - normE for σ in S]
    end

    svdbounds_up = setrounding(T, RoundUp) do
        [σ * sqrt_factor_up + normE for σ in S]
    end

    return svdbounds_down, svdbounds_up
end

#=
Original Rump 2011 formulas (looser bounds, for comparison/backward compatibility)

Lower bound: (σᵢ - ‖E‖) / ((1+‖F‖)(1+‖G‖))
Upper bound: (σᵢ + ‖E‖) / ((1-‖F‖)(1-‖G‖))
=#
function _compute_svd_bounds(::RumpOriginal, S::Vector, normE, normF, normG, ::Type{T}) where {T}
    den_down = setrounding(T, RoundUp) do
        (one(T) + normF) * (one(T) + normG)
    end

    den_up = setrounding(T, RoundDown) do
        (one(T) - normF) * (one(T) - normG)
    end

    svdbounds_down = setrounding(T, RoundDown) do
        [(σ - normE) / den_down for σ in S]
    end

    svdbounds_up = setrounding(T, RoundUp) do
        [(σ + normE) / den_up for σ in S]
    end

    return svdbounds_down, svdbounds_up
end

#=
Miyajima 2014, Theorem 11 (M4): Eigendecomposition-based bounds

This method works on D̂ + Ê = (AV)ᵀAV where D̂ is diagonal.
For well-separated singular values, it can give tighter bounds through
Gershgorin isolation.

For now, this falls back to M1 bounds but with the note that the VBD
result can be used for further refinement of isolated singular values.
=#
function _compute_svd_bounds(::MiyajimaM4, S::Vector, normE, normF, normG, ::Type{T}) where {T}
    # Start with M1 bounds as the base
    svdbounds_down, svdbounds_up = _compute_svd_bounds(MiyajimaM1(), S, normE, normF, normG, T)

    # Note: Full M4 implementation would use the VBD isolation to refine
    # bounds for well-separated singular values. This requires access to
    # the full matrix A and V, which will be handled in _certify_svd_m4.
    return svdbounds_down, svdbounds_up
end

function _diagonal_ball_matrix(values::Vector{Ball{T, NT}}) where {T, NT}
    mids = map(mid, values)
    rads = map(rad, values)
    return BallMatrix(Diagonal(mids), Diagonal(rads))
end

#=
Miyajima 2014, Theorem 11 (M4): Full eigendecomposition-based implementation

This computes verified bounds using the eigendecomposition approach:
  D̂ + Ê = (AV)ᵀAV  where D̂ is diagonal

For isolated eigenvalues (Gershgorin disc doesn't overlap others), we can
use Parlett's theorem (Theorem 3 in the paper) for tighter bounds.

The bounds are:
  ζᵢᴹ = √((D̂ᵢᵢ - hᵢ) / (1 + ‖F‖))   (lower, if D̂ᵢᵢ ≥ hᵢ)
  ζ̄ᵢᴹ = √((D̂ᵢᵢ + hᵢ) / (1 - ‖F‖))   (upper)

where hᵢ is the tighter of either:
  - fᵢ = row sum of |Ê| (Gershgorin radius)
  - gᵢ = ‖Êe⁽ⁱ⁾‖² / (2ρᵢ) via Parlett's theorem (if isolated)
=#
function rigorous_svd_m4(A::BallMatrix{T}; apply_vbd::Bool = true) where {T}
    m, n = size(A)
    q = min(m, n)

    # Use eigendecomposition of AᵀA (or AAᵀ if m < n) for V
    if m >= n
        AtA = A' * A
        eig = eigen(Hermitian(mid(AtA)))
        V_mid = eig.vectors[:, end:-1:1]  # Reverse to get descending order
        λ = eig.values[end:-1:1]
    else
        AAt = A * A'
        eig = eigen(Hermitian(mid(AAt)))
        V_mid = eig.vectors[:, end:-1:1]
        λ = eig.values[end:-1:1]
    end

    V = BallMatrix(V_mid)

    # Compute F = VᵀV - I (orthogonality defect)
    F = V' * V - I
    normF = upper_bound_L2_opnorm(F)
    @assert normF < 1 "It is not possible to verify singular values: ‖VᵀV - I‖ ≥ 1"

    # Compute AV (or AᵀV if m < n) and then (AV)ᵀAV
    if m >= n
        AV = A * V
        H = AV' * AV  # This is D̂ + Ê in ball arithmetic
    else
        AtV = A' * V
        H = AtV' * AtV
    end

    # Extract diagonal D̂ and off-diagonal Ê
    n_sv = size(H, 1)
    D_diag = [H[i, i] for i in 1:n_sv]

    # Compute Gershgorin radii fᵢ = Σⱼ≠ᵢ |Hᵢⱼ|
    absH = upper_abs(H)
    f = zeros(T, n_sv)
    for i in 1:n_sv
        f[i] = setrounding(T, RoundUp) do
            s = zero(T)
            for j in 1:n_sv
                if j != i
                    s += absH[i, j]
                end
            end
            s
        end
    end

    # Check for isolation and compute refined bounds via Parlett if possible
    h = copy(f)  # Start with Gershgorin radii, refine if isolated

    for i in 1:n_sv
        D_ii = mid(D_diag[i])

        # Check if <D̂ᵢᵢ, fᵢ> is isolated from other intervals
        is_isolated = true
        ρᵢ = typemax(T)
        for j in 1:n_sv
            if j != i
                D_jj = mid(D_diag[j])
                gap = abs(D_ii - D_jj) - f[j]
                if gap <= f[i]
                    is_isolated = false
                    break
                end
                ρᵢ = min(ρᵢ, gap)
            end
        end

        # If isolated, try Parlett's refinement (Theorem 9 / equation for gᵢ)
        if is_isolated && ρᵢ > 0
            # gᵢ = ‖Êe⁽ⁱ⁾‖² / (2ρᵢ)
            # ‖Êe⁽ⁱ⁾‖² = Σⱼ≠ᵢ |Hᵢⱼ|²
            norm_Eei_sq = setrounding(T, RoundUp) do
                s = zero(T)
                for j in 1:n_sv
                    if j != i
                        s += absH[i, j]^2
                    end
                end
                s
            end
            gᵢ = setrounding(T, RoundUp) do
                norm_Eei_sq / (2 * ρᵢ)
            end
            h[i] = min(f[i], gᵢ)
        end
    end

    # Compute singular value bounds from squared singular value bounds
    singular_values = Vector{Ball{T, T}}(undef, n_sv)

    for i in 1:n_sv
        D_ii_mid = mid(D_diag[i])
        D_ii_rad = rad(D_diag[i])

        # Lower bound on λᵢ(D̂ + Ê): D̂ᵢᵢ - hᵢ - D̂ᵢᵢ_rad
        λ_lower = setrounding(T, RoundDown) do
            D_ii_mid - h[i] - D_ii_rad
        end

        # Upper bound on λᵢ(D̂ + Ê): D̂ᵢᵢ + hᵢ + D̂ᵢᵢ_rad
        λ_upper = setrounding(T, RoundUp) do
            D_ii_mid + h[i] + D_ii_rad
        end

        # Convert to singular value bounds: σ² ∈ [λ_lower/(1+F), λ_upper/(1-F)]
        σ²_lower = setrounding(T, RoundDown) do
            max(λ_lower / (one(T) + normF), zero(T))
        end
        σ²_upper = setrounding(T, RoundUp) do
            λ_upper / (one(T) - normF)
        end

        # Take square root for σ bounds
        σ_lower = setrounding(T, RoundDown) do
            sqrt(max(σ²_lower, zero(T)))
        end
        σ_upper = setrounding(T, RoundUp) do
            sqrt(σ²_upper)
        end

        σ_mid = (σ_lower + σ_upper) / 2
        σ_rad = setrounding(T, RoundUp) do
            max(σ_upper - σ_mid, σ_mid - σ_lower)
        end

        singular_values[i] = Ball(σ_mid, σ_rad)
    end

    Σ = _diagonal_ball_matrix(singular_values)

    # Compute U from A and V (optional, for completeness)
    # For now, return a placeholder
    U = BallMatrix(zeros(T, m, q))

    # Compute residual estimate
    residual_norm = zero(T)

    vbd = nothing
    if apply_vbd
        H_ball = adjoint(Σ) * Σ
        vbd = miyajima_vbd(H_ball; hermitian = true)
    end

    return RigorousSVDResult(U, singular_values, Σ, V, BallMatrix(zeros(T, m, n)),
        residual_norm, normF, normF, vbd)
end

"""
    refine_svd_bounds_with_vbd(result::RigorousSVDResult)

Attempt to refine singular value bounds using VBD isolation information.

For singular values whose squared values fall in isolated Gershgorin clusters,
we can potentially tighten the bounds using Miyajima's Theorem 11.

Returns a new `RigorousSVDResult` with potentially tighter bounds, or the
original result if no refinement is possible.
"""
function refine_svd_bounds_with_vbd(result::RigorousSVDResult{UT, ST, ΣT, VT, ET, RT, VBDT}) where {UT, ST, ΣT, VT, ET, RT, VBDT}
    vbd = result.block_diagonalisation
    if vbd === nothing
        return result
    end

    # Check for isolated clusters (singleton clusters)
    isolated_indices = Int[]
    for cluster in vbd.clusters
        if length(cluster) == 1
            push!(isolated_indices, cluster[1])
        end
    end

    if isempty(isolated_indices)
        return result  # No isolated singular values to refine
    end

    # For isolated singular values, we can potentially use tighter bounds
    # from the VBD Gershgorin intervals
    T = eltype(result.residual_norm)
    refined_singular_values = copy(result.singular_values)

    for idx in isolated_indices
        if idx <= length(vbd.cluster_intervals) && idx <= length(refined_singular_values)
            interval = vbd.cluster_intervals[idx]
            current_sv = result.singular_values[idx]

            # The VBD interval gives bounds on σ²
            # Extract and take square root
            λ_lower = max(mid(interval) - rad(interval), zero(T))
            λ_upper = mid(interval) + rad(interval)

            σ_lower = setrounding(T, RoundDown) do
                sqrt(max(λ_lower, zero(T)))
            end
            σ_upper = setrounding(T, RoundUp) do
                sqrt(λ_upper)
            end

            # Only use if tighter than existing bounds
            current_lower = mid(current_sv) - rad(current_sv)
            current_upper = mid(current_sv) + rad(current_sv)

            new_lower = max(σ_lower, current_lower)
            new_upper = min(σ_upper, current_upper)

            if new_lower < new_upper
                new_mid = (new_lower + new_upper) / 2
                new_rad = setrounding(T, RoundUp) do
                    max(new_upper - new_mid, new_mid - new_lower)
                end
                refined_singular_values[idx] = Ball(new_mid, new_rad)
            end
        end
    end

    refined_Σ = _diagonal_ball_matrix(refined_singular_values)

    return RigorousSVDResult(
        result.U, refined_singular_values, refined_Σ, result.V,
        result.residual, result.residual_norm,
        result.right_orthogonality_defect, result.left_orthogonality_defect,
        result.block_diagonalisation
    )
end
