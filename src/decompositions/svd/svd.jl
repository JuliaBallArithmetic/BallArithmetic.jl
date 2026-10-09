"""
    SVDMethod

Abstract type for selecting SVD certification algorithms.
"""
abstract type SVDMethod end

"""
    MiyajimaAuto <: SVDMethod

The default of [`rigorous_svd`](@ref): [`MiyajimaM3`](@ref), Theorem 10, for an exact input, and
[`MiyajimaM1`](@ref), Theorem 7, for one carrying a radius.

This is a selection rule of this package and not a result of Miyajima (2014); the measurements
behind it are in [`_svd_auto_theorem`](@ref). Pass `method` explicitly to override it.

On an input carrying a radius this is **not** what [`svdbox`](@ref) does. `svdbox` returns values
only, so it can certify `mid(A)` by Theorem 10 and widen each interval by `‖rad(A)‖₂`
([`_miyajima2014_thm10_weyl`](@ref)), which is tighter; `rigorous_svd` also returns
singular-vector bounds and the residuals `E`, `F`, `G` measured against the whole ball, which a
midpoint-only certification does not produce, so it stays on Theorem 7. For the values alone,
prefer `svdbox`.
"""
struct MiyajimaAuto <: SVDMethod end

"""
    MiyajimaM3 <: SVDMethod

Miyajima 2014, Theorem 10, the algorithm his numerical section labels M3, and what
[`rigorous_svd`](@ref) uses on an exact input.

The economy frames of Theorem 7 with a one-sided residual: with `Λᵢᵢ` and `Λ̄ᵢᵢ` the scalings of
`Σ̂ᵢᵢ` by `√((1∓‖Ĝ‖)/(1±‖F̂‖))` and `ρ = ‖AV̂ − ÛΣ̂‖₂/√(1−‖F̂‖₂)` for `m ≥ n`, with `F̂` and `Ĝ`
exchanged otherwise,

    Λᵢᵢ − ρ ≤ σᵢ(A) ≤ Λ̄ᵢᵢ + ρ.

Cheaper than Theorem 7, `22Qq² + 12q³` against `24Qq² + 12q³`, since the residual is `m × q`
rather than `m × n`, and at least as tight in the paper's Tables 2 and 3.
"""
struct MiyajimaM3 <: SVDMethod end

"""
    MiyajimaM1 <: SVDMethod

Miyajima 2014, Theorem 7, the algorithm his numerical section labels M1.

Bounds:
- Lower: σᵢ · √((1-‖F‖)(1-‖G‖)) - ‖E‖
- Upper: σᵢ · √((1+‖F‖)(1+‖G‖)) + ‖E‖

[`MiyajimaM3`](@ref), Theorem 10, uses the same economy frames and is cheaper and at least as
tight, so [`rigorous_svd`](@ref) prefers it on an exact input and keeps this one on a ball, where
Theorem 10's residual would propagate the input radius through a product.
"""
struct MiyajimaM1 <: SVDMethod end

"""
    MiyajimaM4 <: SVDMethod

Miyajima 2014, Theorem 11, the algorithm his numerical section labels M4.

Works on D̂ + Ê = (AV)ᵀAV where D̂ is diagonal. Uses Gershgorin isolation
and can provide very tight bounds for well-separated singular values.
"""
struct MiyajimaM4 <: SVDMethod end

"""
    RigorousSVDResult

Container returned by [`rigorous_svd`](@ref) bundling the midpoint
factorisation, the certified singular-value enclosures, and the
block-diagonal refinement obtained from [`schur_gershgorin_enclosure`](@ref).  Besides
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
enclosure can be refined by applying [`schur_gershgorin_enclosure`](@ref) to `Σ'Σ`,
yielding a block-diagonal structure with a rigorously bounded remainder.

The returned [`RigorousSVDResult`](@ref) exposes both the enclosures and
the intermediate norm bounds that justify them. When `apply_vbd` is set
to `false`, the `block_diagonalisation` field is `nothing`.

# References

* Miyajima S. (2014), "Verified bounds for all the singular values of matrix",
  Japan J. Indust. Appl. Math. 31, 513–539.
* Rump S.M. (2011), "Verified bounds for singular values", BIT 51, 367–384.
"""
function rigorous_svd(A::BallMatrix{T}; method::SVDMethod = MiyajimaAuto(), apply_vbd::Bool = true) where {T}
    RT = real(T)
    method = _resolve_svd_method(method, A)

    # For BigFloat, use GenericLinearAlgebra's native BigFloat SVD
    if RT === BigFloat
        return _rigorous_svd_bigfloat(A, method; apply_vbd)
    end

    # MiyajimaM4 is Theorem 11, which needs the (AV)'AV frame rather than a two-frame SVD
    method isa MiyajimaM4 && return rigorous_svd_m4(A; apply_vbd)

    # Standard path for Float64/Float32
    svdA = svd(A.c)
    return _certify_svd(A, svdA, method; apply_vbd)
end

"""
    rigorous_svd_gpu(A::BallMatrix{Float64,Float64}; method = MiyajimaM1(),
                     apply_vbd = true, seed_on = :cpu)

GPU-accelerated rigorous SVD of a (CPU-resident) `Float64` ball matrix,
returning the same [`RigorousSVDResult`](@ref) as [`rigorous_svd`](@ref).

The certification math is identical to [`rigorous_svd`](@ref) — only the
`O(n³)` certification products (`UΣVᵀ`, `VᵀV`, `UᵀU`, and the residual
widening term) are evaluated on the GPU through the rigorous INT8-Ozaki
`MMul4` dispatch, while the `O(n²)` rigorous norm bounds and singular-value
enclosures are finished on the CPU with directed rounding. The seed
factorisation defaults to CPU LAPACK (`seed_on = :cpu`, the *hybrid*
configuration, which benchmarks fastest on consumer/A40 cards because their
`Float64` throughput is throttled); pass `seed_on = :gpu` to seed with
cuSOLVER instead.

This method is only available when the `CUDA` extension is loaded
(`using CUDA`) and a functional GPU is present. The enclosures it returns
overlap those of [`rigorous_svd`](@ref); the GPU midpoint truncation makes
the singular-value radii modestly looser (still rigorous, typically `~10⁻¹⁰`).
"""
function rigorous_svd_gpu end

"""
    _rigorous_svd_bigfloat(A, method; apply_vbd, use_cache)

BigFloat version of rigorous SVD using GenericLinearAlgebra's native BigFloat SVD.
Computes SVD directly at BigFloat precision via `svd_bigfloat`, then certifies.
This is 2.5–6.6× faster than the previous Ogita refinement path and achieves
~10⁻⁷⁴ residuals (vs ~10⁻¹⁴ for Float64→BigFloat refinement).

When `use_cache=true` (default), the computed SVD is stored for potential reuse.
The Ogita refinement path (`ogita_svd_refine`, `adaptive_ogita_svd`) remains
available for explicit use.
"""
function _rigorous_svd_bigfloat(A::BallMatrix{BigFloat}, method::SVDMethod;
                                 apply_vbd::Bool = true, use_cache::Bool = true)
    # Check cache first
    m, n_cols = size(A)
    A_hash = hash(A.c)
    cache_valid = use_cache &&
                  _svd_cache_U[] !== nothing &&
                  size(_svd_cache_U[], 1) == m &&
                  size(_svd_cache_V[], 1) == n_cols &&
                  _svd_cache_A_hash[] == A_hash

    if cache_valid
        _svd_cache_hits[] += 1
        svd_result = SVD(Matrix(_svd_cache_U[]), Vector(_svd_cache_S[]),
                         Matrix(_svd_cache_V[]'))
    else
        _svd_cache_misses[] += 1
        # Use GenericLinearAlgebra's native BigFloat SVD
        svd_result = svd_bigfloat(A.c)
    end

    # Update cache with GLA result for future use
    if use_cache
        set_svd_cache!(svd_result.U, svd_result.S, svd_result.V, A_hash)
    end

    return _certify_svd(A, svd_result, method; apply_vbd)
end

"""
    svdbox(A::BallMatrix; method = :auto) -> Vector{Ball}

Verified enclosures of all the singular values of `A`, one ball each, in the order the chosen
theorem produces them.

`svdbox` is the name Miyajima (2014) gives his Theorem 7. This function
selects an enclosure and does nothing else; each enclosure is a separate unexported function
named for the theorem it implements, so a deviation from a paper is visible in the name:

| `method` | routine | what it is |
|---|---|---|
| `:auto` | [`_svd_auto_theorem`](@ref) | **the default**: Theorem 10 for an exact input, Theorem 10 plus the Weyl widening for one with a radius. Not a theorem but a selection rule, with the measurements behind it in its docstring |
| `:miyajima2014_thm10` | [`_miyajima2014_thm10`](@ref) | what `:auto` uses on an exact input: the same economy frames as Theorem 7 with the one-sided residual `‖AV̂ − ÛΣ̂‖₂`, which the paper's tables make at least as tight and which costs `22Qq²` against `24Qq²` |
| `:miyajima2014_thm10_weyl` | [`_miyajima2014_thm10_weyl`](@ref) | Theorem 10 on `mid(A)` alone, each interval widened by `‖rad(A)‖₂`. A variation, not the paper: it keeps Theorem 10 on an exact matrix and pays the input radius once instead of through a ball product |
| `:miyajima2014_thm7` | [`_miyajima2014_thm7`](@ref) | the economy enclosure `Σ̂ᵢᵢ√((1∓‖F̂‖)(1∓‖Ĝ‖)) ∓ ‖Ê‖`, residual measured against `A` |
| `:miyajima2014_thm4` | [`_miyajima2014_thm4`](@ref) | Oishi's, square frames, `Σᵢᵢ ± (Σᵢᵢmax(‖F‖,‖G‖) + ‖E‖)`; Theorem 8 proves Theorem 7 is never worse |
| `:miyajima2014_thm11` | [`_miyajima2014_thm11`](@ref) | one frame, the eigenvectors of `AᵀA`, Gershgorin sharpened by Parlett, with a cluster branch |
| `:rump2011_thm3_1` | [`_rump2011_thm3_1`](@ref) | Rump's two-frame split `UᴴAV = D + E`; square `A` only, and valid only up to an undetermined numbering |

These are the paper's own algorithms M3, M1, O and R1 in its numerical section, which labels
M1 = Theorem 7, M2 = Theorem 9, M3 = Theorem 10, M4 = Theorem 11, O = Theorem 4, R1 = Theorem 5
and R2 = Theorem 6. Theorem 9, the sharpening of Rump's Theorem 5, has no economy form: Remark 5
shows the upper half of it fails on an economy frame and leaves the improvement as an open
challenge.

Prefer the default. Both enclosures it chooses between are anchored to `Σ̂ᵢᵢ` of the SVD, so the
`i`-th interval encloses `σᵢ(A)` with no permutation caveat, unlike `:rump2011_thm3_1`, and
neither degrades with conditioning, unlike `:miyajima2014_thm11`.

For the singular vectors, the residual and the block refinement as well as the values, use
[`rigorous_svd`](@ref).

# Reference

S. Miyajima, *Verified bounds for all the singular values of matrix*, Japan J. Indust. Appl.
Math. **31** (2014) 513-539, doi 10.1007/s13160-014-0145-5.
"""
function svdbox(A::BallMatrix{T}; method::Symbol = :auto) where {T}
    method === :auto && (method = _svd_auto_theorem(A))
    method === :miyajima2014_thm10 && return _miyajima2014_thm10(A)
    method === :miyajima2014_thm10_weyl && return _miyajima2014_thm10_weyl(A)
    method === :miyajima2014_thm7 && return _miyajima2014_thm7(A)
    method === :miyajima2014_thm4 && return _miyajima2014_thm4(A)
    method === :miyajima2014_thm11 && return _miyajima2014_thm11(A)
    method === :rump2011_thm3_1 && return _rump2011_thm3_1(A)
    throw(ArgumentError("svdbox: unknown method $(repr(method)); the implemented methods are " *
                        ":auto, :miyajima2014_thm10, :miyajima2014_thm10_weyl, " *
                        ":miyajima2014_thm7, :miyajima2014_thm4, :miyajima2014_thm11 and " *
                        ":rump2011_thm3_1"))
end

"""
    svd_bounds(A::BallMatrix; method = :auto) -> (lo, hi)

Certified bounds `lo[i] ≤ σᵢ(X) ≤ hi[i]` for every `X` in the ball `A` and every
`i = 1:min(size(A)...)`, singular values in decreasing order, read off [`svdbox`](@ref) with the
given `method` and rounded outward. Where the enclosure fails, or the input carries a non-finite
entry, the bounds are the trivial `lo = 0`, `hi = Inf`, so the result is always a valid bound.

A caller whose numbers must not change with a change of default names the theorem, for instance
`method = :miyajima2014_thm10_weyl`, rather than relying on `:auto`.
"""
function svd_bounds(A::BallMatrix{T}; method::Symbol = :auto) where {T}
    q = minimum(size(A))
    RT = real(T)
    q == 0 && return (RT[], RT[])
    all(isfinite, mid(A)) && all(isfinite, rad(A)) || return (zeros(RT, q), fill(RT(Inf), q))
    s = svdbox(A; method)
    lo = RT[max(zero(RT), sub_down(RT(mid(x)), RT(rad(x)))) for x in s]
    hi = RT[add_up(RT(mid(x)), RT(rad(x))) for x in s]
    for i in 1:q
        (isfinite(lo[i]) && isfinite(hi[i])) || (lo[i] = zero(RT); hi[i] = RT(Inf))
    end
    return lo, hi
end
svd_bounds(A::AbstractMatrix; kwargs...) = svd_bounds(BallMatrix(Matrix(A)); kwargs...)

"""
    svd_lower_bound_sigma_min(A::BallMatrix; method = :auto) -> T

Certified lower bound on the smallest singular value `σ_q`, `q = min(size(A)...)`, of every matrix
in the ball, from [`svd_bounds`](@ref); zero when the enclosure fails.
"""
function svd_lower_bound_sigma_min(A; method::Symbol = :auto)
    lo, _ = svd_bounds(A; method)
    return isempty(lo) ? zero(eltype(lo)) : lo[end]
end

# MiyajimaAuto is resolved before anything downstream sees it, so no routine ever receives a
# method that is a rule rather than a theorem.
#
# The rule is spelled here rather than read off `_svd_auto_theorem`, which the two entry points used
# to share. They no longer agree on an input carrying a radius: `svdbox` routes it to
# `_miyajima2014_thm10_weyl`, which certifies `mid(A)` and widens by `‖rad(A)‖₂`, and
# `rigorous_svd` also returns singular-vector bounds and the residuals `E`, `F`, `G` measured
# against the whole ball, which a midpoint-only certification does not produce. So `rigorous_svd`
# keeps Theorem 7 there, and its values are the wider of the two by the factors in
# `_svd_auto_theorem`'s table.
_resolve_svd_method(method::SVDMethod, ::BallMatrix) = method
function _resolve_svd_method(::MiyajimaAuto, A::BallMatrix)
    return iszero(rad(A)) ? MiyajimaM3() : MiyajimaM1()
end

function _certify_svd(A::BallMatrix{T}, svdA::SVD, method::SVDMethod; apply_vbd::Bool = true) where {T}
    return _certify_svd_impl(A, svdA.U, svdA.S, svdA.V, svdA.Vt, method; apply_vbd)
end

# Method for NamedTuple (used by Ogita refinement)
function _certify_svd(A::BallMatrix{T}, svdA::NamedTuple{(:U, :S, :V, :Vt)}, method::SVDMethod; apply_vbd::Bool = true) where {T}
    return _certify_svd_impl(A, svdA.U, svdA.S, svdA.V, svdA.Vt, method; apply_vbd)
end

function _certify_svd_impl(A::BallMatrix{T}, U_in, S_in, V_in, Vt_in, method::SVDMethod; apply_vbd::Bool = true) where {T}
    method = _resolve_svd_method(method, A)
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

    # Verification requires ‖F‖ < 1 and ‖G‖ < 1 for the Neumann series bounds
    # Return failed result instead of crashing if conditions aren't met
    if normF >= 1 || normG >= 1
        n_sv = length(S_in)
        RT = real(T)
        # Create failed result with infinite bounds
        singular_values = [Ball(S_in[i], RT(Inf)) for i in 1:n_sv]
        Σ = _diagonal_ball_matrix(singular_values)
        residual = E
        @warn "SVD verification failed: ‖V'V - I‖ = $normF, ‖U'U - I‖ = $normG (both must be < 1)"
        return RigorousSVDResult(U, singular_values, Σ, V, residual,
            RT(Inf), normF, normG, nothing)
    end

    # Compute bounds based on method. Theorem 10 needs the one-sided residual as well, which the
    # generic `_compute_svd_bounds` signature does not carry, so it is formed here from the frames
    # already at hand and handed to the same formula `_miyajima2014_thm10` uses.
    local svdbounds_down, svdbounds_up
    if method isa MiyajimaM3
        m_, n_ = size(A)
        q_ = length(S_in)
        Uq = BallMatrix(U_in[:, 1:q_])
        Vq = BallMatrix(V_in[:, 1:q_])
        Sq = BallMatrix(Diagonal(S_in[1:q_]))
        res1 = m_ >= n_ ? A * Vq - Uq * Sq : Uq' * A - Sq * Vq'
        svdbounds_down, svdbounds_up = _miyajima2014_thm10_bounds(S_in[1:q_],
            upper_bound_L2_opnorm(res1), normF, normG, m_ >= n_, T)
    else
        svdbounds_down, svdbounds_up = _compute_svd_bounds(method, S_in, normE, normF, normG, T)
    end

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
            vbd = schur_gershgorin_enclosure(H; hermitian = true)
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
Miyajima 2014, Theorem 11 (M4): Eigendecomposition-based bounds

This method works on D̂ + Ê = (AV)ᵀAV where D̂ is diagonal.
For well-separated singular values, it can give tighter bounds through
Gershgorin isolation.

For now, this falls back to M1 bounds but with the note that the VBD
result can be used for further refinement of isolated singular values.
=#
# MiyajimaM4 never reaches this function: `rigorous_svd` routes it to `rigorous_svd_m4`, which
# has the (AV)'AV quantities Theorem 11 needs and which this signature does not carry. It used to
# return MiyajimaM1's bounds behind a warning, so a caller asking for Theorem 11 was given
# Theorem 7 instead; `sylvester_resolvent_bound.jl` asks for it by name.
function _compute_svd_bounds(::MiyajimaM4, S::Vector, normE, normF, normG, ::Type{T}) where {T}
    throw(ArgumentError("MiyajimaM4 is Theorem 11 of Miyajima (2014) and needs the frame " *
                        "(AV)'AV, which this path does not form; call `rigorous_svd_m4(A)`, " *
                        "`rigorous_svd(A; method = MiyajimaM4())` or " *
                        "`svdbox(A; method = :miyajima2014_thm11)`"))
end

function _diagonal_ball_matrix(values::Vector{Ball{T, NT}}) where {T, NT}
    mids = map(mid, values)
    rads = map(rad, values)
    return BallMatrix(Diagonal(mids), Diagonal(rads))
end

#=
Miyajima 2014, Theorem 11 (M4): Eigendecomposition-based singular value verification

WHAT M4 VERIFIES:
  - Rigorous bounds on ALL singular values σᵢ(A)

WHAT M4 DOES NOT VERIFY:
  - Left singular vectors U (returned as placeholder zeros)
  - Right singular vectors V (approximate, used only for verification)
  - SVD residual ‖A - UΣVᵀ‖ (returned as Inf)

This is BY DESIGN per Miyajima's method. The singular value bounds come from
Gershgorin/Parlett analysis on (AV)ᵀAV, requiring only the orthogonality
defect F = VᵀV - I. No U computation or residual bound is needed or provided.

THEORY:
  D̂ + Ê = (AV)ᵀAV  where D̂ is diagonal, V from eigendecomposition of AᵀA

For isolated eigenvalues (Gershgorin disc doesn't overlap others), we can
use Parlett's theorem (Theorem 3 in the paper) for tighter bounds.

The bounds are:
  ζᵢᴹ = √((D̂ᵢᵢ - hᵢ) / (1 + ‖F‖))   (lower, if D̂ᵢᵢ ≥ hᵢ)
  ζ̄ᵢᴹ = √((D̂ᵢᵢ + hᵢ) / (1 - ‖F‖))   (upper)

where hᵢ is the tighter of either:
  - fᵢ = row sum of |Ê| (Gershgorin radius)
  - gᵢ = ‖Êe⁽ⁱ⁾‖² / (2ρᵢ) via Parlett's theorem (if isolated)

Reference: Miyajima, S. "Verified bounds for all the singular values of matrix"
           Japan J. Indust. Appl. Math. (2014) 31:513–539
=#
function rigorous_svd_m4(A::BallMatrix{T}; apply_vbd::Bool = true) where {T}
    m, n = size(A)
    q = min(m, n)
    # Theorem 11 lives in `_miyajima2014_thm11`; this routine only packages it as a
    # RigorousSVDResult. It used to carry a second copy of the theorem which divided Parlett's
    # bound by 2*rho_i instead of rho_i, and which had no cluster branch, so a clustered index
    # was given its own Gershgorin interval, which Theorem 11 does not license.
    singular_values = _miyajima2014_thm11(A)
    Σ = _diagonal_ball_matrix(singular_values)

    # Theorem 11 uses one frame and says nothing about U, V or the residual: the frame is the
    # eigenvectors of A'A, and only ||V'V - I|| enters. U is a placeholder and the residual Inf.
    W = m >= n ? A' * A : A * A'
    Vm = try
        ev = eigen(Hermitian(mid(W)))
        ev.vectors[:, end:-1:1]
    catch
        Matrix{T}(I, size(W, 1), size(W, 1))
    end
    V = BallMatrix(Vm)
    normF = upper_bound_L2_opnorm(V' * V - I)

    vbd = nothing
    if apply_vbd && T !== BigFloat && all(isfinite, rad(Σ))
        vbd = schur_gershgorin_enclosure(adjoint(Σ) * Σ; hermitian = true)
    end

    return RigorousSVDResult(BallMatrix(zeros(T, m, q)), singular_values, Σ, V,
        BallMatrix(zeros(T, m, n)), T(Inf), normF, T(Inf), vbd)
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
    # from the VBD Gershgorin intervals (σ² enclosures).
    T = eltype(result.residual_norm)
    refined_singular_values = copy(result.singular_values)

    for idx in isolated_indices
        idx <= length(vbd.cluster_intervals) || continue
        interval = vbd.cluster_intervals[idx]

        # The VBD interval gives bounds on σ²; take the square root for σ.
        λ_lower = max(real(mid(interval)) - rad(interval), zero(T))
        λ_upper = real(mid(interval)) + rad(interval)
        σ_lower = setrounding(T, RoundDown) do
            sqrt(max(λ_lower, zero(T)))
        end
        σ_upper = setrounding(T, RoundUp) do
            sqrt(max(λ_upper, zero(T)))
        end

        # Match by VALUE, not index: the VBD discs are in the (permuted) basis order,
        # while `singular_values` is in the original order.  The σ-interval and the
        # matching singular value's current enclosure both contain the same true σ, so
        # they overlap; for an *isolated* cluster exactly one singular value matches.
        candidates = Int[]
        for j in eachindex(refined_singular_values)
            sv = refined_singular_values[j]
            cl = mid(sv) - rad(sv)
            cu = mid(sv) + rad(sv)
            (σ_lower <= cu && cl <= σ_upper) && push!(candidates, j)
        end
        length(candidates) == 1 || continue   # ambiguous or no match — skip (stay rigorous)
        j = candidates[1]

        current_sv = refined_singular_values[j]
        new_lower = max(σ_lower, mid(current_sv) - rad(current_sv))
        new_upper = min(σ_upper, mid(current_sv) + rad(current_sv))
        if new_lower < new_upper
            new_mid = (new_lower + new_upper) / 2
            new_rad = setrounding(T, RoundUp) do
                max(new_upper - new_mid, new_mid - new_lower)
            end
            refined_singular_values[j] = Ball(new_mid, new_rad)
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
