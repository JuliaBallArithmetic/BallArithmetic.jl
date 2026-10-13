# An accurate ball matrix product, with the midpoint accumulated by error-free transformations and
# the radius bounded by
#
#   T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product",
#   SIAM J. Sci. Comput. 26(6):1955-1988, 2005, doi 10.1137/030601818.
#
# Why this exists beside MMul3, MMul4 and MMul5. All three of those obtain the midpoint from an
# ordinary BLAS product and then charge its rounding to the radius: MMul4 brackets `mA*mB` by
# evaluating it twice under `RoundUp` and `RoundDown`, MMul3 and MMul5 add the Revol-Theveny
# `(k+c)*eps` term. Either way the radius carries the accumulation error over the k summands, of
# order `gamma_k |mA| |mB|` with `gamma_k = k u / (1 - k u)`. Here the accumulation is compensated
# instead, so what is charged is the Ogita-Rump-Oishi bound
#
#   |computed - exact|  <=  u |exact| + gamma_N^2 sum_i |a_i| |b_i|,
#
# with N the number of products summed and `gamma_N^2` in place of `gamma_N`: at k = 5000 and
# Float64 that is 1.2e-28 against 1.1e-12, sixteen orders.
#
# Two further properties, both from `error_free_transformations.jl`:
#
#   * nothing here changes the rounding mode for the midpoint. `two_product` needs one FMA and
#     `two_sum` is Knuth's six additions, all in round-to-nearest. Only the radius, which is a
#     bound and not a value, is accumulated with `RoundUp`. That matters because the directed
#     rounding route depends on the BLAS honouring `setrounding`, which threaded and GPU
#     implementations need not do, and which is why `OpenBLASConsistentFPCSR_jll` is a dependency.
#   * the interval part of the radius is unchanged and is the same as MMul4's: for X in A and Y in B,
#     `XY - mA mB = dA mB + mA dB + dA dB`, so `rA (|mB| + rB) + |mA| rB` entrywise.
#
# The cost, measured on 16 threads against the ball `*` on complex k x k matrices, radius zero:
#
#   k      ball-*      this      slowdown    radius ratio
#   200    0.00733 s   0.00400 s   0.55         0.0075
#   400    0.04863 s   0.02012 s   0.41         0.0044
#   800    0.11550 s   0.15310 s   1.32         0.0037
#   1600   0.37610 s   1.35100 s   3.59         0.0029
#
# So it is FASTER than the ball product up to about k = 400 and a few times slower at 1600, for a
# radius two to three orders tighter, the advantage growing with k as gamma_k^2 against gamma_k. The
# midpoint is O(m n k) scalar operations with no BLAS, some ten flops per summand, threaded over the
# output columns; the complex ball product it is compared against itself issues several GEMMs for the
# real and imaginary parts plus the absolute-value products, which is why the crossover is as late as
# it is. `*` dispatches to MMul4, see the note below.
#
# The Ozaki scheme, which recovers full BLAS speed by splitting until the accumulations are
# themselves exact, would remove even the k = 1600 penalty and is not implemented.

# The default product stays `MMul4` (Isaia, 2026-10-12): `*` calls it and the code of the package
# uses `*` only. `MMul4` evaluates BLAS products under `RoundUp` and `RoundDown`, on the OpenBLAS
# built with CONSISTENT_FPCSR that the module loads (see `src/BallArithmetic.jl`). This routine
# and `_product_down_up_dot2err` below are alternatives that are not called by `*`.

export mmul_ogita_rump_oishi_2005

# |a| for a real or complex matrix, structure preserved, rounded up where the modulus is inexact.
# The complex modulus goes through `abs_up`: `abs` is `hypot`, which does not honour the rounding
# mode, and under RoundUp it fell below the exact modulus for 77915 of 10^6 random arguments.
_modulus_up(M::AbstractMatrix{<:Real}) = abs_preserving_structure(M)
_modulus_up(M::AbstractMatrix{<:Complex}) = abs_up.(M)

# The accuracy term for one block of products.
#
#   T. Ogita, S. M. Rump and S. Oishi, Accurate sum and dot product, SIAM J. Sci. Comput. 26(6)
#   (2005) 1955-1988, doi 10.1137/030601818, Proposition 5.5: for x_i, y_i floating point,
#   1 ≤ i ≤ n, n eps < 1, and res the result of Algorithm 5.3 (Dot2), if no underflow occurs
#
#       |res − xᵀy| ≤ eps |xᵀy| + γ_n² |xᵀ||y|,
#
#   and, in the presence of underflow (5.6),
#
#       |res − xᵀy| ≤ eps |xᵀy| + γ_n² |xᵀ||y| + 5 n eta.
#
# There eps is the relative rounding error unit (2^-53 in double precision, `unit_roundoff` here,
# written u below), eta the underflow unit (2^-1074, `subnormal_min`), γ_n = n eps/(1 − n eps).
# Theorem 3.4 of the paper gives the 5 eta per product for TwoProduct by Dekker's splitting
# (Algorithm 3.3); `two_product` here is the paper's Algorithm 3.5 (TwoProductFMA), whose error
# under underflow is one rounding of the exact residual, at most eta/2 per product, so the 5 n eta
# covers it (the paper does not state the FMA case). Two further steps are ours:
#   * the bound is in terms of the exact |xᵀy|; with only the computed c at hand, |xᵀy| ≤ |c| + err
#     gives err ≤ (u|c| + t)/(1 − u), t the remaining terms;
#   * the paper is for real vectors. For a complex entry the real and imaginary parts are each such
#     a dot product of N = 2k terms with Σ|x_i||y_i| at most `absprod`, so the modulus of the error
#     is at most u|xᵀy| + √2 t, and 3/2 ≥ √2 is used.
# `absprod` must already bound Σ_i |a_i||b_i| from above.
function _accuracy_term(absC::AbstractMatrix{T}, absprod::AbstractMatrix{T}, N::Integer,
        ::Type{T}, iscomplex::Bool) where {T <: AbstractFloat}
    u = unit_roundoff(T)
    g = gamma_bound(N, T)
    g2 = mul_up(g, g)
    under = mul_up(T(5 * N), subnormal_min(T))
    κ = iscomplex ? T(3) / T(2) : one(T)
    scale = div_up(one(T), sub_down(one(T), u))
    return mul_up.(scale,
        add_up.(mul_up.(u, absC), mul_up.(κ, add_up.(mul_up.(g2, absprod), under))))
end

"""
    mmul_ogita_rump_oishi_2005(A::BallMatrix, B::BallMatrix) -> BallMatrix

Product of two ball matrices whose midpoint is accumulated with error-free transformations and
whose radius charges the Ogita-Rump-Oishi (2005) accuracy bound rather than the rounding of a BLAS
product.

For every `X ∈ A` and `Y ∈ B` the returned ball contains `XY`. The radius is the sum of

* the interval part, `rad(A)(|mid(B)| + rad(B)) + |mid(A)| rad(B)` entrywise, which is what any
  midpoint-radius product must carry and is identical to [`MMul4`](@ref)'s; and
* the accuracy part, `(u|C| + κ(γ_N² Σᵢ|aᵢ||bᵢ| + 5N·eta))/(1 − u)` with `N` the number of products
  summed, `γ_N = Nu/(1−Nu)`, `eta` the underflow unit and `κ = 1` for real data, `3/2` for complex:
  Proposition 5.5 of the reference with its underflow term, see `_accuracy_term`.

The `γ_N²` is the point: `MMul3`, `MMul4` and `MMul5` all charge the accumulation over the `k`
summands at order `γ_k`, and at `k = 5000` in `Float64` that is `1.1e-12` against this routine's
`1.2e-28`.

The midpoint never changes the rounding mode; `two_product` uses one FMA and `two_sum` is Knuth's
six additions, both in round-to-nearest. Only the radius is accumulated under `RoundUp`.

**Cost.** `O(mnk)` scalar operations with no BLAS, threaded over the output columns. Measured on 16
threads against the ball `*` on complex matrices with radius zero, it is *faster* below about
`k = 400` — `0.55×` at `k = 200`, `0.41×` at `400` — and `1.32×` at `800`, `3.59×` at `1600`, with the
radius ratio improving from `0.0075` to `0.0029` over the same range. `*` dispatches to
[`MMul4`](@ref), which is the default product of the package.

# Reference

T. Ogita, S. M. Rump and S. Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. **26**(6)
(2005) 1955-1988, doi 10.1137/030601818.
"""
function mmul_ogita_rump_oishi_2005(A::BallMatrix{T, S}, B::BallMatrix{T, S}) where {T, S}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    m, k = size(mA)
    k2, n = size(mB)
    k == k2 ||
        throw(DimensionMismatch("mmul_ogita_rump_oishi_2005: size(A, 2) = $k but size(B, 1) = $k2"))

    mC = Matrix{S}(undef, m, n)
    _accurate_midpoint!(mC, mA, mB)

    # the accuracy term needs an upper bound on sum_i |a_i| |b_i|, one GEMM on the moduli
    absA, absB = _modulus_up(mA), _modulus_up(mB)
    absprod = setrounding(T, RoundUp) do
        absA * absB
    end::Matrix{T}
    # N summands per entry: k for a real product, 2k for a complex one, whose real and imaginary
    # parts are each a difference or sum of two real dot products of length k
    N = S <: Complex ? 2k : k
    acc = _accuracy_term(_modulus_up(mC), absprod, N, T, S <: Complex)

    # the interval part, as in MMul4; the assertion gives the compiler the type the closure hides
    rC = _sum_product_up(_add_up(acc, _product_up(absA, rB)), rA, _add_up(absB, rB))::Matrix{T}
    return BallMatrix(mC, rC)
end

# real midpoint: one compensated dot product per entry
function _accurate_midpoint!(mC::Matrix{S}, mA::AbstractMatrix{S},
        mB::AbstractMatrix{S}) where {S <: AbstractFloat}
    m, n = size(mC)
    Base.Threads.@threads for j in 1:n
        for i in 1:m
            @inbounds mC[i, j] = compensated_terms(((mA, mB, one(S)),), i, j)
        end
    end
    return mC
end

# complex midpoint: (Ar + i Ai)(Br + i Bi) = (Ar Br - Ai Bi) + i (Ar Bi + Ai Br), each part a
# compensated accumulation over 2k products, which is exactly what `compensated_terms`'s signed
# `pairs` argument is for
function _accurate_midpoint!(mC::Matrix{S}, mA::AbstractMatrix{S},
        mB::AbstractMatrix{S}) where {S <: Complex}
    T = real(S)
    Ar, Ai = real.(mA), imag.(mA)
    Br, Bi = real.(mB), imag.(mB)
    re_pairs = ((Ar, Br, one(T)), (Ai, Bi, -one(T)))
    im_pairs = ((Ar, Bi, one(T)), (Ai, Br, one(T)))
    m, n = size(mC)
    Base.Threads.@threads for j in 1:n
        for i in 1:m
            @inbounds mC[i, j] = complex(compensated_terms(re_pairs, i, j),
                compensated_terms(im_pairs, i, j))
        end
    end
    return mC
end

# ---------------------------------------------------------------------------------------------
# The product of two floating-point matrices bounded below and above with no rounding mode, by
#
#   T. Ogita, S. M. Rump and S. Oishi, Accurate sum and dot product, SIAM J. Sci. Comput. 26(6)
#   (2005) 1955-1988, doi 10.1137/030601818, Algorithm 5.8 (Dot2Err) and Corollary 5.9.
#
# Algorithm 5.8, for x, y of n floating-point numbers, eps the relative rounding error unit and
# eta the underflow unit, every operation in rounding to nearest:
#
#       if 2n eps ≥ 1, error
#       [p, s] = TwoProduct(x₁, y₁);  e = |s|
#       for i = 2 : n
#           [h, r] = TwoProduct(xᵢ, yᵢ);  [p, q] = TwoSum(p, h)
#           t = fl(q + r);  s = fl(s + t);  e = fl(e + |t|)
#       res = fl(p + s)
#       δ = fl(n eps/(1 − 2n eps));  α = fl(eps|res| + (δe + 3eta/eps));  err = fl(α/(1 − 2eps))
#
# Corollary 5.9: if the algorithm runs to completion then, also in the presence of underflow,
# res − err ≤ xᵀy ≤ res + err.
#
# The loop below starts from p = s = e = 0 and treats the first product as the others; TwoSum(0, h)
# is (h, 0), so the values of p, s and e are those of the algorithm. `two_product` is the paper's
# Algorithm 3.5 (TwoProductFMA) where the corollary is stated for Algorithm 3.3; with underflow its
# two outputs differ from the product by one rounding of the exact residual, at most eta/2, where
# Theorem 3.4 allows 5 eta (the paper does not state the FMA case).

# p, s and e of Algorithm 5.8 for row i of At' and column j of B, At the transpose of the left
# factor so that both are read down a column.
@inline function _dot2err_accumulate(At::AbstractMatrix{T}, B::AbstractMatrix{T}, i::Integer,
        j::Integer) where {T <: AbstractFloat}
    p = zero(T)
    s = zero(T)
    e = zero(T)
    @inbounds for k in axes(B, 1)
        h, r = two_product(At[k, i], B[k, j])
        p, q = two_sum(p, h)
        t = q + r
        s += t
        e += abs(t)
    end
    return p, s, e
end

# δ and 3eta/eps of Algorithm 5.8. n eps and 3eta/eps are exact; the two operations of δ are
# rounded to nearest as the algorithm prescribes.
function _dot2err_constants(n::Integer, ::Type{T}) where {T <: Union{Float32, Float64}}
    u = unit_roundoff(T)
    nu = T(n) * u
    2 * nu < 1 || error("_dot2err: 2n eps = $(2 * nu) ≥ 1 for n = $n, inclusion failed")
    return nu / (1 - 2 * nu), 3 * subnormal_min(T) / u
end

"""
    _dot2err_ogita_rump_oishi_2005(x, y) -> (res, err)

Algorithm 5.8 (Dot2Err) of Ogita, Rump and Oishi (2005) for two vectors of `Float32` or `Float64`:
`res` is the dot product accumulated in twice the working precision and rounded, and
`res − err ≤ xᵀy ≤ res + err` (Corollary 5.9), also in the presence of underflow. Every operation
is in rounding to nearest. Where an intermediate result is not finite `err` is `Inf`.

# Reference

T. Ogita, S. M. Rump and S. Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. **26**(6)
(2005) 1955-1988, doi 10.1137/030601818.
"""
function _dot2err_ogita_rump_oishi_2005(x::AbstractVector{T},
        y::AbstractVector{T}) where {T <: Union{Float32, Float64}}
    length(x) == length(y) ||
        throw(DimensionMismatch("_dot2err: lengths $(length(x)) and $(length(y))"))
    δ, c = _dot2err_constants(length(x), T)
    p, s, e = _dot2err_accumulate(reshape(x, :, 1), reshape(y, :, 1), 1, 1)
    u = unit_roundoff(T)
    res = p + s
    α = u * abs(res) + (δ * e + c)
    err = α / (1 - 2 * u)
    return res, (isfinite(res) && isfinite(err)) ? err : T(Inf)
end

# Lower and upper bound of xᵀy from p, s and e of Algorithm 5.8. The proof of Corollary 5.9 bounds
# the distance of xᵀy from the unevaluated p + s, underflow included, by the Δ of (5.9) less the
# one eta charged there to fl(eps|res|), and shows Δ ≤ fl(δe + 3eta/eps) =: β. The corollary then
# rounds p + s to nearest and charges eps|res| for it; here p + (s ± β) is instead rounded
# downward and upward by the directed operations, which is our step and not the paper's.
@inline function _dot2err_down_up(p::T, s::T, e::T, δ::T, c::T) where {T <: AbstractFloat}
    β = δ * e + c
    (isfinite(p) && isfinite(s) && isfinite(β)) || return T(-Inf), T(Inf)
    return add_down(p, sub_down(s, β)), add_up(p, add_up(s, β))
end

"""
    _product_down_up_dot2err(A, B) -> (down, up)

Entrywise lower and upper bounds of the product of two `Float32` or `Float64` matrices,
`down ≤ AB ≤ up`, with no change of the rounding mode: each entry is accumulated by Algorithm 5.8
(Dot2Err) of Ogita, Rump and Oishi (2005) as an unevaluated sum `p + s` with the bound `β` of the
proof of Corollary 5.9 on its distance from the exact entry, underflow included, and
`p + (s − β)`, `p + (s + β)` are rounded downward and upward by `add_down`, `sub_down`, `add_up`.
An entry with a non-finite intermediate result gets `(-Inf, Inf)`.

The same role as [`_product_down_up`](@ref), which evaluates the BLAS product twice under
`RoundDown` and `RoundUp`. `O(mnk)` scalar operations, no BLAS, threaded over the columns.

# Reference

T. Ogita, S. M. Rump and S. Oishi, *Accurate sum and dot product*, SIAM J. Sci. Comput. **26**(6)
(2005) 1955-1988, doi 10.1137/030601818.
"""
function _product_down_up_dot2err(A::AbstractVecOrMat{T},
        B::AbstractVecOrMat{T}) where {T <: Union{Float32, Float64}}
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("_product_down_up_dot2err: size(A, 2) = $(size(A, 2)) but size(B, 1) = $(size(B, 1))"))
    δ, c = _dot2err_constants(size(A, 2), T)
    At = Matrix{T}(transpose(A isa AbstractVector ? reshape(A, :, 1) : A))
    Bm = B isa AbstractVector ? reshape(B, :, 1) : B
    m, n = size(At, 2), size(Bm, 2)
    down, up = Matrix{T}(undef, m, n), Matrix{T}(undef, m, n)
    Base.Threads.@threads for j in 1:n
        for i in 1:m
            p, s, e = _dot2err_accumulate(At, Bm, i, j)
            @inbounds down[i, j], up[i, j] = _dot2err_down_up(p, s, e, δ, c)
        end
    end
    B isa AbstractVector && return vec(down), vec(up)
    return down, up
end

# ---------------------------------------------------------------------------------------------
# Sums of products of point matrices, to twice the working precision, in the two forms in which
# Rump and Ogita (2024) call INTLAB's `prodK` with k = 2: one output with an error matrix, and
# the unevaluated sum of two matrices.
#
# `prodK` itself is the splitting scheme of M. Lange and S. M. Rump, "Accurate floating-point
# matrix residuals" (manuscript; in INTLAB from Version 14). For Float32 and Float64 the two
# routines below use it, `_lange_rump_two_terms` in `lange_rump.jl`, which says what of the paper
# is implemented. Where it declines, and for other types such as BigFloat, every entry is one
# Dot2 accumulation of Ogita, Rump and Oishi (2005), some ten operations for each product with a
# fused multiply-add and no BLAS, with the a priori γ² bound.
#
# On nighthawk for RA − I, R = inv(A), cond(A) = 1e12, n = 1000, one thread
# (`probe_prodk_timing.jl`, `probe_prodk_timing2.jl`): the floating-point `R*A - I` 0.017 s; the
# splitting scheme with k = 2, 0.124 s, median error 1.2e-20, median bound 1.8e-16; the Dot2
# accumulation 2.0 s, median error 1.5e-22, median bound 5.7e-16 (16 threads).

# The signed real pairs whose compensated sums are the real and the imaginary part of
# Σ_t sgn_t A_t B_t, the A_t and B_t real or complex point matrices.
function _real_pairs(terms, ::Type{T}) where {T}
    re = Tuple{Matrix{T}, Matrix{T}, T}[]
    im_ = Tuple{Matrix{T}, Matrix{T}, T}[]
    for (A, B, sgn) in terms
        Ar, Br = Matrix{T}(real.(A)), Matrix{T}(real.(B))
        push!(re, (Ar, Br, T(sgn)))
        if eltype(A) <: Complex || eltype(B) <: Complex
            Ai, Bi = Matrix{T}(imag.(A)), Matrix{T}(imag.(B))
            push!(re, (Ai, Bi, -T(sgn)))
            push!(im_, (Ar, Bi, T(sgn)))
            push!(im_, (Ai, Br, T(sgn)))
        end
    end
    return re, im_
end

# Σ_t |A_t||B_t|, rounded up, and the number of products in the real or the imaginary part of an
# entry, whichever is larger.
function _abs_products(terms, ::Type{T}) where {T}
    absprod = mapfoldl(((A, B, _),) -> _product_up(_modulus_up(Matrix(A)), _modulus_up(Matrix(B))),
        _add_up, terms)
    N = sum((eltype(A) <: Complex || eltype(B) <: Complex ? 2 : 1) * size(A, 2)
    for (A, B, _) in terms)
    return absprod, N
end

"""
    _two_term_product_sum(terms) -> (P1, P2, R)

`Σ_t sgn_t A_t B_t` for `terms` an iterable of `(A_t, B_t, sgn_t)`, point matrices, as an
unevaluated sum of two matrices: `|Σ_t sgn_t A_t B_t − P1 − P2| ≤ R` entrywise. `P1` is the
floating-point value and `P2` the accumulated error terms of [`compensated_terms2`](@ref), so
that a further product with the pair keeps twice the working precision; this is the output of
the role of `prodK(..., 'OutputTerms', 2)` with `k = 2` in Rump and Ogita (2024).

`R = κ(γ_{N+1}² Σ_t |A_t||B_t| + 5N eta)`, `N` the number of products in a part of an entry,
`κ = 1` for real data and `3/2 ≥ √2` for complex, where the bound holds for the real and for the
imaginary part: the bound of `compensated_terms2`, with the underflow term of
`_accuracy_term`.
"""
function _two_term_product_sum(terms)
    # Float32 and Float64: the splitting scheme of Lange and Rump, on the BLAS (`lange_rump.jl`);
    # it declines where its conditions fail, and other types have no BLAS to gain from
    lr = _lange_rump_two_terms(terms)
    lr === nothing || return lr
    S = promote_type((promote_type(eltype(A), eltype(B)) for (A, B, _) in terms)...)
    T = real(S)
    re, im_ = _real_pairs(terms, T)
    m, n = size(first(terms)[1], 1), size(first(terms)[2], 2)
    P1, P2 = Matrix{S}(undef, m, n), Matrix{S}(undef, m, n)
    Base.Threads.@threads for j in 1:n
        for i in 1:m
            s, e = compensated_terms2(re, i, j)
            if S <: Complex
                si, ei = compensated_terms2(im_, i, j)
                @inbounds P1[i, j], P2[i, j] = complex(s, si), complex(e, ei)
            else
                @inbounds P1[i, j], P2[i, j] = s, e
            end
        end
    end
    absprod, N = _abs_products(terms, T)
    g = gamma_bound(N + 1, T)
    κ = S <: Complex ? T(3) / T(2) : one(T)
    R = mul_up.(κ, add_up.(mul_up.(mul_up(g, g), absprod), mul_up(T(5 * N), subnormal_min(T))))
    return P1, P2, R
end

"""
    _accurate_product_sum(terms) -> BallMatrix

A ball matrix containing `Σ_t sgn_t A_t B_t` for `terms` an iterable of `(A_t, B_t, sgn_t)`, point
matrices: the two terms of [`_two_term_product_sum`](@ref) added, the radius its bound plus the
rounding of that addition. The role of `[R, E] = prodK(...)` with `k = 2` in Rump and Ogita (2024).
"""
function _accurate_product_sum(terms)
    P1, P2, R = _two_term_product_sum(terms)
    M = similar(P1)
    rad = similar(R)
    @inbounds for i in eachindex(M, R)
        M[i], rad[i] = _rounded_sum(P1[i], P2[i], R[i])
    end
    return BallMatrix(M, rad)
end

# fl(a + b) and r plus the rounding of that sum, which `two_sum` gives exactly
@inline function _rounded_sum(a::T, b::T, r::T) where {T <: AbstractFloat}
    s, e = two_sum(a, b)
    return s, add_up(r, abs(e))
end
@inline function _rounded_sum(a::Complex{T}, b::Complex{T}, r::T) where {T <: AbstractFloat}
    sr, er = two_sum(real(a), real(b))
    si, ei = two_sum(imag(a), imag(b))
    return complex(sr, si), add_up(r, add_up(abs(er), abs(ei)))
end
