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
# it is. `*` nonetheless continues to dispatch to MMul4: making this the default is a separate
# decision, since the kernel's own tests assert particular radii in places.
#
# The Ozaki scheme, which recovers full BLAS speed by splitting until the accumulations are
# themselves exact, would remove even the k = 1600 penalty and is not implemented.

# TODO (next version): compute the radius in round-to-nearest, with no `setrounding`. The three
# products in it, |A||B|, |A| rad(B) and rad(A)(|B| + rad(B)), have nonnegative factors, so a
# nearest-rounded GEMM s̃ of k terms satisfies s <= s̃/(1 - γ_k), and one elementwise `mul_up` by
# an upper bound of 1/(1 - γ_k) makes it an upper bound; the accuracy term and the sums are then
# elementwise `mul_up`/`add_up`, O(n²) emulated operations. Nothing would then depend on the BLAS
# honouring the rounding mode, which is what OpenBLASConsistentFPCSR_jll is here for. To settle
# first: that OpenBLAS's gemm satisfies the k-term error model (FMA does; a Strassen-type scheme
# would not), and an absolute term for underflow, where γ_k does not hold. The cost on the radius
# is a relative γ_k, ~1e-12 at k = 5000; the γ_N² of the midpoint is untouched. The plan for the
# next version is to make the prodK route, this one, the default `*` in place of MMul4's
# directed-rounding BLAS products; the measured cost above (faster below k = 400, 3.6x slower at
# 1600) is what that decision has to weigh at large k.

export mmul_ogita_rump_oishi_2005

# |a| for a real or complex matrix, structure preserved, rounded up where the modulus is inexact.
# The complex modulus goes through `abs_up`: `abs` is `hypot`, which does not honour the rounding
# mode, and under RoundUp it fell below the exact modulus for 77915 of 10^6 random arguments.
_modulus_up(M::AbstractMatrix{<:Real}) = abs_preserving_structure(M)
_modulus_up(M::AbstractMatrix{<:Complex}) = abs_up.(M)

# The Ogita-Rump-Oishi accuracy term for one block of products: u |computed| + gamma_N^2 * absprod,
# with N the number of summands. `absprod` must already bound sum_i |a_i| |b_i| from above.
function _accuracy_term(absC::AbstractMatrix{T}, absprod::AbstractMatrix{T}, N::Integer,
        ::Type{T}) where {T <: AbstractFloat}
    u = eps(T) / 2
    g = gamma_bound(N, T)
    return setrounding(T, RoundUp) do
        u .* absC .+ (g * g) .* absprod
    end
end

"""
    mmul_ogita_rump_oishi_2005(A::BallMatrix, B::BallMatrix) -> BallMatrix

Product of two ball matrices whose midpoint is accumulated with error-free transformations and
whose radius charges the Ogita-Rump-Oishi (2005) accuracy bound rather than the rounding of a BLAS
product.

For every `X ∈ A` and `Y ∈ B` the returned ball contains `XY`. The radius is the sum of

* the interval part, `rad(A)(|mid(B)| + rad(B)) + |mid(A)| rad(B)` entrywise, which is what any
  midpoint-radius product must carry and is identical to [`MMul4`](@ref)'s; and
* the accuracy part, `u|C| + γ_N² Σᵢ|aᵢ||bᵢ|` with `N` the number of products summed and
  `γ_N = Nu/(1−Nu)`, from [`compensated_terms`](@ref)'s docstring.

The `γ_N²` is the point: `MMul3`, `MMul4` and `MMul5` all charge the accumulation over the `k`
summands at order `γ_k`, and at `k = 5000` in `Float64` that is `1.1e-12` against this routine's
`1.2e-28`.

The midpoint never changes the rounding mode; `two_product` uses one FMA and `two_sum` is Knuth's
six additions, both in round-to-nearest. Only the radius is accumulated under `RoundUp`.

**Cost.** `O(mnk)` scalar operations with no BLAS, threaded over the output columns. Measured on 16
threads against the ball `*` on complex matrices with radius zero, it is *faster* below about
`k = 400` — `0.55×` at `k = 200`, `0.41×` at `400` — and `1.32×` at `800`, `3.59×` at `1600`, with the
radius ratio improving from `0.0075` to `0.0029` over the same range. `*` nonetheless continues to
dispatch to [`MMul4`](@ref); making this the default is a separate decision, since some of the
kernel's tests assert particular radii. Removing even the `k = 1600` penalty needs the Ozaki scheme,
which is not implemented here.

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
    end
    # N summands per entry: k for a real product, 2k for a complex one, whose real and imaginary
    # parts are each a difference or sum of two real dot products of length k
    N = S <: Complex ? 2k : k
    rC = _accuracy_term(_modulus_up(mC), absprod, N, T)

    # the interval part, as in MMul4
    rC = setrounding(T, RoundUp) do
        rC .+ absA * rB .+ rA * (absB .+ rB)
    end
    # a complex radius must cover both components; the accuracy term above bounds each part, so
    # their sum bounds the modulus
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
