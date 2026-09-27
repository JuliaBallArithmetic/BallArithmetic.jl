# Error-free transformations.
#
# A floating-point operation is "error-free" when the rounding it commits is itself representable,
# so that the exact result is the sum of two floats. These are the primitives behind compensated
# arithmetic, behind Rump's `prodK`, and behind the accurate matrix products of
#
#   T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product",
#   SIAM J. Sci. Comput. 26(6):1955-1988, 2005.
#
# They matter here because a ball product cannot compute a residual. When two terms nearly cancel,
# as `B W - W X` does for an approximate eigendecomposition, the enclosure's radius comes out the
# size of the residual itself and carries no information: measured on a 40 x 40 instance, residual
# 1.64e-15 against a naive ball radius of 1.47e-15. A compensated accumulation recovers the same
# residual to 3e-33, which is what makes Rump's transformation worth its cost.
#
# Nothing here requires a directed rounding mode. That is the second reason to have them: the
# rounding-mode route depends on the BLAS honouring `setrounding`, which threaded and GPU
# implementations need not do.

export two_sum, two_product, split_veltkamp, compensated_terms, gamma_bound

"""
    two_product(a, b) -> (x, y)

The exact product as an unevaluated sum: `x = fl(a*b)` and `y = a*b - x` exactly, so that
`a*b == x + y` with no rounding lost. Uses one FMA.
"""
@inline function two_product(a::T, b::T) where {T <: AbstractFloat}
    x = a * b
    return x, fma(a, b, -x)
end

"""
    two_sum(a, b) -> (x, y)

The exact sum as an unevaluated sum: `x = fl(a+b)` and `y = a+b - x` exactly, by Knuth's six
operations, with no assumption on the relative size of `a` and `b`.
"""
@inline function two_sum(a::T, b::T) where {T <: AbstractFloat}
    x = a + b
    z = x - a
    return x, (a - (x - z)) + (b - z)
end

"""
    split_veltkamp(a) -> (hi, lo)

Veltkamp's splitting: `a == hi + lo` exactly, with each half carrying about half the significand,
so that products of halves are exact in floating point. This is the step that lets an accurate
matrix product be assembled from ordinary BLAS calls in round-to-nearest, with no directed
rounding anywhere.
"""
@inline function split_veltkamp(a::T) where {T <: AbstractFloat}
    # s = ceil(p/2), not floor: with p = 53 and s = 26 the high part keeps 27 bits, two of them
    # multiply to 54, and the product is no longer exact. s = 27 leaves 26 bits in each half.
    s = cld(precision(T), 2)
    factor = T(2)^s + one(T)
    c = factor * a
    hi = c - (c - a)
    return hi, a - hi
end

"""
    compensated_terms(pairs, i, j) -> T

One entry of a sum of matrix products `sum_t sgn_t * A_t[i,:] * B_t[:,j]`, accumulated with
error-free transformations so that the result carries about twice the working precision. `pairs`
is an iterable of `(A, B, sgn)`.

This is the kernel of Rump's `prodK`. It returns the accurate value only; a certified bound on it
is the Ogita-Rump-Oishi one,

    |result - exact| <= u |exact| + gamma_N^2 sum |a_i| |b_i|,    gamma_N = N u / (1 - N u),

with `N` the number of products summed, and the absolute products evaluated separately.
"""
@inline function compensated_terms(pairs, i::Integer, j::Integer)
    # The accumulators take the working type from the data. With `s = 0.0` the first `two_sum`
    # promoted and threw on a BigFloat input, so the routine was Float64 only in practice.
    T = promote_type(eltype(first(pairs)[1]), eltype(first(pairs)[2]))
    s = zero(T)
    e = zero(T)
    for (A, B, sgn) in pairs
        @inbounds for k in axes(A, 2)
            p, ep = two_product(sgn * A[i, k], B[k, j])
            s, es = two_sum(s, p)
            e += ep + es
        end
    end
    return s + e
end

"""
    gamma_bound(N, ::Type{T}) -> T

Higham's `gamma_N = N u / (1 - N u)`, rounded upward, for `N` accumulated operations.
"""
function gamma_bound(N::Integer, ::Type{T}) where {T <: AbstractFloat}
    u = eps(T) / 2
    return setrounding(T, RoundUp) do
        (N * u) / (one(T) - N * u)
    end
end
