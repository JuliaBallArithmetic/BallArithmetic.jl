export collatz_upper_bound_L2_opnorm,
       upper_bound_L1_opnorm, upper_bound_L_inf_opnorm, upper_bound_L2_opnorm,
       svd_bound_L2_opnorm, upper_bound_frobenius, upper_abs

"""
    upper_abs(A)

Entrywise upper bound of `|C| + R` for a ball matrix with midpoint `C` and radius `R`, so that
`|X| ≤ upper_abs(A)` entrywise for every `X` in the ball; for a plain matrix, of `|A|`.

Every norm bound below goes through this matrix: the spectral, Frobenius, ℓ¹ and ℓ^∞ norms are
monotone in the moduli of the entries (Higham, *Accuracy and Stability of Numerical Algorithms*,
2nd ed., Lemma 6.6, for the spectral norm), so a bound on the norm of `|C| + R` bounds the norm of
every matrix in the ball, and it is never larger than the norm of `C` plus the norm of `R`.

For `Float32` and `Float64` the moduli are formed with [`abs_up`](@ref), as `√(re² + im²)` rounded
upward operation by operation, because the complex `abs` is `hypot`, which is not correctly
rounded and so is not a bound under `RoundUp`.
"""
function upper_abs(A::BallMatrix{T}) where {T <: Union{Float32, Float64}}
    C, R = A.c, A.r
    out = Matrix{T}(undef, size(C))
    @inbounds for j in axes(C, 2), i in axes(C, 1)
        out[i, j] = add_up(abs_up(C[i, j]), R[i, j])
    end
    return out
end
function upper_abs(A::BallMatrix{T}) where {T}
    absA = setrounding(T, RoundUp) do
        return abs.(A.c) + A.r
    end
    return absA
end
upper_abs(A::AbstractMatrix{<:Union{Float32, Float64, Complex{Float32}, Complex{Float64}}}) =
    [abs_up(x) for x in A]

"""
    collatz_upper_bound_L2_opnorm(A::BallMatrix; iterates=10)

Give a rigorous upper bound on the ℓ² norm of the matrix `A`
by using the Collatz theorem.

We use Perron theory here: if for two matrices with `B` positive
`|A| < B` we have ρ(A)<=ρ(B) by Wielandt's theorem
[Wielandt's theorem](https://mathworld.wolfram.com/WielandtsTheorem.html)

The keyword argument `iterates` is used to establish how many
times we are iterating the vector of ones before we use Collatz's
estimate.
"""
function collatz_upper_bound_L2_opnorm(A::BallMatrix{T}; iterates = 5) where {T}
    m, k = size(A)
    # Use the real type of the matrix elements for the iteration vectors
    # This ensures BigFloat matrices get BigFloat precision in the iteration
    RT = real(eltype(A.c))
    x_old = ones(RT, k)
    x_new = x_old

    absA = upper_abs(A)
    #@info opnorm(absA, Inf)

    # @info absA

    # using Collatz theorem
    lam = setrounding(T, RoundUp) do
        for _ in 1:iterates
            x_old = x_new
            x_new = absA' * (absA * x_old)
            # @info x_new
            # @info x_old
            # @info maximum(x_new ./ x_old)
        end
        lam = zero(RT)
        for i in 1:k
            if x_old[i] != zero(RT)
                lam = max(lam, x_new[i] / x_old[i])
            end
        end
        return lam
    end
    return sqrt_up(lam)
end

using LinearAlgebra

# the largest column sum and the largest row sum of a nonnegative matrix, rounded upward
function _colrow_sums_up(M::AbstractMatrix{T}) where {T}
    m, n = size(M)
    c1 = zero(T)
    rs = zeros(T, m)
    @inbounds for j in 1:n
        s = zero(T)
        for i in 1:m
            s = add_up(s, M[i, j])
            rs[i] = add_up(rs[i], M[i, j])
        end
        c1 = max(c1, s)
    end
    return c1, (m == 0 ? zero(T) : maximum(rs))
end

"""
    upper_bound_L1_opnorm(A::BallMatrix{T})

Rigorous upper bound on the ℓ¹ operator norm (largest column sum) of every matrix in the ball
`A`: the largest column sum of [`upper_abs`](@ref)`(A)`, rounded upward.
"""
function upper_bound_L1_opnorm(A::BallMatrix{T}) where {T <: Union{Float32, Float64}}
    return _colrow_sums_up(upper_abs(A))[1]
end
function upper_bound_L1_opnorm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, 1) + opnorm(A.r, 1)
    end
    return norm
end

"""
    upper_bound_L_inf_opnorm(A::BallMatrix{T})

Rigorous upper bound on the ℓ^∞ operator norm (largest row sum) of every matrix in the ball `A`:
the largest row sum of [`upper_abs`](@ref)`(A)`, rounded upward.
"""
function upper_bound_L_inf_opnorm(A::BallMatrix{T}) where {T <: Union{Float32, Float64}}
    return _colrow_sums_up(upper_abs(A))[2]
end
function upper_bound_L_inf_opnorm(A::BallMatrix{T}) where {T}
    norm = setrounding(T, RoundUp) do
        return opnorm(A.c, Inf) + opnorm(A.r, Inf)
    end
    return norm
end

"""
    upper_bound_frobenius(A::BallMatrix) -> T

Rigorous upper bound on the Frobenius norm of every matrix in the ball `A`: the Frobenius norm of
[`upper_abs`](@ref)`(A)`, accumulated with upward rounding.
"""
function upper_bound_frobenius(A::BallMatrix{T}) where {T <: Union{Float32, Float64}}
    s = zero(T)
    @inbounds for x in upper_abs(A)
        s = add_up(s, mul_up(x, x))
    end
    return sqrt_up(s)
end
upper_bound_frobenius(A::BallMatrix) = upper_bound_norm(A, 2)

"""
    upper_bound_L2_opnorm(A::BallMatrix; svd = false)

Rigorous upper bound on the spectral norm of every matrix in the ball `A`: the least of the
Collatz bound, the interpolation bound `√(‖·‖₁‖·‖∞)`, the Frobenius bound and, when `svd = true`,
the upper end of the largest singular value enclosed by [`svdbox`](@ref) (an O(n³) factorisation,
see [`svd_bound_L2_opnorm`](@ref)). The first three are computed on [`upper_abs`](@ref)`(A)`.
"""
function upper_bound_L2_opnorm(A::BallMatrix{T}; svd::Bool = false) where {T}
    isempty(A.c) && return zero(real(T))
    norm1 = upper_bound_L1_opnorm(A)
    norminf = upper_bound_L_inf_opnorm(A)
    v = min(collatz_upper_bound_L2_opnorm(A), sqrt_up(mul_up(norm1, norminf)),
        upper_bound_frobenius(A))
    if svd
        s = svd_bound_L2_opnorm(A)
        isfinite(s) && (v = min(v, s))
    end
    return v
end

"""
    svd_bound_L2_opnorm(A::BallMatrix{T})

Returns a rigorous upper bound on the ℓ²-norm (spectral norm) of the ball matrix `A`
using the rigorous SVD enclosure from [`rigorous_svd`](@ref).

This method computes the largest certified singular value, providing the tightest
possible upper bound on `‖A‖₂`. The bound is essentially exact (typically <0.01%
overestimation), but requires O(n³) computation for the SVD.

# Comparison with other L2 norm methods

| Method | Speed | Accuracy | Best use case |
|--------|-------|----------|---------------|
| `svd_bound_L2_opnorm` | O(n³) | ~0% overest. | When accuracy is critical |
| `collatz_upper_bound_L2_opnorm` | Fast | 0-500% overest.* | Structured matrices |
| `upper_bound_L2_opnorm` | Fast | min of above | General use |

*Collatz performs well on structured matrices (tridiagonal: ~0%, Hilbert: ~0%,
diagonally dominant: ~26%) but poorly on random matrices (~200-500%).

# Example
```julia
A = BallMatrix(randn(50, 50))
bound = svd_bound_L2_opnorm(A)  # Tight bound, slower
fast_bound = upper_bound_L2_opnorm(A)  # Looser bound, faster
```

See also: [`upper_bound_L2_opnorm`](@ref), [`collatz_upper_bound_L2_opnorm`](@ref),
[`rigorous_svd`](@ref)
"""
function svd_bound_L2_opnorm(A::BallMatrix{T}; method::Symbol = :auto) where {T}
    _, hi = svd_bounds(A; method)
    return isempty(hi) ? zero(real(T)) : hi[1]
end

"""
    svd_bound_L2_opnorm_inverse(A::BallMatrix)

Returns a rigorous upper bound on the ℓ²-norm of the inverse of the
ball matrix `A` using the rigorous enclosure for the singular values
implemented in svd/svd.jl
"""
function svd_bound_L2_opnorm_inverse(A::BallMatrix)
    σ = svdbox(A)

    if in(0, σ[end])
        return +Inf
    end

    inv_inf = Ball(1.0) / σ[end]

    return @up inv_inf.c + inv_inf.r
end

using LinearAlgebra
"""
    svd_bound_L2_resolvent(A::BallMatrix, lam::Ball)

Returns a rigorous upper bound on the ℓ²-norm of the resolvent
of `A` at `λ`, i.e., ||(A-λ)^{-1}||_{ℓ²}
"""
svd_bound_L2_resolvent(A::BallMatrix, λ::Ball) = svd_bound_L2_opnorm_inverse(A - λ * I)
