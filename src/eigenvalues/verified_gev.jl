# Verified Generalized Eigenvalue Problems
#
# Two methods for  Ax = λBx  live here.
#
# 1. `verify_generalized_eigenpairs` — the symmetric-definite algorithm of
#      Miyajima, S., Ogita, T., Rump, S. M., Oishi, S. (2010)
#      "Fast Verification for All Eigenpairs in Symmetric Positive Definite
#      Generalized Eigenvalue Problems", Reliable Computing 14, pp. 24-45.
#    Requires A symmetric and B symmetric positive definite. Returns per-
#    eigenvalue intervals and eigenvector radii, at the price of the prefactor
#    β ≥ √‖B⁻¹‖₂ (`compute_beta_bound`).
#
# 2. `miyajima_gev_enclosure` — Theorem 1 of
#      Miyajima, S. (2010) "Fast enclosure for all eigenvalues in generalized
#      eigenvalue problems", J. Comput. Appl. Math. 233, pp. 2994-3004.
#    The two-residual method: with R₁ = Y(AX̃ - BX̃D̃) and R₂ = YBX̃ - I,
#    ‖R₂‖_∞ < 1 gives min_i|λ - λ̃ᵢ| ≤ ‖R₁‖_∞/(1 - ‖R₂‖_∞) for every eigenvalue.
#    Y is arbitrary, so no inverse is ever formed and all conditioning folds
#    into ‖R₂‖ — the same device as the Miyajima–Rump eigenvalue certification.
#    Needs neither symmetry, nor positive definiteness, nor β.

export verify_generalized_eigenpairs, compute_beta_bound, GEVResult
export miyajima_gev_enclosure, MiyajimaGEVResult

using LinearAlgebra

"""
    GEVResult{T}

The result of [`verify_generalized_eigenpairs`](@ref) for the problem `A x = λ B x` with `A`
symmetric and `B` symmetric positive definite, of order `n`. The eigenvalues `λ₁ ≤ … ≤ λₙ` and
their approximations `λ̃₁ ≤ … ≤ λ̃ₙ` are both in increasing order.

# Fields
- `certified::Bool`: whether the bounds on the eigenvalues were proved. When it is `false` every
  bound below is infinite and `message` says which condition failed.
- `eigenvalue_intervals::Vector{Tuple{T, T}}`: `(λ̃ᵢ − ηᵢ, λ̃ᵢ + ηᵢ)`, the ends rounded outwards;
  the `i`-th interval contains `λᵢ`. The intervals of a cluster may overlap.
- `eigenvector_centers::Matrix{T}`: the approximate eigenvectors `x̃⁽ⁱ⁾`, as columns.
- `eigenvector_radii::Vector{T}`: `ξᵢ` with `‖x̂⁽ⁱ⁾ − x̃⁽ⁱ⁾‖₂ ≤ ξᵢ` for some eigenvector `x̂⁽ⁱ⁾`
  of `λᵢ` (Theorem 7); `Inf` for the indices where the hypotheses of the theorem were not
  proved, which includes every eigenvalue that is not separated from its neighbours.
- `beta::T`: `β ≥ √‖B⁻¹‖₂`.
- `global_bound::T`: `δ` with `|λᵢ − λ̃ᵢ| ≤ δ` for all `i` (Theorem 4).
- `individual_bounds::Vector{T}`: `εᵢ` with `min_j |λⱼ − λ̃ᵢ| ≤ εᵢ` (Theorem 5).
- `separation_bounds::Vector{T}`: the radii `ηᵢ` of the intervals, by (39): `min(εᵢ, δ)` where
  the eigenvalue nearest to `λ̃ᵢ` was proved to be `λᵢ`, and `δ` elsewhere.
- `residual_norm::T`: an upper bound of `‖A X̃ − B X̃ D̃‖₂`.
- `message::String`: empty on success.

All the statements hold for every symmetric `A` and every symmetric `B` in the given balls.
"""
struct GEVResult{T}
    certified::Bool
    eigenvalue_intervals::Vector{Tuple{T, T}}
    eigenvector_centers::Matrix{T}
    eigenvector_radii::Vector{T}
    beta::T
    global_bound::T
    individual_bounds::Vector{T}
    separation_bounds::Vector{T}
    residual_norm::T
    message::String
end

# the result when nothing is proved: every bound infinite
function _gev_failure(λ::Vector{T}, X::Matrix{T}, β, message::String) where {T}
    n = length(λ)
    return GEVResult{T}(false, fill((T(-Inf), T(Inf)), n), X, fill(T(Inf), n), T(β), T(Inf),
        fill(T(Inf), n), fill(T(Inf), n), T(Inf), message)
end


"""
    compute_beta_bound(B::BallMatrix{T}) -> T

Compute a verified upper bound β ≥ √‖B⁻¹‖₂ via the Miyajima–Rump inversion
bound, valid for every matrix in the ball `B`.

# Arguments
- `B::BallMatrix`: Symmetric positive definite interval matrix

# Returns
- `β`: Upper bound on √‖B⁻¹‖₂ valid for **every** matrix in the ball `B`, or
  `Inf` when the residual condition fails (`B` not certifiably nonsingular).

# Algorithm (Miyajima–Rump inversion bound)

Take any floating-point `Y ≈ B⁻¹` and form the residual `R = I - Y·B` in ball
arithmetic. If `‖R‖₂ < 1` then every matrix in the ball is nonsingular and

    ‖B⁻¹‖₂ ≤ ‖Y‖₂ / (1 - ‖R‖₂),

so `β = √(‖Y‖₂/(1-‖R‖₂))`, rounded outward. This is the Neumann/approximate-
inverse lemma used throughout the package (see `verified_cholesky`) and is the
same two-residual device as Theorem 1 of Miyajima (2010): `Y` is *free*, its
quality entering only through `‖R‖`, and no inverse is ever formed.

Because `R` is a **ball** product it absorbs the radius of `B` automatically, so
the bound covers the whole input interval. The previous implementation followed
Theorem 10 with hand-rolled `γₙ` constants applied to `B.c` alone: it ignored
`B.r` entirely (returning the same β for `B ± 0` and for a ball containing
singular matrices) and, when its conditions failed, fell back to
`sqrt(cond(B.c))` — a non-rigorous estimate of the wrong quantity, since
`cond = ‖B‖·‖B⁻¹‖` undershoots `√‖B⁻¹‖` by `√‖B‖` whenever `‖B‖ < 1`.

Carrying no unit-roundoff constants, this version is also type-generic rather
than Float64-only.

# Complexity
O(n³) for the approximate inverse and the ball product.

# Example
```julia
B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))
β = compute_beta_bound(B)
```
"""
function compute_beta_bound(B::BallMatrix{T}) where {T}
    n = size(B, 1)

    # Y is free: any floating-point approximate inverse will do. Its quality
    # only affects how small ‖R‖ comes out.
    Y = try
        inv(mid(B))
    catch
        return T(Inf)          # midpoint numerically singular: no certificate
    end
    all(isfinite, Y) || return T(Inf)

    YB = BallMatrix(Y) * B     # ball product: rounding *and* B's radius included
    R = BallMatrix(Matrix{eltype(mid(B))}(I, n, n)) - YB

    nR = upper_bound_L2_opnorm(R)
    nR < 1 || return T(Inf)    # not certifiably nonsingular over the ball

    nY = upper_bound_L2_opnorm(BallMatrix(Y))

    # ‖B⁻¹‖₂ ≤ ‖Y‖₂ / (1 - ‖R‖₂): numerator up, denominator down.
    denom = sub_down(one(T), T(nR))
    denom > 0 || return T(Inf)

    return sqrt_up(div_up(T(nY), denom))::T
end

# Step 3 of Algorithm 1: the radii η of (39). The sorted approximations are cut into runs at
# every gap proved larger than 2δ. On a run k̲, …, k̄, write ε′ₖ := min(εₖ, δ); each interval
# [λ̃ₖ ± ε′ₖ] contains an eigenvalue (Theorem 5, or Theorem 4 where δ is the smaller). An
# eigenvalue λⱼ with j outside the run is not in it: for j < k̲,
# λⱼ ≤ λ̃ⱼ + δ ≤ λ̃_{k̲−1} + δ < λ̃_{k̲} − δ ≤ λ̃ₖ − ε′ₖ, and symmetrically for j > k̄. If moreover
# ε′ₖ + ε′ₖ₊₁ < λ̃ₖ₊₁ − λ̃ₖ along the run, its intervals are disjoint, each holds one of the
# eigenvalues of the run, and, both lists being sorted, the k-th interval holds λₖ: this is
# Lemma 2 for a run of one and Lemma 3 for a longer one. The hypothesis of Lemma 3 that the gaps
# inside the run are at most 2δ is not used by this argument, so a gap that rounding does not
# prove larger than 2δ may be left inside a run.
function _gev_eigenvalue_radii(λ::Vector{T}, δ::T, ε::Vector{T}) where {T}
    n = length(λ)
    η = fill(δ, n)
    twoδ = mul_up(T(2), δ)
    first_of_run = 1
    for i in 1:n
        (i == n || sub_down(λ[i + 1], λ[i]) > twoδ) || continue
        matched = true
        for k in first_of_run:(i - 1)
            if !(add_up(min(ε[k], δ), min(ε[k + 1], δ)) < sub_down(λ[k + 1], λ[k]))
                matched = false
                break
            end
        end
        if matched
            for k in first_of_run:i
                η[k] = min(ε[k], δ)
            end
        end
        first_of_run = i + 1
    end
    return η
end

"""
    verify_generalized_eigenpairs(A, B, X̃, λ̃; method) -> GEVResult

Error bounds for all the approximate eigenpairs `(λ̃ᵢ, x̃⁽ⁱ⁾)` of `A x = λ B x`, with `A` real
symmetric and `B` real symmetric positive definite: Algorithm 1 of Miyajima, Ogita, Rump and
Oishi (2010). `X̃` is the matrix of the approximate eigenvectors and `λ̃` the vector of the
approximate eigenvalues, in increasing order, as returned by
`eigen(Symmetric(mid(A)), Symmetric(mid(B)))`. Positive definiteness of `B` is proved, not
assumed. The bounds hold for every symmetric `A` and every symmetric `B` in the given balls.

With `R_g := A X̃ − B X̃ D̃`, `D̃ := diag(λ̃)`, and `G_g := X̃ᵀ B X̃`:

1. `β ≥ √‖B⁻¹‖₂`, by [`compute_beta_bound`](@ref).
2. Theorem 4: if `‖I − G_g‖₂ < 1` then `B` is positive definite and
   `|λᵢ − λ̃ᵢ| ≤ δ := β ‖R_g‖₂ / (1 − ‖I − G_g‖₂)` for all `i`.
   Theorem 5: `min_j |λⱼ − λ̃ᵢ| ≤ εᵢ := β ‖r⁽ⁱ⁾‖₂ / √gᵢ`, with `r⁽ⁱ⁾` the `i`-th column of `R_g`
   and `gᵢ` the `i`-th diagonal entry of `G_g`.
3. The radii `ηᵢ` of (39): `min(εᵢ, δ)` where Lemma 2 or Lemma 3 proves that the eigenvalue
   nearest to `λ̃ᵢ` is `λᵢ`, and `δ` elsewhere.
4. Theorem 7: with `ρᵢ` the smaller of `λ̃ᵢ − λ̃ᵢ₋₁ − ηᵢ₋₁` and `λ̃ᵢ₊₁ − λ̃ᵢ − ηᵢ₊₁` (the one that
   exists for `i = 1` and `i = n`) and `ξ̂ᵢ := β ‖r⁽ⁱ⁾‖₂ / ρᵢ`: if `ρᵢ > 0` and `ξ̂ᵢ < √gᵢ`, there
   is an eigenvector `x̂⁽ⁱ⁾` of `λᵢ` with `‖x̂⁽ⁱ⁾ − x̃⁽ⁱ⁾‖₂ ≤ ξᵢ := β ξ̂ᵢ`.

The result is `certified` when steps 1 and 2 succeed; the eigenvalue bounds then hold for all
`i`, clusters included. The eigenvector bound is given index by index, and is `Inf` where the
hypotheses of Theorem 7 are not proved.

Every scalar operation is rounded in the direction that keeps the bound.

# The two versions

`method = :paper` is Algorithm 1 as the paper computes it, for `A` and `B` in floating point
(`Float32` or `Float64`, radius zero); it is the default when `A` and `B` are passed as plain
matrices:

  - `β` by Theorem 10, from a floating-point Cholesky factor `L̃` of `B`, an approximate inverse
    `X_L` of it obtained by substitution, and the a priori bounds of Theorems 8 and 9;
  - `Z̲ = fl▽(B X̃)`, `Z̄ = fl△(B X̃)` and the same two products for `A X̃`, with the rounding mode
    downward and upward (Technique 1);
  - `‖I − G_g‖∞` by the a priori estimate (48), with `Z_c` and `Z_r` of (49) and (50), from one
    product in rounding to nearest (Technique 2);
  - `δ := β √(‖R_g‖₁ ‖R_g‖∞) / (1 − ‖I − G_g‖∞)`, formula (26);
  - `r⁽ⁱ⁾`, `gᵢ` from the enclosures of `R_g` and of `B X̃` already computed (Techniques 3, 4).

One constant differs from the paper's text: its (51) states the backward error of the Cholesky
factorisation with `γₙ`, citing Higham, whose Theorem 10.3 has `γₙ₊₁`; `α_C` is computed with
`γₙ₊₁`. The paper does not write the formula of the enclosure of `R_g`; here column `j` of
`B X̃ D̃` is enclosed by the products of `λ̃ⱼ` with `Z̲` and `Z̄`, rounded outwards.

`method = :ball`, the default for ball matrices, gives the same four steps for matrices with
radii and for any floating-point type: `R_g` and `G_g` are enclosed by ball products, the two
norms of Theorem 4 are bounded by [`upper_bound_L2_opnorm`](@ref), and `β` comes from
[`compute_beta_bound`](@ref) (Remark 2 of the paper leaves the method for `β` open). It does
not use Technique 2 or Theorem 10.

The input is checked: a `DimensionMismatch` or an `ArgumentError` for wrong sizes or complex
data; a result with `certified = false` when the midpoints or the radii of `A` or `B` are not
symmetric, or `λ̃` is not sorted.

# Reference

S. Miyajima, T. Ogita, S. M. Rump and S. Oishi, "Fast verification for all eigenpairs in
symmetric positive definite generalized eigenvalue problems", Reliable Computing 14 (2010),
24-45.
"""
function verify_generalized_eigenpairs(A::BallMatrix{T}, B::BallMatrix{T},
        X̃::AbstractMatrix{<:Real}, λ̃::AbstractVector{<:Real}; method::Symbol = :ball) where {T}
    method in (:ball, :paper) || throw(ArgumentError("method must be :ball or :paper"))
    n = size(A, 1)
    (size(A) == (n, n) && size(B) == (n, n)) ||
        throw(DimensionMismatch("A and B must be square of the same order"))
    (size(X̃) == (n, n) && length(λ̃) == n) ||
        throw(DimensionMismatch("X̃ must be n×n and λ̃ of length n, with n the order of A"))
    (eltype(mid(A)) <: Real && eltype(mid(B)) <: Real) ||
        throw(ArgumentError("verify_generalized_eigenpairs is for real symmetric matrices"))
    X, λ = Matrix{T}(X̃), Vector{T}(λ̃)
    (all(isfinite, X) && all(isfinite, λ)) ||
        return _gev_failure(λ, X, T(Inf), "the approximate eigenpairs are not finite")
    (issymmetric(mid(A)) && issymmetric(rad(A)) && issymmetric(mid(B)) && issymmetric(rad(B))) ||
        return _gev_failure(λ, X, T(Inf),
            "the midpoints and the radii of A and B must be symmetric")
    issorted(λ) ||
        return _gev_failure(λ, X, T(Inf), "the approximate eigenvalues must be in increasing order")

    if method === :paper
        T <: Union{Float32, Float64} || throw(ArgumentError(
            "method = :paper is for Float32 and Float64: its a priori bounds are those of " *
            "IEEE 754 arithmetic"))
        (iszero(rad(A)) && iszero(rad(B))) || throw(ArgumentError(
            "method = :paper is for matrices with radius zero: the paper takes A and B in " *
            "floating point"))
        return _gev_paper(mid(A), mid(B), X, λ)
    end

    # Step 1
    β = compute_beta_bound(B)
    isfinite(β) || return _gev_failure(λ, X, β,
        "no bound of ‖B⁻¹‖₂: B is not proved nonsingular over the ball")

    # Step 2
    bX = BallMatrix(X)
    BX = B * bX
    Rg = A * bX - BX * BallMatrix(Matrix{T}(Diagonal(λ)))
    Gg = bX' * BX
    nR = T(upper_bound_L2_opnorm(Rg))
    nG = T(upper_bound_L2_opnorm(Gg - I))
    nG < 1 || return _gev_failure(λ, X, β,
        "‖I − X̃ᵀBX̃‖₂ < 1 is not proved: the approximate eigenvectors are not close enough " *
        "to orthonormal in the inner product of B")
    δ = div_up(mul_up(β, nR), sub_down(one(T), nG))
    isfinite(δ) || return _gev_failure(λ, X, β, "the bound δ of Theorem 4 is not finite")

    Rc, Rr, Gc, Gr = mid(Rg), rad(Rg), mid(Gg), rad(Gg)
    τ = Vector{T}(undef, n)            # β ‖r⁽ⁱ⁾‖₂, from above
    μ = Vector{T}(undef, n)            # √gᵢ, from below; zero when gᵢ > 0 is not proved
    for i in 1:n
        s = zero(T)
        for k in 1:n
            a = add_up(abs(Rc[k, i]), Rr[k, i])
            s = add_up(s, mul_up(a, a))
        end
        τ[i] = mul_up(β, sqrt_up(s))
        g = sub_down(Gc[i, i], Gr[i, i])
        μ[i] = g > 0 ? sqrt_down(g) : zero(T)
    end

    return _gev_steps_3_4(λ, X, β, δ, τ, μ, nR)
end


# floating-point matrices: the paper's own setting
function verify_generalized_eigenpairs(A::AbstractMatrix{T}, B::AbstractMatrix{T},
        X̃::AbstractMatrix{<:Real}, λ̃::AbstractVector{<:Real};
        method::Symbol = :paper) where {T <: AbstractFloat}
    return verify_generalized_eigenpairs(BallMatrix(Matrix(A)), BallMatrix(Matrix(B)), X̃, λ̃;
        method)
end

# Theorem 5 from τᵢ ≥ β‖r⁽ⁱ⁾‖₂ and 0 ≤ μᵢ ≤ √gᵢ, then Steps 3 and 4 of Algorithm 1
function _gev_steps_3_4(λ::Vector{T}, X::Matrix{T}, β::T, δ::T, τ::Vector{T}, μ::Vector{T},
        nR::T) where {T}
    n = length(λ)
    ε = T[μ[i] > 0 ? div_up(τ[i], μ[i]) : T(Inf) for i in 1:n]

    # Step 3
    η = _gev_eigenvalue_radii(λ, δ, ε)

    # Step 4
    ξ = fill(T(Inf), n)
    if n == 1
        # every nonzero vector is an eigenvector
        μ[1] > 0 && (ξ[1] = zero(T))
    else
        for i in 1:n
            ρ = T(Inf)
            i > 1 && (ρ = min(ρ, sub_down(sub_down(λ[i], λ[i - 1]), η[i - 1])))
            i < n && (ρ = min(ρ, sub_down(sub_down(λ[i + 1], λ[i]), η[i + 1])))
            ρ > 0 || continue
            ξhat = div_up(τ[i], ρ)
            ξhat < μ[i] || continue
            ξ[i] = mul_up(β, ξhat)
        end
    end

    intervals = [(sub_down(λ[i], η[i]), add_up(λ[i], η[i])) for i in 1:n]
    return GEVResult{T}(true, intervals, X, ξ, β, δ, ε, η, nR, "")
end

# ---------------------------------------------------------------------------
# Algorithm 1 as the paper computes it, for floating-point (point) matrices
# ---------------------------------------------------------------------------

# Theorem 10: β ≥ √‖B⁻¹‖₂ from the floating-point Cholesky factor L̃ of B and an approximate
# inverse X_L of it computed by substitution, with the a priori bounds of Theorems 8 and 9.
# u is the unit roundoff and u̲ the underflow unit. Returns Inf when the factorisation does
# not run to completion or a condition of the theorem fails.
#
# One constant is not the paper's. Its (51) states the backward error of the Cholesky
# factorisation with γₙ and cites Higham; Theorem 10.3 of Higham, "Accuracy and Stability of
# Numerical Algorithms" (SIAM, 1996), has γₙ₊₁. α_C is computed here with γₙ₊₁.
function _gev_beta_theorem_10(B::AbstractMatrix{T}) where {T <: Union{Float32, Float64}}
    n = size(B, 1)
    u, u_ = eps(T) / 2, nextfloat(zero(T))
    nu = mul_up(T(n), u)
    nu < 1 || return T(Inf)
    # the Cholesky factorisation, in rounding to nearest
    L = zeros(T, n, n)
    for j in 1:n
        s = B[j, j]
        for k in 1:(j - 1)
            s -= L[j, k] * L[j, k]
        end
        (s > 0 && isfinite(s)) || return T(Inf)
        L[j, j] = sqrt(s)
        for i in (j + 1):n
            t = B[i, j]
            for k in 1:(j - 1)
                t -= L[i, k] * L[j, k]
            end
            L[i, j] = t / L[j, j]
        end
    end
    all(isfinite, L) || return T(Inf)
    # X_L, column by column, by forward substitution on L̃ (X_L eᵢ) = eᵢ
    XL = zeros(T, n, n)
    for i in 1:n
        XL[i, i] = one(T) / L[i, i]
        for k in (i + 1):n
            t = zero(T)
            for m in i:(k - 1)
                t -= L[k, m] * XL[m, i]
            end
            XL[k, i] = t / L[k, k]
        end
    end
    all(isfinite, XL) || return T(Inf)

    absL, absX = abs.(L), abs.(XL)
    rowsum_up(M, i) = (acc = zero(T); for j in axes(M, 2); acc = add_up(acc, M[i, j]); end; acc)
    colsum_up(M, j) = (acc = zero(T); for i in axes(M, 1); acc = add_up(acc, M[i, j]); end; acc)
    matvec_up(M, v) = T[(acc = zero(T); for j in axes(M, 2); acc = add_up(acc, mul_up(M[i, j], v[j])); end; acc)
                        for i in axes(M, 1)]
    Ls = T[rowsum_up(absL, i) for i in 1:n]                 # |L̃| s
    LTs = T[colsum_up(absL, j) for j in 1:n]                # |L̃ᵀ| s
    XLs = matvec_up(absX, Ls)                               # |X_L| |L̃| s
    LLTs = matvec_up(absL, LTs)                             # |L̃| |L̃ᵀ| s
    d1 = T[add_up(T(n), absL[i, i]) for i in 1:n]           # n s + diag(|L̃|)
    d2 = T[add_up(T(n - 1), absL[i, i]) for i in 1:n]       # (n − 1) s + diag(|L̃|)
    norm1_up(v) = (acc = zero(T); for x in v; acc = add_up(acc, x); end; acc)
    γ(m) = (mu = mul_up(T(m), u); div_up(mu, sub_down(one(T), mu)))
    under = div_up(mul_up(T(n), u_), sub_down(one(T), nu))
    ζ1 = add_up(mul_up(γ(n), norm1_up(XLs)), mul_up(under, norm1_up(d1)))
    ζ∞ = add_up(mul_up(γ(n), maximum(XLs)), mul_up(under, maximum(d1)))
    (ζ1 < 1 && ζ∞ < 1) || return T(Inf)
    normX1 = maximum(colsum_up(absX, j) for j in 1:n)
    normX∞ = maximum(rowsum_up(absX, i) for i in 1:n)
    α1 = div_up(normX1, sub_down(one(T), ζ1))
    α∞ = div_up(normX∞, sub_down(one(T), ζ∞))
    n1u = mul_up(T(n - 1), u)
    underC = div_up(mul_up(T(n), u_), sub_down(one(T), n1u))
    mul_up(T(n + 1), u) < 1 || return T(Inf)
    αC = add_up(mul_up(γ(n + 1), maximum(LLTs)), mul_up(underC, maximum(d2)))
    a = mul_up(α1, α∞)
    q = mul_up(a, αC)
    q < 1 || return T(Inf)
    β = sqrt_up(div_up(a, sub_down(one(T), q)))::T
    return isfinite(β) ? β : T(Inf)
end

# Algorithm 1 with Techniques 1 to 4 and Theorem 10, on floating-point matrices
function _gev_paper(A::AbstractMatrix{T}, B::AbstractMatrix{T}, X::Matrix{T},
        λ::Vector{T}) where {T <: Union{Float32, Float64}}
    n = length(λ)
    u, u_ = eps(T) / 2, nextfloat(zero(T))
    nu = mul_up(T(n), u)
    nu < 1 || return _gev_failure(λ, X, T(Inf), "n u < 1 fails")
    γn = div_up(nu, sub_down(one(T), nu))

    # Step 1: Theorem 10
    β = _gev_beta_theorem_10(B)
    isfinite(β) || return _gev_failure(λ, X, β,
        "Theorem 10 gives no bound of ‖B⁻¹‖₂: the Cholesky factorisation of B did not run " *
        "to completion, or α₁ α∞ α_C < 1 fails")

    # Step 2. Technique 1: Z̲ = fl▽(B X̃), Z̄ = fl△(B X̃), used for R_g and for G_g
    Zlo, Zhi = _product_down_up(Matrix{T}(B), X)
    Wlo, Whi = _product_down_up(Matrix{T}(A), X)
    (all(isfinite, Zlo) && all(isfinite, Zhi) && all(isfinite, Wlo) && all(isfinite, Whi)) ||
        return _gev_failure(λ, X, β, "a product overflowed")
    # |R_g| entrywise, R_g = A X̃ − B X̃ D̃: column j of B X̃ D̃ lies between the products of λ̃ⱼ
    # with the two ends
    absR = Matrix{T}(undef, n, n)
    for j in 1:n, i in 1:n
        plo = λ[j] >= 0 ? mul_down(Zlo[i, j], λ[j]) : mul_down(Zhi[i, j], λ[j])
        phi = λ[j] >= 0 ? mul_up(Zhi[i, j], λ[j]) : mul_up(Zlo[i, j], λ[j])
        absR[i, j] = max(abs(sub_down(Wlo[i, j], phi)), abs(sub_up(Whi[i, j], plo)))
    end
    R1, R∞ = zero(T), zero(T)
    for j in 1:n
        acc = zero(T)
        for i in 1:n
            acc = add_up(acc, absR[i, j])
        end
        R1 = max(R1, acc)
    end
    for i in 1:n
        acc = zero(T)
        for j in 1:n
            acc = add_up(acc, absR[i, j])
        end
        R∞ = max(R∞, acc)
    end
    nR = sqrt_up(mul_up(R1, R∞))::T                         # ‖R_g‖₂ ≤ √(‖R_g‖₁ ‖R_g‖∞)

    # Technique 2, (48) to (50)
    Zc = similar(Zlo)
    Zr = similar(Zlo)
    for k in eachindex(Zlo)
        Zc[k] = add_up(Zlo[k], div_up(sub_up(Zhi[k], Zlo[k]), T(2)))
        Zr[k] = sub_up(Zc[k], Zlo[k])
    end
    F = I - X' * Zc                                         # fl□(I − X̃ᵀ Z_c)
    absXt = abs.(permutedims(X))
    function absXt_rowsums_norm(M)                          # ‖ |X̃ᵀ| |M| s ‖∞
        v = T[(acc = zero(T); for j in 1:n; acc = add_up(acc, abs(M[i, j])); end; acc)
              for i in 1:n]
        best = zero(T)
        for i in 1:n
            acc = zero(T)
            for j in 1:n
                acc = add_up(acc, mul_up(absXt[i, j], v[j]))
            end
            best = max(best, acc)
        end
        return best
    end
    normF = zero(T)
    for i in 1:n
        acc = zero(T)
        for j in 1:n
            acc = add_up(acc, abs(F[i, j]))
        end
        normF = max(normF, acc)
    end
    nG = add_up(add_up(add_up(normF, absXt_rowsums_norm(Zr)),
            mul_up(γn, add_up(absXt_rowsums_norm(Zc), one(T)))), mul_up(T(n), u_))
    nG < 1 || return _gev_failure(λ, X, β,
        "‖I − X̃ᵀBX̃‖∞ < 1 is not proved by (48): the approximate eigenvectors are not close " *
        "enough to orthonormal in the inner product of B")
    δ = div_up(mul_up(β, nR), sub_down(one(T), nG))         # (26)
    isfinite(δ) || return _gev_failure(λ, X, β, "the bound δ of (26) is not finite")

    # Techniques 3 and 4: τᵢ = β‖r⁽ⁱ⁾‖₂ from the columns of |R_g|, μᵢ = √gᵢ from x̃⁽ⁱ⁾ᵀ[Z̲, Z̄]eᵢ
    τ = Vector{T}(undef, n)
    μ = Vector{T}(undef, n)
    for i in 1:n
        s = zero(T)
        g = zero(T)
        for k in 1:n
            s = add_up(s, mul_up(absR[k, i], absR[k, i]))
            x = X[k, i]
            g = add_down(g, x >= 0 ? mul_down(x, Zlo[k, i]) : mul_down(x, Zhi[k, i]))
        end
        τ[i] = mul_up(β, sqrt_up(s)::T)
        μ[i] = g > 0 ? sqrt_down(g)::T : zero(T)
    end
    return _gev_steps_3_4(λ, X, β, δ, τ, μ, nR)
end

# ---------------------------------------------------------------------------
# Miyajima (2010), Theorem 1 — the two-residual generalized enclosure
# ---------------------------------------------------------------------------

"""
    MiyajimaGEVResult

Result of [`miyajima_gev_enclosure`](@ref).

# Fields
- `certified::Bool`: whether `‖R₂‖_∞ < 1` held, i.e. whether the enclosure is proved
- `centers::Vector`: the approximate eigenvalues `λ̃`
- `radius`: a common radius valid for all eigenvalues (the smaller of Theorem 1's
  `‖R₁‖_∞/(1-‖R₂‖_∞)` and `maximum(radii)`)
- `radii`: the radii of Corollary 3.2 of Miyajima (2014), one for each `λ̃ᵢ`; every eigenvalue
  of the pencil lies in `⋃ᵢ {z : |z - λ̃ᵢ| ≤ radii[i]}`, and `radii[i] ≤ radius`. Infinite
  when not `certified`
- `nrmR1`, `nrmR2`: the two certified residual norms `‖R₁‖_∞`, `‖R₂‖_∞`
- `message::String`
"""
struct MiyajimaGEVResult{T, CT}
    certified::Bool
    centers::Vector{CT}
    radius::T
    radii::Vector{T}
    nrmR1::T
    nrmR2::T
    message::String
end

"""
    miyajima_gev_enclosure(A::BallMatrix, B::BallMatrix, X̃, λ̃; Y = nothing)

Enclose **all** eigenvalues of the generalized problem `Ax = λBx` by Theorem 1 of

> S. Miyajima, *Fast enclosure for all eigenvalues in generalized eigenvalue
> problems*, J. Comput. Appl. Math. **233** (2010) 2994–3004.

Given approximate eigenpairs `X̃`, `λ̃` (so that `AX̃ ≈ BX̃D̃`, `D̃ = diag(λ̃)`) and
an **arbitrary** matrix `Y`, set

    R₁ := Y(AX̃ - BX̃D̃),    R₂ := YBX̃ - I .

If `‖R₂‖_∞ < 1` then `B`, `X̃` and `Y` are nonsingular and every eigenvalue `λ`
satisfies `minᵢ |λ - λ̃ᵢ| ≤ ε` with

    ε = ‖R₁‖_∞ / (1 - ‖R₂‖_∞) .

The returned `radii` are sharper: Corollary 3.2 of Miyajima (2014) replaces the
two global norms by row sums `u = |R₁|𝟙`, `t = |R₂|𝟙`, giving

    radii = u + ‖u‖_t · t,    ‖v‖_w := maxᵢ vᵢ/(1 - wᵢ),

which satisfies `radii[i] ≤ ε` for every `i` — usually with room to spare, since
a row with a small residual is no longer charged the worst row's.

`Y` defaults to a floating-point `(B X̃)⁻¹` (Remark 2 of the paper). Its accuracy
affects only how small `ε` comes out, never correctness — the same "`Y` is free"
device as the Miyajima–Rump eigenvalue certification: no inverse is ever formed,
all conditioning is folded into the single residual `‖R₂‖`.

Both residuals are evaluated in ball arithmetic, so the enclosure covers every
matrix in the input balls `A` and `B`, including all rounding errors.

Unlike [`verify_generalized_eigenpairs`](@ref) — which implements the
symmetric-definite algorithm and returns per-eigenvalue intervals plus
eigenvector bounds — this needs **no** symmetry, **no** positive definiteness and
**no** `√‖B⁻¹‖₂` prefactor; it returns one common radius for all eigenvalues.

# Example
```julia
A = BallMatrix([4.0 1.0; 1.0 3.0])
B = BallMatrix([2.0 0.5; 0.5 2.0])
F = eigen(Symmetric(mid(A)), Symmetric(mid(B)))
res = miyajima_gev_enclosure(A, B, Matrix(F.vectors), collect(F.values))
res.certified && (res.radius)
```
"""
function miyajima_gev_enclosure(A::BallMatrix{T}, B::BallMatrix{T},
        X̃::AbstractMatrix, λ̃::AbstractVector; Y = nothing) where {T}
    n = size(A, 1)
    (size(A) == size(B) == (n, n) && size(X̃) == (n, n) && length(λ̃) == n) ||
        throw(DimensionMismatch("A, B, X̃ must be n×n and λ̃ of length n"))

    Xb = BallMatrix(collect(X̃))
    BX = B * Xb

    # Y is arbitrary (Remark 2: Y ≈ (BX̃)⁻¹). A bad Y only inflates ‖R₂‖.
    Yc = if Y === nothing
        try
            inv(mid(BX))
        catch
            return MiyajimaGEVResult(false, collect(λ̃), T(Inf), fill(T(Inf), n), T(Inf), T(Inf),
                "B*X̃ numerically singular: no approximate inverse available")
        end
    else
        collect(Y)
    end
    all(isfinite, Yc) ||
        return MiyajimaGEVResult(false, collect(λ̃), T(Inf), fill(T(Inf), n), T(Inf), T(Inf),
            "supplied Y contains non-finite entries")

    Yb = BallMatrix(Yc)

    # R₂ = Y·B·X̃ - I
    R2 = Yb * BX - BallMatrix(Matrix{eltype(mid(A))}(I, n, n))
    nR2 = upper_bound_L_inf_opnorm(R2)

    if !(nR2 < 1)
        return MiyajimaGEVResult(false, collect(λ̃), T(Inf), fill(T(Inf), n), T(Inf), nR2,
            "‖R₂‖_∞ = $nR2 ≥ 1: B, X̃, Y not certifiably nonsingular")
    end

    # R₁ = Y(A X̃ - B X̃ D̃).  D̃ is wrapped as a BallMatrix: multiplying a
    # BallMatrix by a bare `Diagonal` falls through to the generic
    # element-wise path and returns a `Matrix{Ball}`, which the rigorous
    # operator-norm bounds do not accept.
    D̃ = BallMatrix(Matrix(Diagonal(collect(λ̃))))
    R1 = Yb * (A * Xb - BX * D̃)
    nR1 = upper_bound_L_inf_opnorm(R1)

    # Corollary 3.2 of Miyajima (2014): with t := |R₂|𝟙 and u := |R₁|𝟙 (row sums) and
    # ⟨u⟩_t := maxᵢ uᵢ/(1 − tᵢ), the radii are r = u + ⟨u⟩_t t. They are at most the ε of
    # Theorem 1: uᵢ ≤ ‖R₁‖_∞, tᵢ ≤ ‖R₂‖_∞ and ⟨u⟩_t ≤ ‖R₁‖_∞/(1 − ‖R₂‖_∞).
    absR1 = upper_abs(R1)
    absR2 = upper_abs(R2)
    u = [_colrow_sums_up(view(absR1, i:i, :))[2] for i in 1:n]
    t = [_colrow_sums_up(view(absR2, i:i, :))[2] for i in 1:n]
    unorm_t = _miyajima2014a_lem2_3(u, t)
    isfinite(unorm_t) || return MiyajimaGEVResult(false, collect(λ̃), T(Inf), fill(T(Inf), n),
        T(nR1), T(nR2), "a row of the residual gives 1 − tᵢ ≤ 0")
    radii = T[add_up(u[i], mul_up(unorm_t, t[i])) for i in 1:n]

    # the uniform radius of Theorem 1, and the largest of the radii, which is one too
    ε = min(div_up(T(nR1), sub_down(one(T), T(nR2))), maximum(radii))

    return MiyajimaGEVResult(true, collect(λ̃), ε, radii, nR1, nR2,
        "All eigenvalues lie in ⋃ᵢ {|z - λ̃ᵢ| ≤ rᵢ}")
end
