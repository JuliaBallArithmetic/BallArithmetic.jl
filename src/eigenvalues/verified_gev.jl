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

    return sqrt_up(div_up(T(nY), denom))
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
    verify_generalized_eigenpairs(A::BallMatrix, B::BallMatrix, X̃, λ̃) -> GEVResult

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

How it is computed. `R_g` and `G_g` are enclosed in ball arithmetic from one enclosure of
`B X̃` (Technique 1 of the paper), and `r⁽ⁱ⁾`, `gᵢ`, `β ‖r⁽ⁱ⁾‖₂`, `√gᵢ` are taken from them and
used in both steps 2 and 4 (Techniques 3 and 4). The two norms of Theorem 4 are bounded by
[`upper_bound_L2_opnorm`](@ref); the paper uses `√(‖·‖₁‖·‖∞)` for the first and `‖·‖∞` for the
second, in its (26). Every scalar operation is rounded in the direction that keeps the bound.

What is not reproduced: Technique 2, an a priori estimate of the rounding error of `X̃ᵀ(B X̃)`,
whose place is taken by the ball product; and Theorem 10, the bound of `‖B⁻¹‖₂` from a Cholesky
factor, whose place is taken by the bound of [`compute_beta_bound`](@ref) (Remark 2 of the
paper leaves the method for `β` open).

The input is checked: a `DimensionMismatch` or an `ArgumentError` for wrong sizes or complex
data; a result with `certified = false` when the midpoints or the radii of `A` or `B` are not
symmetric, or `λ̃` is not sorted.

# Reference

S. Miyajima, T. Ogita, S. M. Rump and S. Oishi, "Fast verification for all eigenpairs in
symmetric positive definite generalized eigenvalue problems", Reliable Computing 14 (2010),
24-45.
"""
function verify_generalized_eigenpairs(A::BallMatrix{T}, B::BallMatrix{T},
        X̃::AbstractMatrix{<:Real}, λ̃::AbstractVector{<:Real}) where {T}
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
    ε = Vector{T}(undef, n)
    for i in 1:n
        s = zero(T)
        for k in 1:n
            a = add_up(abs(Rc[k, i]), Rr[k, i])
            s = add_up(s, mul_up(a, a))
        end
        τ[i] = mul_up(β, sqrt_up(s))
        g = sub_down(Gc[i, i], Gr[i, i])
        μ[i] = g > 0 ? sqrt_down(g) : zero(T)
        ε[i] = μ[i] > 0 ? div_up(τ[i], μ[i]) : T(Inf)
    end

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
- `radii`: the **per-eigenvalue** radii of Corollary 3.2; every eigenvalue of the
  pencil lies in `⋃ᵢ {z : |z - λ̃ᵢ| ≤ radii[i]}`, and `radii[i] ≤ radius` always
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
            return MiyajimaGEVResult(false, collect(λ̃), T(Inf), T[], T(Inf), T(Inf),
                "B*X̃ numerically singular: no approximate inverse available")
        end
    else
        collect(Y)
    end
    all(isfinite, Yc) ||
        return MiyajimaGEVResult(false, collect(λ̃), T(Inf), T[], T(Inf), T(Inf),
            "supplied Y contains non-finite entries")

    Yb = BallMatrix(Yc)

    # R₂ = Y·B·X̃ - I
    R2 = Yb * BX - BallMatrix(Matrix{eltype(mid(A))}(I, n, n))
    nR2 = upper_bound_L_inf_opnorm(R2)

    if !(nR2 < 1)
        return MiyajimaGEVResult(false, collect(λ̃), T(Inf), T[], T(Inf), nR2,
            "‖R₂‖_∞ = $nR2 ≥ 1: B, X̃, Y not certifiably nonsingular")
    end

    # R₁ = Y(A X̃ - B X̃ D̃).  D̃ is wrapped as a BallMatrix: multiplying a
    # BallMatrix by a bare `Diagonal` falls through to the generic
    # element-wise path and returns a `Matrix{Ball}`, which the rigorous
    # operator-norm bounds do not accept.
    D̃ = BallMatrix(Matrix(Diagonal(collect(λ̃))))
    R1 = Yb * (A * Xb - BX * D̃)
    nR1 = upper_bound_L_inf_opnorm(R1)

    denom = setrounding(T, RoundDown) do
        one(T) - nR2
    end

    # ── Corollary 3.2 of Miyajima (2014): per-eigenvalue radii ──
    #
    # With t := |S|𝟙 and u := |R₁|𝟙 (row sums, S = I − YBX̃ so |S| = |R₂|), and
    # ‖v‖_w := maxᵢ vᵢ/(1−wᵢ), the radii are  r = u + ‖u‖_t·t.
    #
    # These are never worse than Theorem 1's uniform ε = ‖R₁‖_∞/(1−‖R₂‖_∞):
    # uᵢ ≤ ‖u‖_∞ = ‖R₁‖_∞, tᵢ ≤ ‖t‖_∞ = ‖R₂‖_∞ and
    # ‖u‖_t = maxⱼ uⱼ/(1−tⱼ) ≤ ‖u‖_∞/(1−‖t‖_∞), so
    # rᵢ ≤ ‖R₁‖_∞ + ‖R₁‖_∞‖R₂‖_∞/(1−‖R₂‖_∞) = ε, usually with room to spare.
    absR1 = upper_abs(R1)
    absR2 = upper_abs(R2)
    u = setrounding(T, RoundUp) do
        [sum(view(absR1, i, :)) for i in 1:n]
    end
    t = setrounding(T, RoundUp) do
        [sum(view(absR2, i, :)) for i in 1:n]
    end

    # ‖u‖_t = maxᵢ uᵢ/(1 − tᵢ): numerators up, denominators down.
    unorm_t = zero(T)
    for i in 1:n
        di = setrounding(T, RoundDown) do
            one(T) - t[i]
        end
        di > 0 || return MiyajimaGEVResult(false, collect(λ̃), T(Inf), T[], nR1, nR2,
            "row $i of the residual gives 1 - tᵢ ≤ 0")
        unorm_t = max(unorm_t, setrounding(T, RoundUp) do
            u[i] / di
        end)
    end

    radii = setrounding(T, RoundUp) do
        [u[i] + unorm_t * t[i] for i in 1:n]
    end

    # Theorem 1's uniform radius, kept as the scalar summary. `maximum(radii)` is
    # itself a valid uniform radius and never exceeds it, so report the smaller.
    ε_uniform = setrounding(T, RoundUp) do
        nR1 / denom
    end
    ε = min(ε_uniform, maximum(radii))

    return MiyajimaGEVResult(true, collect(λ̃), ε, radii, nR1, nR2,
        "All eigenvalues lie in ⋃ᵢ {|z - λ̃ᵢ| ≤ rᵢ}")
end
