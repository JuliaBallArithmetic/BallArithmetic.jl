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
    GEVResult

Result structure for verified generalized eigenvalue problem.

All numeric fields use Float64 precision. This struct is not currently
parametric; extension to other numeric types would require making it GEVResult{T}.

# Fields
- `success::Bool`: Whether verification succeeded
- `eigenvalue_intervals::Vector{Tuple{Float64, Float64}}`: Verified intervals [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] for each eigenvalue
- `eigenvector_centers::Matrix{Float64}`: Approximate eigenvectors (centers)
- `eigenvector_radii::Vector{Float64}`: Verified radii ξᵢ for eigenvector balls
- `beta::Float64`: Preconditioning factor β ≥ √‖B⁻¹‖₂
- `global_bound::Float64`: Global eigenvalue bound δ̂ (Theorem 4)
- `individual_bounds::Vector{Float64}`: Individual eigenvalue bounds ε (Theorem 5)
- `separation_bounds::Vector{Float64}`: Separation bounds η (Lemma 2)
- `residual_norm::Float64`: Norm of residual matrix ‖Rg‖₂
- `message::String`: Diagnostic message (especially if success = false)

# Interpretation
- If `success = true`: All eigenvalue intervals are guaranteed to contain exactly
  one true eigenvalue, and all eigenvector balls contain the corresponding normalized
  eigenvector, rigorously accounting for all matrices in the input intervals [A] and [B].
- If `success = false`: Check `message` for diagnostic information. Common failures:
  - Approximate eigenvectors not sufficiently orthogonal (‖I - Gg‖₂ >= 1)
  - Eigenvalues too clustered to separate
  - B not positive definite
"""
struct GEVResult
    success::Bool
    eigenvalue_intervals::Vector{Tuple{Float64, Float64}}
    eigenvector_centers::Matrix{Float64}
    eigenvector_radii::Vector{Float64}

    # Diagnostic information
    beta::Float64
    global_bound::Float64
    individual_bounds::Vector{Float64}
    separation_bounds::Vector{Float64}
    residual_norm::Float64
    message::String
end

"""
    compute_beta_bound(B::BallMatrix) -> Float64

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
    denom = setrounding(T, RoundDown) do
        one(T) - nR
    end
    denom > 0 || return T(Inf)

    return setrounding(T, RoundUp) do
        sqrt(nY / denom)
    end
end

"""
    compute_residual_matrix(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector) -> Matrix

Compute residual matrix Rg = AX̃ - BX̃D̃ where D̃ = diag(λ̃).

Uses interval arithmetic to account for uncertainties in A and B.

# Complexity
O(n³) - dominated by matrix multiplications
"""
function compute_residual_matrix(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector)
    D̃ = Diagonal(λ̃)

    # Convert to Ball matrices for exact arithmetic
    AX = A * X̃
    BX = B * X̃
    BXD = BX * D̃

    # Residual with interval arithmetic
    Rg = AX - BXD

    return Rg
end

"""
    compute_gram_matrix(B::BallMatrix, X̃::Matrix) -> Matrix

Compute Gram matrix Gg = X̃ᵀBX̃.

This matrix should be close to identity if X̃ contains approximate eigenvectors
that are nearly orthogonal with respect to the B inner product.

# Complexity
O(n³) - matrix multiplications
"""
function compute_gram_matrix(B::BallMatrix, X̃::Matrix)
    BX = B * X̃
    Gg = X̃' * BX
    return Gg
end

"""
    compute_global_eigenvalue_bound(Rg, Gg, β::Float64) -> Float64

Compute global eigenvalue bound δ̂ using Theorem 4.

δ̂ = (β‖Rg‖₂) / (1 - ‖I - Gg‖₂)

All eigenvalues satisfy |λⱼ - λ̃ⱼ| ≤ δ̂.

# Returns
- δ̂ if successful, Inf if ‖I - Gg‖₂ >= 1
"""
function compute_global_eigenvalue_bound(Rg::BallMatrix, Gg::BallMatrix, β::Float64)
    n = size(Rg, 1)

    # Compute norms using interval arithmetic
    # Rg is a BallMatrix
    norm_Rg = svd_bound_L2_opnorm(Rg)

    # I - Gg: compute the identity minus Gg as a BallMatrix
    I_ball = BallMatrix(Matrix{Float64}(I, n, n))
    I_Gg = I_ball - Gg
    norm_I_Gg = svd_bound_L2_opnorm(I_Gg)

    # Check condition
    denominator = 1 - norm_I_Gg
    if denominator <= 0
        @warn "Global bound failed: ‖I - Gg‖₂ >= 1, eigenvectors not sufficiently orthogonal"
        return Inf
    end

    δ̂ = (β * norm_Rg) / denominator

    return δ̂
end

"""
    compute_individual_eigenvalue_bounds(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, β::Float64) -> Vector{Float64}

Compute individual eigenvalue bounds εᵢ using Theorem 5.

εᵢ = (β‖r⁽ⁱ⁾‖₂) / √gᵢ

where r⁽ⁱ⁾ = Ax̃⁽ⁱ⁾ - λ̃ᵢBx̃⁽ⁱ⁾ and gᵢ = x̃⁽ⁱ⁾ᵀBx̃⁽ⁱ⁾.

At least one true eigenvalue lies in [λ̃ᵢ - εᵢ, λ̃ᵢ + εᵢ].

# Complexity
O(n²) using Technique 3 (reuse Rg and Gg if available)
"""
function compute_individual_eigenvalue_bounds(
        A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, β::Float64)
    n = length(λ̃)
    ε = zeros(Float64, n)

    for i in 1:n
        # Individual residual r⁽ⁱ⁾ = Ax̃⁽ⁱ⁾ - λ̃ᵢBx̃⁽ⁱ⁾
        x_i = X̃[:, i]
        Ax_i = A * x_i
        Bx_i = B * x_i
        r_i = Ax_i - λ̃[i] * Bx_i

        # Rigorous upper bound on ‖r⁽ⁱ⁾‖₂ using existing infrastructure
        # Uses upper_bound_norm which computes ‖mid‖₂ + ‖rad‖₂ with RoundUp
        norm_r_i = upper_bound_norm(r_i, 2)

        # Gram element gᵢ = x̃⁽ⁱ⁾ᵀBx̃⁽ⁱ⁾
        g_i = dot(x_i, Bx_i)

        # Rigorous lower bound on √gᵢ for denominator
        # Since gᵢ is in denominator, we need lower bound to get upper bound on result
        if isa(g_i, Ball)
            # Lower bound on |g_i|: max(0, |g_i.c| - g_i.r)
            g_i_lower = setrounding(Float64, RoundDown) do
                max(0.0, abs(g_i.c) - g_i.r)
            end
        else
            g_i_lower = abs(g_i)
        end

        if g_i_lower <= 0
            @warn "Individual bound $i: gᵢ lower bound ≤ 0, using large bound"
            ε[i] = Inf
        else
            # Rigorous: β * norm_r_i / sqrt_down(g_i_lower)
            ε[i] = setrounding(Float64, RoundUp) do
                sqrt_g_lower = setrounding(Float64, RoundDown) do
                    sqrt(g_i_lower)
                end
                (β * norm_r_i) / sqrt_g_lower
            end
        end
    end

    return ε
end

"""
    compute_eigenvalue_separation(λ̃::Vector, δ̂::Float64, ε::Vector{Float64}) -> Vector{Float64}

Compute separation bounds η using Lemma 2.

Finds the largest ηᵢ ≤ min(δ̂, εᵢ) such that intervals [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ]
are pairwise disjoint.

This ensures each interval contains exactly one eigenvalue.

# Algorithm
Iteratively shrink overlapping intervals until all are disjoint.

# Complexity
O(n²) in worst case (highly clustered eigenvalues)
"""
function compute_eigenvalue_separation(λ̃::Vector, δ̂::Float64, ε::Vector{Float64})
    n = length(λ̃)

    # Initialize with minimum of global and individual bounds
    η = min.(δ̂, ε)

    # Handle infinite bounds
    for i in 1:n
        if isinf(η[i])
            η[i] = δ̂
        end
    end

    # Iteratively resolve overlaps
    max_iterations = 100
    for iter in 1:max_iterations
        changed = false

        for i in 1:(n - 1)
            for j in (i + 1):n
                # Check if intervals [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] and [λ̃ⱼ - ηⱼ, λ̃ⱼ + ηⱼ] overlap
                if λ̃[i] + η[i] > λ̃[j] - η[j]
                    # They overlap, shrink both to half the gap
                    gap = (λ̃[j] - λ̃[i]) / 2

                    if η[i] > gap
                        η[i] = gap
                        changed = true
                    end
                    if η[j] > gap
                        η[j] = gap
                        changed = true
                    end
                end
            end
        end

        if !changed
            break
        end

        if iter == max_iterations
            @warn "Eigenvalue separation did not converge after $max_iterations iterations"
        end
    end

    return η
end

"""
    compute_eigenvector_bounds(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, η::Vector{Float64}, β::Float64) -> Vector{Float64}

Compute eigenvector bounds ξᵢ using Theorem 7.

ξᵢ = β² ‖r⁽ⁱ⁾‖₂ / ρᵢ

where ρᵢ is the distance to the nearest other eigenvalue interval.

Guarantees ‖x̂⁽ⁱ⁾ - x̃⁽ⁱ⁾‖₂ ≤ ξᵢ for the true eigenvector x̂⁽ⁱ⁾.

# Complexity
O(n²) using Technique 4 (reuse residual norms)
"""
function compute_eigenvector_bounds(
        A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, η::Vector{Float64}, β::Float64)
    n = length(λ̃)
    ξ = zeros(Float64, n)

    for i in 1:n
        # Compute individual residual norm
        x_i = X̃[:, i]
        Ax_i = A * x_i
        Bx_i = B * x_i
        r_i = Ax_i - λ̃[i] * Bx_i

        # Rigorous upper bound on ‖r⁽ⁱ⁾‖₂ using directed rounding
        # Per Rump-Ogita 2024: all certification bounds must use RoundUp
        # Upper bound on |x|² is (|x.c| + x.r)², NOT x.c² + x.r² (which underestimates)
        norm_r_i = setrounding(Float64, RoundUp) do
            sqrt(sum((abs(x.c) + x.r)^2 for x in r_i))
        end

        # Compute ρᵢ: distance to nearest other eigenvalue interval
        # Use RoundDown since ρᵢ is in denominator
        ρ_i = Inf

        if i > 1
            # Distance to previous eigenvalue (rigorous lower bound)
            dist_prev = setrounding(Float64, RoundDown) do
                (λ̃[i] - η[i]) - (λ̃[i - 1] + η[i - 1])
            end
            ρ_i = min(ρ_i, dist_prev)
        end

        if i < n
            # Distance to next eigenvalue (rigorous lower bound)
            dist_next = setrounding(Float64, RoundDown) do
                (λ̃[i + 1] - η[i + 1]) - (λ̃[i] + η[i])
            end
            ρ_i = min(ρ_i, dist_next)
        end

        if ρ_i <= 0 || isinf(ρ_i)
            @warn "Eigenvector bound $i: ρᵢ ≤ 0 or infinite, eigenvalues not separated"
            ξ[i] = Inf
        else
            # Rigorous upper bound: numerator up, denominator down
            ξ[i] = setrounding(Float64, RoundUp) do
                (β^2 * norm_r_i) / ρ_i
            end
        end
    end

    return ξ
end

"""
    verify_generalized_eigenpairs(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector) -> GEVResult

Verify all eigenpairs of the generalized eigenvalue problem Ax = λBx.

Implements Algorithm 1 from Miyajima et al. (2010).

# Numeric Type Support
**Currently supports Float64 only.** All computations and error bounds are
performed using Float64 arithmetic with IEEE 754 double precision. The
implementation uses Float64-specific rounding error constants for rigorous
verification.

Extension to BigFloat would require:
- Parametric GEVResult{T} struct
- Type-dependent unit roundoff (eps(T))
- Modified error analysis for arbitrary precision

The mathematical algorithms (Theorems 4, 5, 7, 10) are precision-independent,
but this implementation is optimized for Float64 hardware arithmetic.

# Arguments
- `A::BallMatrix`: Symmetric interval matrix (n×n, Float64 elements)
- `B::BallMatrix`: Symmetric positive definite interval matrix (n×n, Float64 elements)
- `X̃::Matrix`: Approximate eigenvectors (n×n), typically from `eigen(A.c, B.c)`
- `λ̃::Vector`: Approximate eigenvalues (n), assumed sorted

# Returns
- `GEVResult` with verified eigenvalue intervals and eigenvector balls

# Algorithm (4 steps)
1. Compute β ≥ √‖B⁻¹‖₂ using Theorem 10
2. Compute global bound δ̂ and individual bounds ε using Theorems 4, 5
3. Determine separation bounds η using Lemma 2
4. Compute eigenvector bounds ξ using Theorem 7

# Complexity
O(12n³) dominated by matrix multiplications with interval arithmetic

# Verification Guarantees
When `success = true`:
- Each interval [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] contains exactly one true eigenvalue
- Each ball B(x̃⁽ⁱ⁾, ξᵢ) contains the normalized true eigenvector
- Results are rigorous for ALL matrices in the intervals [A] and [B]

# Example
```julia
using LinearAlgebra

A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))
B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))

F = eigen(Symmetric(A.c), Symmetric(B.c))
result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

if result.success
    println("Eigenvalue 1 ∈ ", result.eigenvalue_intervals[1])
    println("Eigenvalue 2 ∈ ", result.eigenvalue_intervals[2])
    println("Eigenvector radii: ", result.eigenvector_radii)
end
```

# References
Miyajima, S., Ogita, T., Rump, S. M., Oishi, S. (2010).
"Fast Verification for All Eigenpairs in Symmetric Positive Definite
Generalized Eigenvalue Problems". Reliable Computing 14, pp. 24-45.
"""
function verify_generalized_eigenpairs(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector)
    n = size(A, 1)

    # Input validation
    if size(A) != (n, n) || size(B) != (n, n)
        return GEVResult(false, [], X̃, [], NaN, NaN, [], [], NaN,
            "Matrix dimensions must be square and matching")
    end

    if size(X̃) != (n, n) || length(λ̃) != n
        return GEVResult(false, [], X̃, [], NaN, NaN, [], [], NaN,
            "Eigenvector matrix must be n×n and eigenvalue vector must have length n")
    end

    # Check symmetry (approximately)
    if norm(A.c - A.c', Inf) > 1e-10
        @warn "Matrix A does not appear to be symmetric"
    end
    if norm(B.c - B.c', Inf) > 1e-10
        @warn "Matrix B does not appear to be symmetric"
    end

    try
        # Step 1: Compute β using Theorem 10
        β = compute_beta_bound(B)

        if isinf(β) || isnan(β)
            return GEVResult(false, [], X̃, [], β, NaN, [], [], NaN,
                "Failed to compute β bound (B may not be positive definite)")
        end

        # Step 2: Compute global and individual bounds
        Rg = compute_residual_matrix(A, B, X̃, λ̃)
        Gg = compute_gram_matrix(B, X̃)

        # Residual norm for diagnostics
        # Rg is already a BallMatrix
        residual_norm = svd_bound_L2_opnorm(Rg)

        δ̂ = compute_global_eigenvalue_bound(Rg, Gg, β)

        if isinf(δ̂)
            return GEVResult(false, [], X̃, [], β, δ̂, [], [], residual_norm,
                "Global bound failed: approximate eigenvectors not sufficiently orthogonal (‖I - Gg‖₂ >= 1)")
        end

        ε = compute_individual_eigenvalue_bounds(A, B, X̃, λ̃, β)

        # Check for infinite individual bounds
        if any(isinf.(ε))
            @warn "Some individual eigenvalue bounds are infinite"
        end

        # Step 3: Determine η using Lemma 2
        η = compute_eigenvalue_separation(λ̃, δ̂, ε)

        # Check if all eigenvalues are separated
        if any(η .<= 0)
            return GEVResult(false, [], X̃, [], β, δ̂, ε, η, residual_norm,
                "Failed to separate all eigenvalues (some η ≤ 0)")
        end

        # Step 4: Compute eigenvector bounds using Theorem 7
        ξ = compute_eigenvector_bounds(A, B, X̃, λ̃, η, β)

        # Check for infinite eigenvector bounds
        if any(isinf.(ξ))
            @warn "Some eigenvector bounds are infinite"
        end

        # Construct eigenvalue intervals
        eigenvalue_intervals = [(λ̃[i] - η[i], λ̃[i] + η[i]) for i in 1:n]

        # Success!
        return GEVResult(true, eigenvalue_intervals, X̃, ξ,
            β, δ̂, ε, η, residual_norm,
            "All eigenpairs successfully verified")

    catch e
        return GEVResult(false, [], X̃, [], NaN, NaN, [], [], NaN,
            "Verification failed with error: $(e)")
    end
end

# ---------------------------------------------------------------------------
# Miyajima (2010), Theorem 1 — the two-residual generalized enclosure
# ---------------------------------------------------------------------------

"""
    MiyajimaGEVResult

Result of [`miyajima_gev_enclosure`](@ref).

# Fields
- `success::Bool`: whether `‖R₂‖_∞ < 1` held, i.e. whether the enclosure is proved
- `centers::Vector`: the approximate eigenvalues `λ̃`
- `radius`: a common radius valid for all eigenvalues (the smaller of Theorem 1's
  `‖R₁‖_∞/(1-‖R₂‖_∞)` and `maximum(radii)`)
- `radii`: the **per-eigenvalue** radii of Corollary 3.2; every eigenvalue of the
  pencil lies in `⋃ᵢ {z : |z - λ̃ᵢ| ≤ radii[i]}`, and `radii[i] ≤ radius` always
- `nrmR1`, `nrmR2`: the two certified residual norms `‖R₁‖_∞`, `‖R₂‖_∞`
- `message::String`
"""
struct MiyajimaGEVResult{T, CT}
    success::Bool
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
res.success && (res.radius)
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
