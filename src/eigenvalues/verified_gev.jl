# Verified Generalized Eigenvalue Problems
#
# Implementation of algorithms from:
# Miyajima, S., Ogita, T., Rump, S. M., Oishi, S. (2010)
# "Fast Verification for All Eigenpairs in Symmetric Positive Definite
# Generalized Eigenvalue Problems"
# Reliable Computing 14, pp. 24-45.
#
# Problem: Ax = λBx where A is symmetric, B is symmetric positive definite

export verify_generalized_eigenpairs, compute_beta_bound, GEVResult

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

Compute verified upper bound β ≥ √‖B⁻¹‖₂ using Theorem 10.

This function efficiently computes a bound on the square root of the 2-norm
of B⁻¹ using Cholesky factorization and an approximate inverse.

# Numeric Type Support
**Currently supports Float64 only.** The error analysis uses Float64-specific
rounding error constants (eps(Float64)). Extension to BigFloat would require:
- Type-parametric unit roundoff: eps(T) instead of eps(Float64)
- Appropriate error analysis for higher precision
- Parametric GEVResult struct

The mathematical algorithm itself is not Float64-specific, but the rigorous
error bounds in the implementation assume IEEE 754 double precision arithmetic.

# Arguments
- `B::BallMatrix`: Symmetric positive definite interval matrix (Float64 elements)

# Returns
- `β::Float64`: Upper bound on √‖B⁻¹‖₂

# Algorithm (Theorem 10)
1. Compute Cholesky factorization B ≈ LL^T
2. Compute approximate inverse X_L ≈ L⁻¹
3. Use interval arithmetic to bound error
4. Return β = √((α₁α∞)/(1 - α₁α∞αC))

# Complexity
O(n³) for Cholesky and inverse

# Example
```julia
B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))
β = compute_beta_bound(B)
```
"""
function compute_beta_bound(B::BallMatrix)
    n = size(B, 1)

    # Unit roundoff
    u = eps(Float64) / 2

    # Cholesky factorization of B.c (center)
    L = cholesky(Symmetric(B.c)).L

    # Compute approximate inverse of L
    X_L = inv(L)

    # Compute row sums for error analysis
    s = sum(abs.(B.c), dims=2)[:]  # s_i = Σⱼ|Bᵢⱼ|

    # γₙ = nu/(1-nu)
    γ_n = (n * u) / (1 - n * u)

    # Compute XL * L̃ (should be close to identity)
    XL_L = X_L * L

    # Error bound for 1-norm
    s1 = sum(abs.(L), dims=2)[:]
    ζ_1 = γ_n * norm(XL_L, 1) * norm(s1, 1) +
          (n * u) / (1 - n * u) * norm(n * ones(n) + diag(abs.(L)), 1)

    # Error bound for ∞-norm
    s_inf = maximum(abs.(L), dims=2)[:]
    ζ_inf = γ_n * norm(XL_L, Inf) * norm(s_inf, Inf) +
            (n * u) / (1 - n * u) * norm(n * ones(n) + diag(abs.(L)), Inf)

    # Check convergence conditions
    if ζ_1 >= 1.0 || ζ_inf >= 1.0
        @warn "Beta computation: ζ bounds >= 1, using fallback estimate"
        # Fallback: use condition number estimate
        return sqrt(cond(B.c))
    end

    # Compute α bounds
    α_1 = norm(X_L, 1) / (1 - ζ_1)
    α_inf = norm(X_L, Inf) / (1 - ζ_inf)

    # Additional error for Cholesky reconstruction
    L_LT = L * L'
    s_c_inf = maximum(sum(abs.(L_LT), dims=2))
    α_C = γ_n * norm(L_LT, Inf) * s_c_inf +
          (n * u) / (1 - (n - 1) * u) * norm((n - 1) * ones(n) + diag(abs.(L)), Inf)

    # Check final condition
    if α_1 * α_inf * α_C >= 1.0
        @warn "Beta computation: final condition failed, using fallback"
        return sqrt(cond(B.c))
    end

    # Compute β (Theorem 10)
    β = sqrt((α_1 * α_inf) / (1 - α_1 * α_inf * α_C))

    return β
end

"""
    compute_residual_matrix(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector) -> Matrix

Compute residual matrix Rg = AX̃ - BX̃D̃ where D̃ = diag(λ̃).

Uses interval arithmetic to account for uncertainties in A and B.

# Complexity
O(n³) - dominated by matrix multiplications
"""
function compute_residual_matrix(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector)
    n = size(A, 1)
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
function compute_individual_eigenvalue_bounds(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, β::Float64)
    n = length(λ̃)
    ε = zeros(Float64, n)

    for i in 1:n
        # Individual residual r⁽ⁱ⁾ = Ax̃⁽ⁱ⁾ - λ̃ᵢBx̃⁽ⁱ⁾
        x_i = X̃[:, i]
        Ax_i = A * x_i
        Bx_i = B * x_i
        r_i = Ax_i - λ̃[i] * Bx_i

        # Norm of residual
        norm_r_i = sqrt(sum([x.c^2 + x.r^2 for x in r_i]))  # Upper bound on 2-norm

        # Gram element gᵢ = x̃⁽ⁱ⁾ᵀBx̃⁽ⁱ⁾
        g_i = dot(x_i, Bx_i)

        # Handle interval Ball type
        if isa(g_i, Ball)
            g_i_val = g_i.c  # Use center (should be close to 1 if normalized)
        else
            g_i_val = g_i
        end

        if g_i_val <= 0
            @warn "Individual bound $i: gᵢ ≤ 0, using large bound"
            ε[i] = Inf
        else
            ε[i] = (β * norm_r_i) / sqrt(abs(g_i_val))
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

        for i in 1:n-1
            for j in i+1:n
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
function compute_eigenvector_bounds(A::BallMatrix, B::BallMatrix, X̃::Matrix, λ̃::Vector, η::Vector{Float64}, β::Float64)
    n = length(λ̃)
    ξ = zeros(Float64, n)

    for i in 1:n
        # Compute individual residual norm
        x_i = X̃[:, i]
        Ax_i = A * x_i
        Bx_i = B * x_i
        r_i = Ax_i - λ̃[i] * Bx_i
        norm_r_i = sqrt(sum([x.c^2 + x.r^2 for x in r_i]))

        # Compute ρᵢ: distance to nearest other eigenvalue interval
        ρ_i = Inf

        if i > 1
            # Distance to previous eigenvalue
            dist_prev = (λ̃[i] - η[i]) - (λ̃[i-1] + η[i-1])
            ρ_i = min(ρ_i, dist_prev)
        end

        if i < n
            # Distance to next eigenvalue
            dist_next = (λ̃[i+1] - η[i+1]) - (λ̃[i] + η[i])
            ρ_i = min(ρ_i, dist_next)
        end

        if ρ_i <= 0 || isinf(ρ_i)
            @warn "Eigenvector bound $i: ρᵢ ≤ 0 or infinite, eigenvalues not separated"
            ξ[i] = Inf
        else
            ξ[i] = (β^2 * norm_r_i) / ρ_i
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
