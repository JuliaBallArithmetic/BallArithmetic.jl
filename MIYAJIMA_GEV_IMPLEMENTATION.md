# Verified Generalized Eigenvalue Problems - Implementation Plan

## Overview

This document outlines the implementation of verified methods for generalized eigenvalue problems based on:

**Miyajima, S., Ogita, T., Rump, S. M., Oishi, S. (2010)**
"Fast Verification for All Eigenpairs in Symmetric Positive Definite Generalized Eigenvalue Problems"
*Reliable Computing* 14, pp. 24-45.

## Problem Statement

Compute verified enclosures for all eigenpairs (λᵢ, xᵢ) of the generalized eigenvalue problem:

```
Ax = λBx
```

where:
- A is symmetric (A = Aᵀ)
- B is symmetric positive definite (B = Bᵀ, B > 0)
- Both A and B are n×n matrices

## Mathematical Background

### Cholesky-QR Approach

The method transforms the generalized problem to a standard eigenvalue problem:

1. Compute Cholesky factorization: B = LLᵀ
2. Transform: L⁻¹AL⁻ᵀy = λy
3. Compute QR factorization of L⁻¹A: L⁻¹A = QR
4. Standard problem: RRᵀy = λy (symmetric positive definite)
5. Recover: x = L⁻ᵀy

### Key Quantities

Given approximate eigenpairs (λ̃ᵢ, x̃ᵢ):

**Residual Matrix:**
```
Rg = AX̃ - BX̃D̃
where X̃ = [x̃₁, ..., x̃ₙ], D̃ = diag(λ̃₁, ..., λ̃ₙ)
```

**Gram Matrix:**
```
Gg = X̃ᵀBX̃
```

**Preconditioning Factor:**
```
β ≥ √‖B⁻¹‖₂
```

**Individual Residuals:**
```
r⁽ⁱ⁾ = Ax̃⁽ⁱ⁾ - λ̃ᵢBx̃⁽ⁱ⁾
gᵢ = x̃⁽ⁱ⁾ᵀBx̃⁽ⁱ⁾
```

## Core Theorems to Implement

### Theorem 4: Global Eigenvalue Bounds

**Input:** Approximate eigenvalues λ̃₁ ≤ ... ≤ λ̃ₙ
**Output:** Verified bound δ̂

```julia
δ̂ = (β * ‖Rg‖₂) / (1 - ‖I - Gg‖₂)
```

**Guarantee:** Each true eigenvalue λⱼ satisfies:
```
|λⱼ - λ̃ⱼ| ≤ δ̂
```

**Conditions:**
- ‖I - Gg‖₂ < 1 (approximate eigenvectors nearly orthogonal w.r.t. B)
- β ≥ √‖B⁻¹‖₂

### Theorem 5: Individual Eigenvalue Bounds

**Input:** Approximate eigenpair (λ̃ᵢ, x̃⁽ⁱ⁾)
**Output:** Verified bound εᵢ

```julia
εᵢ = (β * ‖r⁽ⁱ⁾‖₂) / √gᵢ
```

**Guarantee:** At least one true eigenvalue λⱼ satisfies:
```
|λⱼ - λ̃ᵢ| ≤ εᵢ
```

### Lemma 2: Eigenvalue Separation

Given bounds δ̂ and ε = (ε₁, ..., εₙ)ᵀ, find the largest ηᵢ such that:

```
[λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] ∩ [λ̃ⱼ - ηⱼ, λ̃ⱼ + ηⱼ] = ∅  for all j ≠ i
```

with ηᵢ ≤ min(δ̂, εᵢ).

**Algorithm:**
```julia
function compute_eta(λ̃, δ̂, ε)
    η = min.(δ̂, ε)
    changed = true
    while changed
        changed = false
        for i in 1:n
            for j in i+1:n
                if λ̃[i] + η[i] + η[j] > λ̃[j]
                    # Intervals overlap, shrink both
                    gap = (λ̃[j] - λ̃[i]) / 2
                    if η[i] > gap || η[j] > gap
                        η[i] = min(η[i], gap)
                        η[j] = min(η[j], gap)
                        changed = true
                    end
                end
            end
        end
    end
    return η
end
```

### Theorem 7: Eigenvector Bounds

**Input:** Verified eigenvalue interval [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] containing exactly one eigenvalue
**Output:** Eigenvector bound ξᵢ

```julia
ρᵢ = min(λ̃[i] - η[i] - λ̃[i-1] - η[i-1],  # distance to previous eigenvalue
         λ̃[i+1] - η[i+1] - λ̃[i] - η[i])  # distance to next eigenvalue

ξ̂ᵢ = (β * ‖r⁽ⁱ⁾‖₂) / ρᵢ

ξᵢ = β * ξ̂ᵢ = β² * ‖r⁽ⁱ⁾‖₂ / ρᵢ
```

**Guarantee:** If x̂⁽ⁱ⁾ is the true eigenvector:
```
‖x̂⁽ⁱ⁾ - x̃⁽ⁱ⁾‖₂ ≤ ξᵢ
```

### Theorem 10: Fast β Computation

Efficiently compute β ≥ √‖B⁻¹‖₂ using interval arithmetic.

**Method:** Use approximate inverse X_L and Cholesky factor L̃:

```julia
# Compute error bounds
ζ₁ = γₙ * ‖X_L L̃‖₁ * s₁ + (n*u)/(1-n*u) * ‖n*s + diag(|L̃|)‖₁
ζ∞ = γₙ * ‖X_L L̃‖∞ * s∞ + (n*u)/(1-n*u) * ‖n*s + diag(|L̃|)‖∞

# Compute norms with errors
α₁ = ‖X_L‖₁ / (1 - ζ₁)
α∞ = ‖X_L‖∞ / (1 - ζ∞)

# Additional Cholesky error
αC = γₙ * ‖L̃ L̃ᵀ‖s∞ + (n*u)/(1-(n-1)*u) * ‖(n-1)*s + diag(|L̃|)‖∞

# Final bound
if α₁ * α∞ * αC < 1
    β = √((α₁ * α∞) / (1 - α₁ * α∞ * αC))
end
```

where:
- γₙ = n*u/(1-n*u) is the rounding error constant
- u = eps(Float64)/2 is the unit roundoff
- s = sum(|B|, dims=2) is the row sum vector

## Complete Algorithm

### Algorithm 1: Verify All Eigenpairs

**Input:**
- A, B: interval matrices (centers with radii)
- X̃, D̃: approximate eigenpairs from floating-point solver

**Output:**
- Verified intervals [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] for eigenvalues
- Verified balls B(x̃⁽ⁱ⁾, ξᵢ) for eigenvectors

**Steps:**

```julia
function verify_generalized_eigenpairs(A::BallMatrix, B::BallMatrix,
                                       X̃::Matrix, λ̃::Vector)
    # Step 1: Compute β using Theorem 10
    β = compute_beta_bound(B)

    # Step 2: Compute global and individual bounds
    Rg = compute_residual_matrix(A, B, X̃, λ̃)
    Gg = compute_gram_matrix(B, X̃)

    δ̂ = (β * norm(Rg, 2)) / (1 - norm(I - Gg, 2))

    ε = zeros(n)
    for i in 1:n
        r_i = A * X̃[:, i] - λ̃[i] * B * X̃[:, i]
        g_i = X̃[:, i]' * B * X̃[:, i]
        ε[i] = (β * norm(r_i, 2)) / sqrt(g_i)
    end

    # Step 3: Determine η using Lemma 2
    η = compute_eta(λ̃, δ̂, ε)

    # Step 4: Compute eigenvector bounds using Theorem 7
    ξ = zeros(n)
    for i in 1:n
        ρ_i = compute_eigenvalue_separation(λ̃, η, i)
        r_i = A * X̃[:, i] - λ̃[i] * B * X̃[:, i]
        ξ[i] = β^2 * norm(r_i, 2) / ρ_i
    end

    return (eigenvalue_intervals = [(λ̃[i] - η[i], λ̃[i] + η[i]) for i in 1:n],
            eigenvector_centers = X̃,
            eigenvector_radii = ξ)
end
```

## Computational Efficiency

### Overall Cost: 10n³ flops

**Matrix multiplications (interval arithmetic):**
1. fl△(AX̃) - upper bound: n³ flops
2. fl▽(AX̃) - lower bound: n³ flops
3. fl△(BX̃) - upper bound: n³ flops
4. fl▽(BX̃) - lower bound: n³ flops
5. fl✷(X̃ᵀZc) where Zc = (AX̃)c or (BX̃)c: 6n³ flops

**Total:** 10n³ flops (vs. 44n³ for previous methods)

### Optimization Techniques

**Technique 1: Reuse BX̃**
- Compute BX̃ once for both Rg and Gg
- Saves n³ flops

**Technique 2: Fast ‖I - Gg‖∞ in O(n²)**
```julia
Z = B * X̃  # already computed
Zc = mid(Z)
Zr = rad(Z)
g_infinity = norm(fl✷(I - X̃ᵀ * Zc), Inf) +
             norm(abs.(X̃ᵀ) * Zr * ones(n), Inf) +
             γₙ * (norm(abs.(X̃ᵀ) * abs.(Zc) * ones(n), Inf) + 1) + n*u
```

**Technique 3: Reuse Rg columns for εᵢ**
- Extract r⁽ⁱ⁾ from i-th column of Rg
- Extract gᵢ from i-th diagonal of Gg
- Total O(n²) instead of n × O(n²)

**Technique 4: Reuse for eigenvector verification**
- τᵢ = β‖r⁽ⁱ⁾‖₂ already computed in Step 2
- μᵢ = √gᵢ already computed in Step 2
- Only need to compute ρᵢ (O(n) per vector)

## Implementation Files

### Primary Implementation

**`src/eigenvalues/verified_gev.jl`**

Main functions:
```julia
# High-level interface
verify_generalized_eigenpairs(A, B, X̃, λ̃) -> GEVResult

# Core verification components
compute_beta_bound(B) -> Float64                    # Theorem 10
compute_global_eigenvalue_bound(A, B, X̃, λ̃, β) -> Float64  # Theorem 4
compute_individual_eigenvalue_bounds(A, B, X̃, λ̃, β) -> Vector{Float64}  # Theorem 5
compute_eigenvalue_separation(λ̃, δ̂, ε) -> Vector{Float64}  # Lemma 2
compute_eigenvector_bounds(A, B, X̃, λ̃, η, β) -> Vector{Float64}  # Theorem 7

# Helper functions
compute_residual_matrix(A, B, X̃, D̃) -> BallMatrix
compute_gram_matrix(B, X̃) -> BallMatrix
```

Result structure:
```julia
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

    # Computational info
    iterations::Int
    residual_norm::Float64
end
```

### Testing

**`test/test_eigenvalues/test_verified_gev.jl`**

Test cases:
1. Small matrices (2×2, 3×3) with known eigenvalues
2. Diagonal matrices (easy case)
3. Nearly singular B (challenging case)
4. Clustered eigenvalues (test Lemma 2 separation logic)
5. Large matrices (n=100, 500) for performance
6. Comparison with standard eigenvalue solver
7. Perturbation tests (interval matrices with large radii)

## Usage Examples

### Example 1: Basic Usage

```julia
using BallArithmetic, LinearAlgebra

# Define matrices (intervals with small radii)
A = BallMatrix([4.0 1.0; 1.0 3.0], fill(1e-10, 2, 2))  # symmetric
B = BallMatrix([2.0 0.5; 0.5 2.0], fill(1e-10, 2, 2))  # SPD

# Get approximate solution using floating-point
F = eigen(Symmetric(A.c), Symmetric(B.c))
X̃ = F.vectors
λ̃ = F.values

# Verify with interval arithmetic
result = verify_generalized_eigenpairs(A, B, X̃, λ̃)

# Check results
if result.success
    println("All eigenpairs verified!")
    for i in 1:length(λ̃)
        println("λ$i ∈ ", result.eigenvalue_intervals[i])
        println("‖x̂$i - x̃$i‖ ≤ ", result.eigenvector_radii[i])
    end
else
    println("Verification failed: ", result.message)
end
```

### Example 2: Interval Matrices with Uncertainties

```julia
# Matrices with measurement uncertainties
A_center = [4.0 1.0 0.5;
            1.0 3.0 0.2;
            0.5 0.2 5.0]
A_radius = fill(0.01, 3, 3)  # 1% uncertainty
A = BallMatrix(A_center, A_radius)

B_center = [2.0 0.0 0.0;
            0.0 2.0 0.0;
            0.0 0.0 2.0]
B_radius = fill(0.001, 3, 3)  # 0.1% uncertainty
B = BallMatrix(B_center, B_radius)

# Approximate solution
F = eigen(Symmetric(A.c), Symmetric(B.c))

# Verify
result = verify_generalized_eigenpairs(A, B, F.vectors, F.values)

# The result accounts for all possible matrices in the intervals
```

### Example 3: High-Accuracy Requirements

```julia
# Use higher precision for initial approximation
A = BallMatrix(A_center, A_radius)
B = BallMatrix(B_center, B_radius)

# Get better approximate solution
using GenericLinearAlgebra
F = eigen(Symmetric(BigFloat.(A.c)), Symmetric(BigFloat.(B.c)))
X̃ = Float64.(F.vectors)
λ̃ = Float64.(F.values)

# Verify (will get tighter bounds)
result = verify_generalized_eigenpairs(A, B, X̃, λ̃)
```

### Example 4: Diagnostic Information

```julia
result = verify_generalized_eigenpairs(A, B, X̃, λ̃)

println("β = ", result.beta)
println("Global bound δ̂ = ", result.global_bound)
println("Individual bounds ε:")
for i in 1:length(result.individual_bounds)
    println("  ε[$i] = ", result.individual_bounds[i])
end
println("Separation bounds η:")
for i in 1:length(result.separation_bounds)
    println("  η[$i] = ", result.separation_bounds[i])
end
println("Residual norm: ", result.residual_norm)
```

## Integration with Existing Code

### Related Functions

The new GEV verification complements existing eigenvalue methods:

**Standard eigenvalue problems:**
- Existing methods in `src/eigenvalues/` (if any)
- Can use `verify_generalized_eigenpairs(A, I, X̃, λ̃)` with B = I

**SVD verification:**
- Related to `rigorous_svd()` and `_certify_svd()`
- Uses similar interval arithmetic techniques
- GEV is more specialized (symmetric, SPD)

**Linear systems:**
- Horáček methods in `src/linear_system/`
- GEV uses similar preconditioning ideas (β bound)
- Can use verified linear solves for computing β

### Module Structure

```julia
# In src/BallArithmetic.jl
include("eigenvalues/verified_gev.jl")

export verify_generalized_eigenpairs,
       compute_beta_bound,
       GEVResult
```

## Mathematical Guarantees

### What We Prove

1. **Eigenvalue containment:** Each interval [λ̃ᵢ - ηᵢ, λ̃ᵢ + ηᵢ] contains exactly one true eigenvalue

2. **Eigenvector proximity:** Each ball B(x̃⁽ⁱ⁾, ξᵢ) contains the normalized true eigenvector

3. **Accounting for uncertainties:** Results hold for ALL matrices in the interval matrices [A] and [B]

### Failure Modes

The verification can fail if:
1. ‖I - Gg‖₂ ≥ 1 (approximate eigenvectors not nearly orthogonal w.r.t. B)
2. Eigenvalues too clustered (Lemma 2 cannot separate them)
3. Matrix radii too large (interval overestimation)
4. Floating-point approximation too poor

In failure cases, the function returns `success = false` with diagnostic information.

## Performance Characteristics

### Complexity

| Operation | Cost | Notes |
|-----------|------|-------|
| β computation (Theorem 10) | O(n³) | Cholesky + approximate inverse |
| Matrix multiplications | 10n³ | Interval arithmetic, 5 products |
| Global bound (Theorem 4) | O(n³) | Dominated by matrix norms |
| Individual bounds (Theorem 5) | O(n²) | Using Technique 3 |
| Separation (Lemma 2) | O(n²) | Iterative interval shrinking |
| Eigenvector bounds (Theorem 7) | O(n²) | Using Technique 4 |
| **Total** | **~12n³** | Dominated by β and matrix products |

### Comparison with Previous Method

**Rump 1999 method:** ~44n³ flops
**This method:** ~12n³ flops
**Speedup:** ~3.7×

### Scaling

Based on numerical examples in the paper:

| Matrix Size | Time (this method) | Time (Rump 1999) | Speedup |
|-------------|-------------------|------------------|---------|
| n = 50 | 0.01s | 0.03s | 3.0× |
| n = 100 | 0.05s | 0.15s | 3.0× |
| n = 500 | 5.0s | 17.0s | 3.4× |
| n = 1000 | 40s | 140s | 3.5× |
| n = 2500 | 600s | 2200s | 3.7× |

(Approximate times on 2010-era hardware; modern systems will be faster)

## Extensions

### Quadratic Eigenvalue Problems

The paper also discusses extension to quadratic eigenvalue problems:
```
(λ²A + λB + C)x = 0
```

This can be linearized to a generalized problem:
```
[0  I] [x ]     [-C  0] [x ]
[-I 0] [λx] = λ [-B -A] [λx]
```

Then apply the verification algorithm. This extension could be implemented as:

```julia
function verify_quadratic_eigenpairs(A, B, C, X̃, λ̃)
    # Linearize to generalized form
    n = size(A, 1)
    A_lin = [zeros(n,n) I; -I zeros(n,n)]
    B_lin = [-C zeros(n,n); -B -A]

    # Extend eigenvectors
    X̃_lin = [X̃; λ̃' .* X̃]

    # Verify
    return verify_generalized_eigenpairs(A_lin, B_lin, X̃_lin, λ̃)
end
```

### Indefinite B Matrices

The current method requires B to be positive definite. For indefinite B:
1. Could use QZ factorization instead of Cholesky-QR
2. Would need different theoretical bounds
3. More complex implementation

This is **not** covered in the paper and would require additional research.

### Clustered Eigenvalues

When eigenvalues are very close, Lemma 2 may produce very small ηᵢ or fail to separate them. Possible improvements:
1. Use block verification for clusters (verify the cluster as a whole)
2. Refine approximation with higher precision
3. Report clusters instead of individual eigenvalues

## Implementation Priority

### Phase 1: Core Implementation (High Priority)
- [x] Read and understand paper
- [ ] Implement Theorem 10 (β computation)
- [ ] Implement Theorem 4 (global bounds)
- [ ] Implement Theorem 5 (individual bounds)
- [ ] Implement Lemma 2 (separation)
- [ ] Implement Theorem 7 (eigenvector bounds)
- [ ] Implement Algorithm 1 (main function)
- [ ] Basic test suite

### Phase 2: Optimization (Medium Priority)
- [ ] Implement Technique 1 (reuse BX̃)
- [ ] Implement Technique 2 (fast ‖I - Gg‖∞)
- [ ] Implement Technique 3 (reuse Rg columns)
- [ ] Implement Technique 4 (reuse for eigenvectors)
- [ ] Performance benchmarks

### Phase 3: Extensions (Low Priority)
- [ ] Quadratic eigenvalue problems
- [ ] Block verification for clusters
- [ ] Adaptive precision
- [ ] Parallel computation

## References

1. **Primary:** Miyajima, S., Ogita, T., Rump, S. M., Oishi, S. (2010). "Fast Verification for All Eigenpairs in Symmetric Positive Definite Generalized Eigenvalue Problems". *Reliable Computing* 14, pp. 24-45.

2. **Related Methods:**
   - Rump, S.M. (1999). "Fast and parallel interval arithmetic". *BIT Numerical Mathematics* 39, 539-560.
   - Rump, S.M. (2001). "Computational error bounds for multiple or nearly multiple eigenvalues". *Linear Algebra and its Applications* 324, 209-226.

3. **Theoretical Background:**
   - Parlett, B.N. (1980). "The Symmetric Eigenvalue Problem". Prentice-Hall.
   - Wilkinson, J.H. (1965). "The Algebraic Eigenvalue Problem". Clarendon Press.

4. **Implementation Reference:**
   - INTLAB toolbox by Rump: http://www.ti3.tu-harburg.de/rump/intlab/

## Files to Create/Modify

### New Files
1. `src/eigenvalues/verified_gev.jl` - Main implementation
2. `test/test_eigenvalues/test_verified_gev.jl` - Test suite
3. `MIYAJIMA_GEV_IMPLEMENTATION.md` - This documentation (DONE)

### Modified Files
1. `src/BallArithmetic.jl` - Add exports
2. `test/runtests.jl` - Include new tests (if not automatic)
3. `README.md` - Add documentation reference (optional)

---

**Status:** Implementation plan complete, ready to begin Phase 1.
**Date:** 2026-01-26
**Next Step:** Implement core functions starting with Theorem 10 (β computation).
