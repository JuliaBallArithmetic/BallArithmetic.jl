# Sylvester-Based Certified Resolvent Bounds

## Overview

This module implements **certified upper bounds on the resolvent norm** `‖(zI - T)⁻¹‖₂` for Schur-triangular matrices T. The method is:
- **Non-normal safe**: Works for any upper triangular T, not just normal matrices
- **Efficient**: O(k³) for small block + O(m²) for large block per evaluation point
- **Certified**: All bounds are rigorous upper bounds using interval/ball arithmetic

## Algorithm Summary

### Key Idea: Approximate Block-Diagonalization via Sylvester

Given T split as:
```
T = [T₁₁  T₁₂]
    [0    T₂₂]
```

1. Solve the Sylvester equation `T₁₁X - XT₂₂ = -T₁₂` to find X̃
2. Apply similarity: `S(X̃)⁻¹ T S(X̃) = [T₁₁  R; 0  T₂₂]` where R is the residual
3. Bound the resolvent of the almost block-diagonal matrix
4. Transfer back via `κ₂(S(X̃))`

### V1 Bound (Conservative)

```
‖(zI - T)⁻¹‖₂ ≤ K_S · (M_A + M_D + M_A · r · M_D)
```

where:
- `K_S = κ₂(S(X̃)) = ψ(‖X̃‖)²` — similarity penalty
- `M_A = ‖(zI - T₁₁)⁻¹‖₂` — small block inverse (Miyajima SVD)
- `M_D = ‖(zI - T₂₂)⁻¹‖₂` — large block inverse (triangular recursion)
- `r = ‖R‖₂` — Sylvester residual

### V2 Bound (Tighter)

```
‖(zI - T)⁻¹‖₂ ≤ K_S · (M_A + M_D + M_AR · M_D)
```

where `M_AR = ‖(zI - T₁₁)⁻¹R‖₂` is computed by:
1. Solve `Ŷ = (zI - T₁₁) \ R` (triangular solve)
2. Compute residual `Δ = R - (zI - T₁₁) · Ŷ`
3. Bound: `M_AR ≤ ‖Ŷ‖ + M_A · ‖Δ‖`

V2 is tighter when R is aligned with well-conditioned directions of A_z.

## Usage

### Basic Usage (V1)

```julia
using BallArithmetic

# Create Schur-triangular matrix
T = schur(A).T  # or any upper triangular matrix

# Split at index k (T₁₁ is k×k)
k = 5
z = 2.0 + 1.0im

# Compute bound
precomp, result = sylvester_resolvent_bound(T, k, z)

if result.success
    println("‖(zI - T)⁻¹‖₂ ≤ ", result.resolvent_bound)
end

# Print diagnostics
print_sylvester_diagnostics(precomp)
print_point_result(result)
```

### V2 (Tighter Bounds)

```julia
precomp, R, result = sylvester_resolvent_bound_v2(T, k, z)

print_point_result_v2(result)
# Shows both V1 and V2 bounds, and the improvement percentage
```

### Multiple Points

```julia
z_list = [1.0+0.5im, 2.0+1.0im, 3.0+0.5im]

# V1
precomp, results = sylvester_resolvent_bound(T, k, z_list)

# V2
precomp, R, results = sylvester_resolvent_bound_v2(T, k, z_list)
```

### Finding Optimal Split

```julia
best_k, best_precomp, best_result = find_optimal_split(T, z;
    k_range=2:20, version=:V1)

# For V2
best_k, best_precomp, best_R, best_result = find_optimal_split(T, z;
    k_range=2:20, version=:V2)
```

## Quality Indicators

The precomputation returns diagnostics:

- **`reduction = ‖R‖/‖T₁₂‖`**: Should be << 1 (Sylvester worked well)
  - ✓ excellent: < 0.1
  - ○ good: < 0.5
  - △ marginal: < 1.0
  - ✗ poor: ≥ 1.0

- **`penalty = κ₂(S(X̃))`**: Similarity condition number
  - ✓ excellent: < 2
  - ○ good: < 5
  - △ marginal: < 10
  - ✗ poor: ≥ 10

- **`net = penalty × reduction`**: Overall effectiveness
  - ✓ excellent: < 0.5
  - ○ good: < 1.0
  - △ marginal: < 2.0
  - ✗ poor: ≥ 2.0

## Failure Modes

The method correctly handles and reports failures:

1. **z at eigenvalue of T₁₁**: Miyajima SVD returns σ_min ≤ 0
2. **z at eigenvalue of T₂₂**: Triangular inverse bound → ∞
3. **Sylvester solve fails**: Singular or ill-conditioned equation
4. **Large K_S**: Poor similarity conditioning (try different split)

## Complexity

| Component | Cost | When |
|-----------|------|------|
| Sylvester solve | O(k²m + km²) | Once |
| Residual computation | O(k²m) | Once |
| Miyajima SVD | O(k³) | Per z |
| Triangular bounds | O(m²) | Per z |
| V2 triangular solve | O(k²m) | Per z |

Where k = small block size, m = n - k = large block size.

## BigFloat Support

```julia
setprecision(256) do
    T_bf = Complex{BigFloat}.(T)
    precomp, result = sylvester_resolvent_bound(T_bf, k, Complex{BigFloat}(z))
end
```

The Sylvester oracle uses Float64 internally but all certification is done in BigFloat.

## References

- Miyajima SVD: Miyajima (2014), "Verified bounds for all singular values"
- Rump-Oishi ψ function: Rump & Oishi (2024), "A Note on Oishi's Lower Bound..."
- Triangular inverse bounds: Standard backward recursion analysis
