# BigFloat Precision Fix for Ball Arithmetic

## Summary

This document describes the fixes implemented to enable proper BigFloat precision support in the BallArithmetic.jl package, specifically for SVD certification using the Ogita-Miyajima method.

## Problem

When using BigFloat matrices (e.g., 256-bit precision), the certified singular value radii were incorrectly at Float64 precision level (~1e-14) instead of BigFloat precision level (~1e-77). This was caused by two issues:

1. **Hardcoded Float64 epsilon**: The constant `ϵp = machine_epsilon(Float64)` was used throughout the codebase instead of type-parametric `machine_epsilon(T)`.

2. **Collatz bound underflow**: The Collatz power iteration for computing operator norms would underflow when applied to very small residual matrices, causing the ratio `x_new[i] / x_old[i]` to return 1.0 (stagnation) instead of the correct small value.

## Files Modified

### 1. `src/types/MMul/MMul3.jl`

Replaced hardcoded `ϵp` and `η` with type-parametric versions in all four `MMul3` method signatures:

```julia
# Before
rprimeB = ((k + 2) * ϵp * abs_mB + rB)
rC = abs_mA * rprimeB + rA * (abs_mB + rB) .+ η / ϵp

# After
ϵ = machine_epsilon(T)
η_val = subnormal_min(T)
rprimeB = ((k + 2) * ϵ * abs_mB + rB)
rC = abs_mA * rprimeB + rA * (abs_mB + rB) .+ η_val / ϵ
```

### 2. `src/types/MMul/MMul5.jl`

Similar fix for the `MMul5` function:

```julia
# Before
γ = (k + 1) * eps.(Γ) .+ 0.5 * η / ϵp

# After
ϵ = machine_epsilon(T)
η_val = subnormal_min(T)
γ = (k + 1) * eps.(Γ) .+ 0.5 * η_val / ϵ
```

Note: `eps.(Γ)` is correct and unchanged - it uses Julia's built-in `eps` which returns the local ULP for each element's magnitude.

### 3. `src/svd/svd.jl`

Changed from `collatz_upper_bound_L2_opnorm` to `upper_bound_L2_opnorm` in the certification code:

```julia
# Before
normE = collatz_upper_bound_L2_opnorm(E)
normF = collatz_upper_bound_L2_opnorm(F)
normG = collatz_upper_bound_L2_opnorm(G)

# After
normE = upper_bound_L2_opnorm(E)
normF = upper_bound_L2_opnorm(F)
normG = upper_bound_L2_opnorm(G)
```

The `upper_bound_L2_opnorm` function computes:
```julia
min(collatz_upper_bound_L2_opnorm(A), sqrt_up(norm1 * norminf))
```

This provides a fallback to the L1/L∞ interpolation bound when Collatz underflows.

## Root Cause Analysis

### Collatz Underflow Issue

When certifying SVD with BigFloat precision, the residual matrix E has entries ~1e-76. The Collatz power iteration computes:

```
x_{k+1} = |A|^T |A| x_k
```

With ||A|| ~ 1e-75, after a few iterations:
- iter 1: x ~ 1e-150
- iter 2: x ~ 1e-300
- iter 3: x underflows to 0

When both `x_old` and `x_new` contain zeros or stagnate at the same tiny value, the ratio `x_new[i] / x_old[i]` becomes 1.0, giving the useless bound ||E|| ≤ 1.

The L1/L∞ interpolation bound `sqrt(||A||_1 * ||A||_∞)` doesn't suffer from this issue because it uses simple column/row sums.

## Results

Before fix:
```
Certified σ_min: mid=0.0, rad=1.0
```

After fix:
```
Certified σ_min: mid=0.0, rad=5.7e-75
```

The certification radius is now at BigFloat precision level (~300 ulps), which is appropriate for a 10x10 matrix with multiple operations.

## Testing

The fix was verified with:
1. Unit tests for BallMatrix multiplication showing radii at correct precision
2. Full Ogita+Miyajima certification pipeline producing BigFloat-level radii
3. Comparison of L1/L∞ bounds vs Collatz bounds confirming the fallback works

## Notes

- The original `ϵp` constant is kept in `src/rounding/rounding.jl` for backwards compatibility
- The `eps(x)` function (Julia built-in) is different from `machine_epsilon(T)` and correctly scales with value magnitude
- BigFloat `setrounding` is respected by MPFR for basic operations and matrix multiplication
