# API - Norm Bounds and Matrix Classes

The rigorous norm and singular value bounds, and the tests that classify a matrix.

The matrix of entrywise moduli that every bound here is computed from, `upper_abs`, and the
bounds on the entrywise, ℓ¹, ℓ^∞ and spectral norms of a whole ball matrix are on the Core Types
page. This page holds the Frobenius bound, the bounds for the norm of an inverse and of a
resolvent, the rounding constant of an inner product, the steps of the Schur complement bounds
of Oishi and of Rump and Oishi, and the two matrices the M-matrix and H-matrix tests are built
from.

## The Frobenius Norm

```@docs
upper_bound_frobenius
```

## The Inverse of a Triangular Matrix

Operator norms of `U⁻¹`, bounded by a recursion on the entries of `U` and without forming the
inverse.

```@docs
triangular_inverse_inf_norm_bound
triangular_inverse_one_norm_bound
```

## The Resolvent

```@docs
BallArithmetic.svd_bound_L2_resolvent
```

## A Block Triangular Similarity

```@docs
similarity_condition_number
```

## The Rounding of an Inner Product

```@docs
gamma_bound
```

## Steps of the Schur Complement Bounds

The quantities that `oishi_2023_sigma_min_bound` and `rump_oishi_2024_sigma_min_bound` form from
the blocks of `G`, each reached through those routines rather than called directly.

```@docs
BallArithmetic._fast_gamma_bound
BallArithmetic._inverse_diagonal_norm
BallArithmetic._diagonal_scale_left
BallArithmetic._solve_ball_system
BallArithmetic._solve_ball_system_transpose
```

## Matrix Classes

The two matrices read by `BallArithmetic.is_M_matrix` and `BallArithmetic.is_H_matrix`.

```@docs
BallArithmetic.off_diagonal_abs
BallArithmetic.diagonal_abs_lower_bound
```

## The Rounding Mode of the Array Operations

```@docs
BallArithmetic.NumericalTest.array_rounding_test
```
