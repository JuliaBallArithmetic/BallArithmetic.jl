# Linear Systems

This page documents verified linear system solvers.

## Overview

Verified linear system solvers provide rigorous enclosures for solutions of ``Ax = b``
with guaranteed error bounds. Multiple methods are available depending on the
matrix properties and desired tightness of bounds.

## Krawczyk Method

The Krawczyk method uses interval Newton iteration to verify and refine solutions.
See `src/linear_system/Krawczyk.jl` for the basic implementation.

## HBR Method

The Hansen-Bliek-Rohn enclosure method provides componentwise bounds.

- [`hbr_method`](@ref) - Main HBR solver
- [`hbr_method_simple`](@ref) - Simplified variant
- [`HBRResult`](@ref) - Result type

## Gaussian Elimination

Verified Gaussian elimination with interval pivoting.

- [`interval_gaussian_elimination`](@ref) - Verified Gaussian elimination
- [`interval_gaussian_elimination_det`](@ref) - With determinant computation
- [`is_regular_gaussian_elimination`](@ref) - Regularity test via GE
- [`GaussianEliminationResult`](@ref) - Result type

## Iterative Methods

Interval iterative methods for large sparse systems.

- [`interval_gauss_seidel`](@ref) - Interval Gauss-Seidel iteration
- [`interval_jacobi`](@ref) - Interval Jacobi iteration
- [`IterativeResult`](@ref) - Result type

## Preconditioning

Preconditioning strategies for improving convergence.

- [`compute_preconditioner`](@ref) - Compute preconditioner
- [`apply_preconditioner`](@ref) - Apply preconditioner to system
- [`is_well_preconditioned`](@ref) - Check preconditioning quality
- [`PreconditionerResult`](@ref) - Result type

Preconditioner types (see [`PreconditionerType`](@ref)):
- `MidpointInverse` - Inverse of midpoint matrix
- `LUFactorization` - LU-based preconditioner
- `LDLTFactorization` - LDLT-based preconditioner (symmetric)
- `IdentityPreconditioner` - No preconditioning

## Overdetermined Systems

Least squares solutions for overdetermined systems.

- [`subsquares_method`](@ref) - Subsquares method
- [`multi_jacobi_method`](@ref) - Multi-Jacobi method
- [`interval_least_squares`](@ref) - Interval least squares
- [`OverdeterminedResult`](@ref) - Result type

## Shaving

Interval shaving for tightening bounds.

- [`interval_shaving`](@ref) - Iterative shaving
- [`sherman_morrison_inverse_update`](@ref) - Efficient inverse updates
- [`ShavingResult`](@ref) - Result type

## H-Matrix Systems

Verified solvers for H-matrices (hierarchical matrices).

- [`verified_linear_solve_hmatrix`](@ref) - H-matrix linear solver
- [`VerifiedLinearSystemResult`](@ref) - Result type

## Matrix Regularity

Functions to verify matrix regularity (invertibility).

- [`is_regular`](@ref) - Main regularity test
- [`is_regular_sufficient_condition`](@ref) - Sufficient condition check
- [`is_regular_gershgorin`](@ref) - Gershgorin-based check
- [`is_regular_diagonal_dominance`](@ref) - Diagonal dominance check
- [`is_singular_sufficient_condition`](@ref) - Singularity test
- [`RegularityResult`](@ref) - Result type

## Determinant Bounds

Rigorous bounds on matrix determinants.

- [`interval_det`](@ref) - Interval determinant
- [`det_hadamard`](@ref) - Hadamard bound
- [`det_gershgorin`](@ref) - Gershgorin bound
- [`det_cramer`](@ref) - Cramer-based bound
- [`contains_zero`](@ref) - Check if determinant contains zero
- [`DeterminantResult`](@ref) - Result type
