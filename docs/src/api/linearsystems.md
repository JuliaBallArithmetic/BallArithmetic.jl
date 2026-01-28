# API - Linear Systems

Verified linear system solvers and related functions.

## Gaussian Elimination

```@docs
GaussianEliminationResult
interval_gaussian_elimination
interval_gaussian_elimination_det
is_regular_gaussian_elimination
```

## Iterative Methods

```@docs
IterativeResult
interval_gauss_seidel
interval_jacobi
```

## HBR Method

```@docs
HBRResult
hbr_method
hbr_method_simple
```

## Shaving

```@docs
ShavingResult
interval_shaving
sherman_morrison_inverse_update
```

## Preconditioning

```@docs
PreconditionerType
PreconditionerResult
compute_preconditioner
apply_preconditioner
is_well_preconditioned
```

## Overdetermined Systems

```@docs
OverdeterminedResult
subsquares_method
multi_jacobi_method
interval_least_squares
```

## H-Matrix Systems

```@docs
VerifiedLinearSystemResult
verified_linear_solve_hmatrix
```

## Sylvester Equations

```@docs
sylvester_miyajima_enclosure
triangular_sylvester_miyajima_enclosure
```

## Matrix Regularity

```@docs
RegularityResult
is_regular
is_regular_sufficient_condition
is_regular_gershgorin
is_regular_diagonal_dominance
is_singular_sufficient_condition
```

## Determinant Bounds

```@docs
DeterminantResult
interval_det
det_hadamard
det_gershgorin
det_cramer
contains_zero
```
