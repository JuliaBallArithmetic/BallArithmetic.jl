# API - Linear Systems

Verified linear system solvers and related functions.

## Triangular Systems

```@docs
BallArithmetic.backward_substitution
BallArithmetic.forward_substitution
```

## Cholesky Factorization

```@docs
verified_cholesky
VerifiedCholeskyResult
```

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
interval_initial_enclosure
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
BallArithmetic._shaving_lp
```

## Preconditioning

```@docs
PreconditionerType
MidpointInverse
LUFactorization
LDLTFactorization
IdentityPreconditioner
PreconditionerResult
compute_preconditioner
apply_preconditioner
is_well_preconditioned
```

## Overdetermined Systems

```@docs
OverdeterminedResult
select_subsquares
subsquares_method
multi_jacobi_method
interval_least_squares
```

## H-Matrix Systems

```@docs
VerifiedLinearSystemResult
verified_linear_solve_hmatrix
BallArithmetic.comparison_matrix
BallArithmetic.is_H_matrix
```

## Krawczyk Operator

```@docs
krawczyk_linear_system
KrawczykResult
```

## Epsilon Inflation

```@docs
verifylss
VerifyLssResult
```

## Sylvester Equations

```@docs
sylvester_miyajima_enclosure
verified_sylvester_enclosure
triangular_sylvester_miyajima_enclosure
SylvesterResolventResult
parametric_resolvent_bound
ParametricResolventResult
```

### The Approximate Solution and the Triangular Route

The approximate solution that the enclosures take as their midpoint, computed in floating point
with no bound attached, and the enclosure used for a triangular equation when the hypotheses of
Miyajima's theorems are not proved.

```@docs
schur_sylvester_midpoint
BallArithmetic._sylvester_triangular_columns
BallArithmetic._forward_substitution
BallArithmetic._sylvester_triangular_direct_ball
BallArithmetic._has_distinct_diagonal
```

### How the Two Inner Bounds Are Taken

```@docs
NormEstimator
OneInfNorm
FrobeniusNorm
DInverseEstimator
TriBacksub
NeumannOneInf
NeumannCollatz2
CouplingEstimator
CouplingNone
CouplingARSolve
CouplingOffDirect
```

## Matrix Regularity

```@docs
RegularityResult
is_regular
is_regular_sufficient_condition
is_regular_gershgorin
is_regular_diagonal_dominance
is_regular_spectral_radius
is_singular_sufficient_condition
```

## Determinant Bounds

```@docs
DeterminantResult
interval_det
det_hadamard
det_gershgorin
det_cramer
det_cofactor
contains_zero
```
