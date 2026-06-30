# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

BallArithmetic.jl does **rigorous numerical linear algebra** using midpoint–radius (ball) arithmetic. Every operation guarantees the true mathematical result lies inside the returned ball. The core type is `Ball{T, NT}` (radius type `T`, center type `NT`, real or complex); `BallMatrix`/`BallVector` store centers and radii as *separate* dense arrays so the heavy lifting can still go through BLAS. Most algorithms implement papers by Rump, Miyajima, Ogita, Oishi and collaborators (see `references/` and `CITATION.bib`).

## Commands

```bash
# Run the full test suite (use --project, NOT --project=test; the main project pulls in test deps)
julia --project -e 'using Pkg; Pkg.test()'

# Run a single test file directly — much faster while iterating
julia --project test/test_eigenvalues/test_ordschur_ball.jl

# Format code (BlueStyle, see .JuliaFormatter.toml)
julia -e 'using JuliaFormatter; format(".")'

# Build docs locally
julia --project=docs docs/make.jl
```

When adding a test dependency, add it to `test/Project.toml` **and** run
`julia --project=test -e 'using Pkg; Pkg.resolve()'`. Note `test/` is a full separate environment (`Project.toml` + `Manifest.toml`), not the `[extras]`/`[targets]` mechanism in the root `Project.toml`.

CI (`.github/workflows/CI.yml`) runs on **ubuntu-latest only** (macOS/Windows were dropped due to BLAS floating-point mismatches), on Julia `1` and `nightly`.

## Rigor depends on rounding-mode control (read this before touching arithmetic)

The whole package is only correct because floating-point operations use IEEE directed rounding. Two mechanisms:

- **Float64 BLAS path**: `__init__` in `src/BallArithmetic.jl` swaps in `OpenBLASConsistentFPCSR_jll` (`ConsistentFPCSR=1`) so the rounding mode is consistent across *all* OpenBLAS threads, then runs `NumericalTest.rounding_test`. On non-x86_64 architectures it falls back to single-threaded BLAS. **Do not assume a normal BLAS gives rigorous results.**
- **Scalar / BigFloat path**: `src/rounding/rounding.jl` defines `add_up`/`add_down`/`mul_up`/… (extending `RoundingEmulator` to BigFloat and Complex) plus the `@up` / `@down` macros, which rewrite an arithmetic expression so every `+ - * /` rounds outward. Use these — never bare arithmetic — when computing a radius or any guaranteed bound. Radii are inflated using `machine_epsilon(T)` and kept strictly positive with `subnormal_min(T)`.

Matrix multiplication has several variants (`MMul3/4/5`, Oishi–Rump complex product) trading precision against speed; see `src/types/matrix.jl` and `test/test_types/test_MMul.jl`.

## Code map

`src/BallArithmetic.jl` is the single include/export manifest — read it first to locate anything. Major directories:

- `types/` — `Ball`, `BallArray`, `BallMatrix`, `BallVector`, conversion/promotion, triangularize
- `rounding/` — directed-rounding primitives and macros (above)
- `norm_bounds/` — rigorous L1/L2/L∞ operator-norm bounds, Oishi & Rump–Oishi 2024 σ_min bounds, Collatz iteration
- `eigenvalues/` — multiple verified eigenvalue algorithms (`miyajima/`, `rump_2022a`, `rump_lange_2023`, `verified_gev`), Schur refinement (`iterative_schur_refinement`, `ordschur_ball`), spectral/Riesz projectors, Newton–Kantorovich eigenpair certification
- `decompositions/` — verified LU/Cholesky/QR/polar/Takagi, rigorous residual helpers, iterative refinement; `decompositions/svd/` holds rigorous SVD, Miyajima VBD, NJD, adaptive Ogita, precision-cascade SVD
- `linear_system/` — Krawczyk, HBR, Gaussian elimination, shaving, inflation, Sylvester, preconditioning, overdetermined/least-squares, H-matrix solver
- `matrix_classifiers/` & `matrix_properties/` — M-matrix test, regularity, determinant enclosures
- `pseudospectra/` — rigorous resolvent/contour certification; `CertifScripts.jl` (serial + distributed)
- `numerical_test/` — the startup rounding self-test

`test/` mirrors `src/` directory-for-directory; `test/runtests.jl` is the index of every test file.

### Extensions (`ext/`, weakdeps in `Project.toml`)

Loaded automatically when the trigger package is present: `IntervalArithmeticExt` (Interval↔Ball), `ArbNumericsExt`, `FFTExt` (rigorous FFT), `DistributedExt` (parallel pseudospectrum certification), `DoubleFloatsExt`, `MultiFloatsExt`, `GenericLinearAlgebraExt` (native-BigFloat verified decompositions — the `verified_*_gla` functions are *stubbed* in `src/BallArithmetic.jl` and implemented here), plus experimental `CUDAExt` / `CuTileExt` GPU paths.

`GenericLinearAlgebra` and `GenericSchur` are regular `[deps]`, always loaded — no extension gate needed (used for BigFloat eigen/Schur where LAPACK is Float64-only).

## Conventions and gotchas

- Many high-level routines return a dedicated `*Result` struct (e.g. `RigorousEigenvaluesResult`, `RigorousSVDResult`) rather than a bare tuple — check the struct's fields.
- Keep helper functions **type-parametric** (`BallMatrix{T,NT}`) so BigFloat inputs work; several bugs have come from Float64-only assumptions.
- `MultiFloats` is pinned to `2` and support is experimental — don't bump it aggressively.
- Minimum Julia is `1.10`.
