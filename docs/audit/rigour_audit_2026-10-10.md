# Rigour audit of BallArithmetic.jl, 2026-10-10

Commit audited: `fe17455` (branch `consolidate-rp`). Scope: every file under `src/` and `ext/`.

## How it was done

Eight auditors, one per area, each read the code against the papers its docstrings cite and looked
for bounds that are not bounds. A finding is marked

- **confirmed** when a probe produced a counterexample (numbers given) or the paper and the code
  were quoted against each other;
- **suspected** when it rests on reading only.

All probes ran on nighthawk in the worktree at `fe17455`, against BigFloat references of 512 to
2048 bits. The scripts are in `nighthawk:~/scratch/audit_*/`. The findings marked **re-run** below
were run a second time by the coordinator and gave the reported numbers.

Categories: UNSOUND (a returned bound can be false), UNFAITHFUL (differs from the cited paper
without saying so), MISCITED, STUB (exposed, not implemented), UNTESTED, DOC.

Not checked: any claim resting on a paper that is not in `~/Library` (list at the end).

## 1. What the certified pseudospectra pipeline depends on

RigorousPseudospectra does not use `CertifScripts`; its circle cover is its own `_check_arcs`
(`src/twoblock.jl`), which bisects in angle, samples on the circle, charges the half arc length
plus `8u(|c| + r)`, and uses emulated directed operations only (read by the coordinator, no fault
found). Its use of BallArithmetic goes through `verifyeigall`, the SVD bounds, the ball matrix
product and the norm bounds.

| item | where | status | what |
|---|---|---|---|
| P1 | `src/types/ball.jl:485`, `src/types/array.jl:282`, `in0` on complex balls | UNSOUND, confirmed, re-run | `abs(c1 - c2) + r1 < r2` under RoundUp: the complex subtraction and `hypot` do not round the modulus up. True for a point outside the open ball in 9 of 200000 (scalar) and 16 of 200000 (array) random cases. Example: `|c1 - c2| - r2 = +2.58e-17`, `in0 = true`. This is the test (2.10) of `verifyeigall` (`rump_verifyeigall.jl:328`), the only acceptance test of Theorem 2.2. Fix: `add_up(dist_up(c1, c2), r1) < r2`. |
| P2 | `src/types/ball.jl:353`, `inv` of a complex ball | UNSOUND, confirmed | With radius `eps|c|`, members on the boundary of the input leave the result in 1987 of 50000 cases, by up to 32% of the radius; `inv(Ball(1.0+0im, prevfloat(1.0)))` returns a negative radius. Used for `R̃` of (2.8)-(2.9) in `verifyeigall` (`rump_verifyeigall.jl:423`): the exact `1/(D_l - D_j)` was outside the ball formed in 4 of 100000 cases. |
| P3 | `rump_verifyeigall.jl:222`, Neumann transform | UNSOUND, confirmed | `defect * ‖P‖ / (1 - defect)` under RoundUp rounds the denominator up; below the exact value in 5144 of 100000 scalar cases. Only `:rump2022aneumann`. |
| P4 | `rump_verifyeigall.jl:487`, `blocks[i]` | UNSOUND, by reading | `λ_i I + Z_ii` added in nearest, no radius for it. `radii[i]` is not affected. |
| P5 | `rump_verifyeigall.jl:185, 297`, `abs.` of complex matrices under RoundUp | suspected | No input found on which the radius is too small. The complex accuracy term in `_rump2022a_prodK` charges `γ²` where the real and imaginary parts together need `√2 γ²`; no underflow term. |
| P6 | `src/eigenvalues/upper_bound_spectral.jl:14`, `collatz_upper_bound` | UNSOUND, confirmed | Returns NaN when an iterate has a zero entry (`[0 1; 0 0]`). At its call sites in `verifyeigall` and `miyajima_2014a` the matrices are strictly positive and NaN fails `< 1`. |
| P7 | `src/norm_bounds/rigorous_opnorm_bounds.jl:51, 157`, `upper_bound_L2_opnorm` | UNSOUND in two edge cases, confirmed | `iterates = 0` returns 1.0 for `3I`; an overflowing iterate returns NaN (entries 1e40), which `min` propagates. |
| P8 | `src/types/ball.jl:310`, complex `Ball * Ball`; `matrix.jl:168`, `vector.jl:99` scalar times ball matrix | UNSOUND, confirmed, re-run | Radius charges `2u|c|`; a complex product needs `√2 γ₂ ≈ 2.83u`. Example: radius 2.2211e-16, exact error 2.4660e-16. |

No false final result of `verifyeigall` was observed: 60 of 60 certified eigenvalues inside their
discs in the auditor's check, and the sweeps of 2026-10-09 (120 matrices, 2048-bit reference).
P1 and P2 are nevertheless what the certificate rests on.

Checked against the papers and found to match: `_miyajima2014_thm4`, `_thm7`, `_thm10`
(Miyajima 2014, JJIAM), `_rump2011_thm3_1` for real input (Rump 2011), `_rump2010_alg10_7` /
`verifylss` (Rump 2010, Algorithm 10.7, Theorem 10.8), `miyajima_2014a.jl` (rigour pass), `MMul4`
(by reading, given a BLAS that honours the rounding mode; 0 of 2500 entries not enclosed on each of
three non-BLAS paths).

## 2. `CertifScripts` and the resolvent drivers

These are what `run_certification` and its variants compute; other projects of the estate use them.

| item | where | status | what |
|---|---|---|---|
| C1 | `CertifScripts.jl:745-906`, `adaptive_arcs!` | UNSOUND, confirmed, re-run | An arc is bisected at the midpoint of its chord, so refinement points lie on the inscribed polygon and the accepted discs fall short of the circle. `A = diag(p, 0)`, `p = (1 + δ) cis(π/256)`, unit circle, 256 samples: `resolvent_schur` = 11409 for δ = 1e-4 (true 1e4), 23447 for δ = 1e-5 (true 1e5), 26213 for δ = 1e-6 (true 1e6); every sample has modulus ≥ 0.9999247. Fix: bisect in angle and bound the arc. |
| C2 | `CertifScripts.jl:646, 1940`, parametric evaluator | UNSOUND, confirmed | The arc test is fed `1/M_A`, the bound for the leading block only. 4x4 example: 165.8 returned, true maximum 1e6. |
| C3 | `CertifScripts.jl:957-1001`, `bound_res_original` | UNSOUND, confirmed | The lift from the Schur form omits the term `|z| ‖Z'Z - I‖`. Constructed case with `errF = 1e-3`: returns 2.004 at a point where `zI - A` is singular. Not tested with a floating-point Schur factor, where the term is `|z|·1e-16`. No reference is cited for the formula. |
| C4 | `CertifScripts.jl:1619-1649`, BigFloat Ogita evaluator | UNSOUND, confirmed | When the certified ball for σ_min contains zero, it is replaced by a positive number (`100 eps |c|`). At an exact eigenvalue: `lo_val = 1.7e-77`, `hi_res = NaN`. |
| C5 | `CertifScripts.jl:318`, Ogita cache path | UNSOUND, confirmed | The radius of `T` is dropped. `T = diag(2, 0.5) ± 0.3`: returned 0.5, a member has 0.2. Applies when `polynomial` is given. |
| C6 | `rigorous_contour.jl` | STUB / UNSOUND, by reading | `bound_enclosure` is exported and does not exist; `errF`, `errT` are computed and unused; the loop-closure flag is the literal `true`; points and radii returned belong to different discs. |
| C7 | `gram_transform.jl:379` | UNSOUND at the last bit, confirmed | BigFloat upper bounds converted to Float64 in nearest: below the bound in 30, 23 and 37 of 60 cases for the three diagnostic fields. |
| C8 | `sylvester_resolvent_bound.jl` | UNSOUND, confirmed (probes on bemtivi, not re-run) | `estimate_2norm(M, RowCol2Norm)` returns a lower bound (2.83 for `ones(8,8)`, exact 8); the residual `R = T12 + T11 X - X T22` and the coupling `Δ` are nearest-rounded (computed `‖Δ‖` below the exact by up to 2.13 in 16 of 200); `T22` accepted as triangular up to `sqrt(eps)` and its lower part then ignored (1e12 returned, exact 1e15). The README says "All bounds are rigorous upper bounds using interval/ball arithmetic". |

Untested: `run_certification_ogita`, `run_certification_parametric`, `adaptive_arcs!`,
`bound_res_original`, `bound_resolvent_schur`, `schur_to_original_resolvent`. No test compares a
returned resolvent bound with one computed independently.

## 3. Ball types and products (`src/types`, `src/rounding`)

Besides P1, P2, P8:

| where | status | what |
|---|---|---|
| `rounding.jl:28`, `machine_epsilon` | UNSOUND, confirmed | `2^-52` for Float64 (= 2u) and `2^-precision` for BigFloat (= u). Complex BigFloat products at 128 bits: radius below the exact error in 1525 of 20000. |
| `ball.jl:220-238`, conversions, `Ball(c)` | UNSOUND, confirmed, re-run | Midpoint rounded with no radius: `convert(Ball{Float64}, Ball(big(1)/3, 0))` and `Ball(1//3)` have radius 0. |
| `ball.jl:407`, `abs(::Ball)` complex | UNSOUND, confirmed, re-run | `hypot` in nearest, radius unchanged: misses `|z|` in 100000 of 100000. |
| `ball.jl:153, 194`, `ball_hull`, `intersect_ball` | UNSOUND, confirmed, re-run | Centre in nearest, radius without its rounding: `ball_hull(1, nextfloat(1))` excludes `nextfloat(1)`. |
| `ball.jl:431`, `in(x, B)` | UNSOUND, confirmed | Nearest-rounded difference: `in(-2^-60, Ball(1, 1))` is true. |
| `oishi_mmul.jl:75, 210`, `_ccr`, `_cr` | UNSOUND, UNFAITHFUL, confirmed | Same centre/radius pattern; Miyajima 2010, Algorithms 3 and 5, compute the radius as `fl△(centre - lower)`. |
| `ogita_rump_oishi.jl`, `mmul_ogita_rump_oishi_2005` | UNSOUND under underflow, confirmed, re-run | No underflow term: 100 products of 3e-165 by 7e-160 return `0 ± 5e-324`, exact 2.1e-322. Complex case charges one accuracy term where `√2` is needed (by reading). |
| `MMul5.jl` | UNFAITHFUL, by reading | Revol-Théveny 2013 require the two products computed in the same order. |
| `MMul2.jl` | STUB | `@warn "Not Implemented"`. |
| `MMul4.jl` | DOC | No citation; the construction is Rump 1999, Algorithms 3.4 and 4.3. |

## 4. Linear systems (`src/linear_system`)

Faithful: `verifylss` / `_rump2010_alg10_7`; Gaussian elimination and back substitution (ball
operations only).

| where | status | what |
|---|---|---|
| `krawczyk_complete.jl:91`, `krawczyk_linear_system` | UNSOUND, confirmed | Nearest-rounded `I - RA`; the test omits the residual term. `3x = 1`: `verified = true`, radius 0. Called by `riesz_projections.jl:251`. |
| `verified_linear_system_hmatrix.jl:84`, all four methods | UNSOUND, UNFAITHFUL, confirmed | Docstrings say "with directed rounding"; there is none. `3x = 1`: `verified = true`, `error_bound = 0`. `:minamihata_2015` returns the Rump bound ("for simplicity"). |
| `verified_linear_system_hmatrix.jl:278`, `mag` | UNSOUND, confirmed | `mag(Ball(-1, 2^-60)) = 1.0`. |
| `hbr_method.jl` | UNSOUND, UNFAITHFUL, STUB, confirmed | Not Theorem 5.12 of the thesis; `rad(b)` ignored: `b = 3 ± 0.1` gives solution radius 0. |
| `shaving.jl` | UNSOUND, STUB, confirmed | "simplified implementation"; removed the exact solution (3 excluded from [2.69, 2.979]). |
| `sylvester.jl:269`, `triangular_sylvester_miyajima_enclosure(::BallMatrix, k)` | UNSOUND, confirmed | First-order inflation by a diagonal gap: 8 of 16 vertex solutions outside, ratio 4367. Used by `spectral_projection_schur.jl:397, 732`. |
| `sylvester.jl:729`, `schur_sylvester_miyajima_enclosure` | UNSOUND, confirmed | Schur factors taken as exact; radii of solved blocks not propagated: 3 of 900 entries outside. |
| `sylvester.jl:13`, `sylvester_miyajima_enclosure` | UNFAITHFUL, confirmed by the paper | `R_V` of Miyajima 2013, Theorem 2, is not computed; denominators rounded the wrong way. No false enclosure in 500 entries. Called by `block_schur.jl:329`. |
| `overdetermined.jl`, `solvable` field; `ball_hull(::BallVector, ::BallVector)` | UNSOUND, confirmed | `solvable = true` for the inconsistent `3x = 1, x = fl(1/3)`. |
| `iterative_methods.jl`, Gauss-Seidel, Jacobi | UNSOUND as documented, confirmed | The unverified initial box is intersected with every iterate: `[0.52, 3.0]` returned for a solution set `[1/1.9, 10]`. |
| `preconditioning.jl:259`, `is_well_preconditioned` | UNSOUND, confirmed | 30 of 200 false at the threshold. |
| `krawczyk_sylvester`, `sylvester_krawczyk_enclosure`, `interval_least_squares(method = :qr)` | STUB | Never verifies / throws / falls back. |
| `Krawczyk.jl`, `system_solvers.jl` | dead | Not included anywhere. |

`test_triangular_eigenvectors.jl` is not included by `runtests.jl`.

## 5. Eigenvalues (`src/eigenvalues`, other than `verifyeigall` and Miyajima 2014a)

| where | status | what |
|---|---|---|
| `iterative_schur_refinement.jl:668`, `rigorous_schur_bigfloat` | UNSOUND, confirmed | Radii from the residual with no separation: `T_rad = 6.4e-77`, nearest eigenvalue at 1.65e-71. |
| `iterative_schur_refinement.jl:1067`, `rigorous_symmetric_eigen_bigfloat` | UNSOUND, confirmed | Eigenvector at distance 0.129 against radius 4e-10. |
| `ordschur_ball.jl:59`, Givens steps | UNSOUND, confirmed | Row k+1 uses the magnitude bound of row k: entry (2,2) outside in 2000 of 2000 swaps, ratio up to 1.66e7. `orth_defect` and `fact_defect` are returned as 0. |
| `spectral_projection_schur.jl:463`, `compute_spectral_projector_hermitian` | UNSOUND, confirmed | Eigenvectors taken as exact, input radius unread: true projector outside by 1.37e-7 against radius 1.1e-16. |
| `spectral_projection_schur.jl:146`, `compute_spectral_projector_schur`; `:751`, `compute_spectral_coefficient` | UNSOUND, confirmed | Input radius never used; member projector outside by 2.56e-5. |
| `spectral_projectors.jl:99`, `miyajima_spectral_projectors`, `gev_invariant_subspaces` | UNSOUND, confirmed | `P = V * inv(V)` with a floating-point inverse wrapped in zero-radius balls: exact projector outside in 12 of 12 clusters. The four defects are what is proved. |
| `block_schur.jl:309`, `refine_off_diagonal_block` | UNSOUND, broken, confirmed | Wrong sign, wrong shape, radii dropped; throws `DimensionMismatch` on non-square blocks; its test accepts any exception. |
| `block_schur.jl:96`, `rigorous_block_schur` | DOC / overclaim | `Q_inv` is a floating-point inverse; truncated blocks get zero radius. |
| `rump_lange_2023.jl` | STUB, UNSOUND, MISCITED, confirmed | Gershgorin discs of `A` itself; no theorem of Rump-Lange 2023 is implemented. `[0 1; 4 0]`: "certified" `Ball(0, 1)` holds no eigenvalue. |
| `verified_gev.jl:374`, eigenvector bounds | UNSOUND for unsorted input, UNFAITHFUL, confirmed | The hypothesis `ξ̂_i < √g_i` of Theorem 7 (Miyajima-Ogita-Rump-Oishi) is never tested. |
| `gev.jl`, `rigorous_generalized_eigenvalues`, `rigorous_eigenvalues`, `gevbox`, `evbox` | DOC, confirmed | The union statement of Theorem 2 is returned as one ball per eigenvalue; a given ball may hold none. |
| `sep_clusters` (`schur_gershgorin.jl:467`) | UNSOUND, confirmed | "Rigorous lower bound" above the exact distance in 49775 of 100000. |
| `miyajima/proceduresMiyajima2010.jl` | dead | `T` undefined; throws. |

## 6. SVD (`src/decompositions/svd`)

| where | status | what |
|---|---|---|
| `singular_gerschgorin.jl`, `qi_intervals`, `qi_sqrt_intervals` | UNSOUND on non-square input, confirmed | The interval `B_{n+1}` of Qi 1984, Theorem 2, is missing: 134 of 6000 cases with a singular value outside. `qi_sqrt_intervals` contains no square root and returns Theorem 2's intervals. |
| `precision_cascade_svd.jl`, `σ_min` "certified" | UNSOUND as a certificate, by reading | `Σ[end] - ‖A - UΣV'‖_F` in nearest, with no orthogonality defect. No numerical failure in 40 cases. |
| `schur_gershgorin.jl:143`, `block_enclosure` | suspected | `β₂` enters without the factor `√q` of the stated derivation. |
| `svd.jl:580`, `refine_svd_bounds_with_vbd` | UNSOUND at the last bit, MISCITED | Interval ends in nearest; does not use Theorem 11 as its docstring says. |
| `rump_2011.jl:99` | UNSOUND at the last bit for complex input, by reading | `abs` where `abs_down` / `abs_up` are needed. |
| `svd.jl`, `RigorousSVDResult.U`, `.V` | DOC, confirmed | Documented as enclosures of the singular vectors; they are the floating-point factors with radius 0. |
| `miyajima_2014.jl`, `_miyajima2014_thm11`, `rigorous_svd_m4` | UNFAITHFUL in the statement, suspected | The theorem gives a numbering ν; the code reads the i-th interval as σ_i without sorting. 18300 of 18300 probes inside. |
| `adaptive_ogita_svd.jl` | UNFAITHFUL, confirmed | Fails on every non-square shape; the precision doubling does not happen. |
| `rigorous_svd(...; method = MiyajimaM4())` on BigFloat | STUB | Throws an error that advises the same call. |
| `rigorous_svd` with an Inf radius | confirmed | Returns `Ball(NaN, NaN)` silently. |

## 7. Other decompositions (`src/decompositions`) and the extended-precision extensions

| where | status | what |
|---|---|---|
| `verified_lu.jl` | UNSOUND, confirmed, re-run | `X_L A X_U` in floating point taken as exact, `X⁻¹ = L̃` assumed. n = 20: 190 entries of L and 204 to 209 of U outside, `success = true`. |
| `verified_qr.jl` | UNSOUND, confirmed, re-run | Same pattern; Q radius is a nearest-rounded heuristic. n = 20: 100 to 190 entries of Q and 139 to 210 of R outside. |
| `verified_takagi.jl`, `:svd`, `:svd_simplified` | UNSOUND / STUB, confirmed | `:svd_simplified` does not compute a Takagi factor (reconstruction error 0.56). |
| `rigorous_residual.jl:29`, `_rigorous_MMul_real` | UNSOUND, confirmed | Midpoint-radius conversion loses the bracket: 65 of 7200 entries outside. |
| `rigorous_residual.jl:81` | UNSOUND, confirmed | `abs(fl↑(x))` of a negative difference. |
| `iterative_refinement.jl`, `refine_polar_qdwh` | wrong, confirmed | `converged = true` with orthogonality defect 1.86. `refine_cholesky`, `refine_lu`, `refine_takagi` do not converge from a 1e-8 perturbation. |
| `verified_polar.jl` | STUB | Heuristic radii under a "Mathematical Guarantee" docstring. |
| `verified_cholesky.jl` | held | No entry outside; declines otherwise. |
| `ext/`: `verified_{lu,qr,cholesky,takagi}_{double64,multifloat}`, `verified_*_gla` | UNSOUND, confirmed | Radii of 1e-47 to 1e-77 around midpoints off by 1e-32 to 1e-16. None is tested. |
| `ext/`: `refine_schur_multifloat` | STUB | `converged` is the literal `true`. |
| `ext/FFTExt.jl` | MISCITED, suspected | The constant of a radix-2 analysis applied to `FFTW.fft`; cited paper not available. |
| `ext/CUDAExt.jl` | suspected (no GPU) | No check on the size bound `K`. |

## 8. Norm bounds and matrix properties

| where | status | what |
|---|---|---|
| `rump_oishi_2024.jl`, `:backward`, `:hybrid` (default), `backward_singular_value_bound` | UNSOUND, MISCITED, confirmed | `[1 0.5; 0 0.5]`: exact ‖T⁻¹‖₂ = 2.288, returned 2.000. "Theorem 3.2" does not exist in Rump-Oishi 2024. |
| `rump_oishi_2024.jl`, `:psi` | UNSOUND, confirmed | `_collatz_strictly_triangular` returns 1.0 for norms 5 and 0.1; 2.0005 returned against 10.099. |
| `oishi_2023_schur.jl:311, 155`, `oishi_2023_sigma_min_bound`, `rump_oishi_2024_sigma_min_bound` | UNSOUND, confirmed | `_diagonal_scale_left` returns radius 0 on a rounded product (50000 of 50000); the lower bound of `|d|` is rounded up: `σ_min ≥ 1.0` certified for a member with `1 - 1e-17`. |
| `triangular_inverse_bounds.jl`, `psi_squared`, `similarity_condition_number` | UNSOUND as documented, confirmed | Nearest rounding throughout; upper-triangularity unchecked (1.0 against 101). |
| `determinant.jl`, `det_gershgorin`, `det_hadamard` | UNSOUND, confirmed | `det = 10`, enclosure `[-16, 4]`. |
| `regularity.jl`, `is_regular*`, `is_singular_sufficient_condition` | UNSOUND, confirmed | Regular certified for a ball containing a singular matrix in 423 of 4000; "singular" returned for a ball of matrices of determinant 1. |
| `is_M_matrix.jl` | MISCITED | Tests an H-matrix condition; true for `-I`. |
| `poly_range.jl:77, 173` | UNSOUND in principle | Box and derivative coefficients formed in nearest; no false range produced. |

## 9. Citations

- `RumpOgita2024` is cited in every `verified_*` file and has no entry in `docs/src/refs.bib`; the
  nearest entry gives another title and journal. Crossref: Rump and Ogita, "Verified Error Bounds
  for Matrix Decompositions", SIAM J. Matrix Anal. Appl. 45 (2024) 2155-2183, doi
  10.1137/24m165096x. `Cariolaro2016`, `Dieci2022` have no entry either.
- `refs.bib`, key `Miyajima2010`: title "Fast verified matrix multiplication" with the coordinates
  of "Fast enclosure for all eigenvalues in generalized eigenvalue problems", JCAM 233 (2010)
  2994-3004.
- `rump_2011.jl:11`: doi 10.1007/s10543-010-0295-z; the paper is 10.1007/s10543-010-0294-0.
- `svdbox` is said to be Miyajima's name for his Theorem 7; the word does not occur in the paper,
  which calls the algorithm M1. `rigorous_svd` says it follows Theorem 3.1 of Rump 2011; no path
  uses it.
- `block_schur.jl`, `spectral_projectors.jl`: `Miyajima2014` (singular values) and `Miyajima2014a`
  cited for a block Schur form and projectors; neither paper has them.
- `verified_gev.jl`: "Corollary 3.2 of Miyajima (2014)" is in Miyajima 2014a; "Lemma 2" is another
  statement; `gev.jl`: "JCAM 246" is JCAM 236.
- `iterative_schur_refinement.jl`: "Numer. Algorithms 95 (2024)" is 92 (2023); "equation (23)" is
  (25); "Example 10" does not exist.
- `rump_lange_2023.jl`: "SIAM J. Matrix Anal. Appl., to appear" is JCAM 434 (2023) 115332.
- "Horáček (2012)": the thesis in `references/` is dated 2019.
- No reference at all: the arc-cover argument, the Schur-to-original lifting, the Gram change of
  variables, `sylvester_resolvent_bound.jl` ("Custom algorithm").

## 10. Papers cited and not available, so not checked

Ogita, Rump, Oishi 2005 (the bound in `compensated_terms`); Rump and Ogita 2024; Ogita and Aishima
2018 (part I; read from `references/` for RefSyEv only); Oishi 2023; Oishi 2001 (LAA 324); Krawczyk
1969; Neumaier 1990; Alefeld and Herzberger 1983; Hansen and Bliek 1992; Rohn 1989, 1993, 2006;
Feingold and Varga 1962; Collatz 1942; Mirsky 1960; Higham 1986, 2008; Nakatsukasa, Bai, Gygi 2010;
Nakatsukasa and Higham 2013; Brisebarre, Muller, Picot 2023; Knight and Kaiser 1979; Kantorovich
and Akilov; Cariolaro 2016; Dieci 2022; Ozaki, Uchino, Imamura (arXiv:2504.08009).

`~/Library/ToBeProcessed/Rump2010.pdf`, `RumpActaNum.pdf`, `MiyajimaEtAl2010.pdf`, `krawczyk.pdf`
are annex pointers without content on bemtivi. `Higham1996.pdf` is the 2002 second edition.
`Rump2011a.pdf` is "Fast interval matrix multiplication"; the singular value paper is
`Rump2011.pdf`.

## 11. State of the entries of Section 2 on the branch `rigour-fixes`

Added after the audit; the entries above are left as written on 2026-10-10.

| | commits | what the code does now |
|---|---|---|
| C1 | `da9262c`, `1857a49` | The entry misread the contract. The refinement bisects the sides of the inscribed polygon and the certified contour is that polygon, which the docstrings now say. The side test, the stop conditions and the bound on the polygon use the emulated directed operations; a refinement that stops (non-positive bound, evaluation cap, a side that cannot be split) makes the driver throw. `circle_resolvent_bound` carries the bound from the polygon to the circle through the sagitta `h ≤ r(π/N)²/2`, `M/(1 − Mh)` when `Mh < 1`. |
| C2 | `8148699`, `7498e24` | The side test is fed the reciprocal, rounded down, of the bound for the whole matrix. |
| C3 | `0cac707`, `9379198` | `schur_to_original_resolvent` is Lemma 5.1 of Blumenthal, Nisoli and Taylor-Crush (arXiv:2507.09021) with its hypothesis (28) checked, and needs `zmax`. The code had `(1 + ε²)` where the lemma has `(1 + ε)²` and did not check the hypothesis. `schur_to_original_resolvent_defects` is a second bound with the two defects separate, proved in its docstring. |
| C4 | `8148699` | A ball for σ_min that is not proved positive is returned as it is, with resolvent bound `Inf`. |
| C5 | `8148699` | The matrix certified is the ball `T − zI`. |
| C8 | `c693b19`, `081f7c0` | One implementation, `parametric_resolvent_bound`. `R` and the residuals of the solves are ball matrices, the shifts and the scalar operations are rounded outward, `RowCol2Norm` is removed, a `T22` with a nonzero entry below the diagonal is refused. The four other implementations are removed. |

`b34d7e5` leaves one copy of the code the four drivers, the two refinement loops and the four
workers shared. On the matrix of `probe_drivers.jl` (n = 12, 16 samples, nighthawk) the seven
drivers that returned before it returned the same `minimum_singular_value`, `resolvent_schur` and
`resolvent_original` to 17 digits after it; `‖Z‖` and `‖Z⁻¹‖` for Float64 input went from
`1 + 1.7e-14` to `1 + 3.3e-15`, being now computed from `errF`. The distributed parametric driver
did not return within 240 s before it and returns after it.

`refine_svd_bounds_with_vbd` is removed. `apply_vbd = true` only stores a block diagonalisation of
`Σ*Σ` in the result and leaves the singular values as they are, so the evaluators never reached
that function (an earlier line of this section said they did); its only caller was its own test.

Not done: C6, C7; `ordschur_ball`, `compute_spectral_coefficient`, `rigorous_block_schur`.
