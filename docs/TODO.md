# To do and further research

Opened 2026-10-11 on the branch `rigour-fixes`. Each item says what was run, where the output is,
and what was not tried. The probe scripts are one-off and live in `~/scratch` on nighthawk, with
their outputs beside them (`<script>.nighthawk.txt`).

The papers are in Isaia's library (`~/Library`), not in the repository: the PDFs were taken out
of `references/` on 2026-10-11 (they remain in the history), `references/references.bib` stays,
and `references/*.pdf` is ignored.

## Frames for `verifyeigall` and for the block resolvent floor

`block_resolvent_floor` divides by the condition number of the similarity, so the frame decides
the bound. The frames that exist: the eigenvector matrix (`:rump2022a`), the Schur factor
(`:rump2022aschur`), the Schur factor followed by step 6 (`:rump2022aschurstep6`), the choice by
`cond` of the eigenvector matrix (`:rump2022aconditioned`), and the Schur-Newton frame of
`miyajima2014a_schurnewton`.

1. **What is a block for the recursion in the Schur frame.** Step 6 transforms all uncertified
   columns together by the eigenvectors of their diagonal block. In the Schur frame the clusters
   of step 2 are singletons, so "inside each block" needs a grouping of the uncertified columns:
   by closeness of the diagonal entries (the rule of `:rump2022adiscclusters`, or a `sep` as in
   Schur-Newton), each group transformed separately. Not implemented; the question is open.

2. **A restarted frame.** Eigenvectors for the well-conditioned eigenvalues, an orthonormal
   basis of the invariant subspace of the others, one transformation from the original matrix,
   the others kept as one cluster. Probe: `probe_restart_frame.jl` (commit `28e82fa`), split by
   `s_i = ‖x_i‖‖y_i‖/|y_i* x_i| ≤ tau`. On the Jordan 6 + diagonal 6 matrix with `tau = 10` the
   frame had `κ = 1.04`, all 12 columns certified, median floor/true 0.963 (eigenvector frame:
   `κ = 1.23e3`, 0.001). On Grcar 32 + diagonal 8 the split put 38 of 40 eigenvalues in the good
   set and `κ` was 44.1 against 46. Not tried: another definition of the bad set, more than two
   parts, recursion inside the bad block, ball input.

3. **The default of `verifyeigall(B)`.** It is `:rump2022a` with `:rump2022aschurstep6` as
   fallback where the spectrum is left uncovered. `:rump2022aconditioned` is not the default: on
   Grcar 48 and 64 it certifies nothing where `:rump2022a` certifies every eigenvalue
   (`probe_rump_schur_sizes.jl`). Whether the selection should depend on what the caller wants,
   eigenvalue discs or a resolvent bound, is undecided.

4. **One entry point for the floor.** Every route gives a valid lower bound of `σ_min(A − zI)`,
   so the pointwise maximum over routes is one too. Not implemented.

5. **The merge rule of the block diagonalisation draft.** `miyajima2014a_schurnewton` splits
   Grcar into singletons by default and keeps it as one block with `sep = 2`; the draft's rule
   that would choose the separation was not located.

6. **`:rump2022aconditioned` for a pencil**, with the generalised Schur factor `Z` as the
   alternative frame. Not implemented.

7. **`transform_defect` above one on an accepted transformation.** On the 24-fold Jordan block
   under an orthogonal similarity `verifyeigall` returned `transform_defect = 1.5455` with the
   transformation accepted. The field is `spectral_radius_bound` of `verifylss`, an upper bound
   of `‖I − RA‖₂`, while the acceptance is the inclusion test of Algorithm 10.7, which can hold
   when that norm bound is not below one; the sentence "must be below one" in the docstring of
   `VerifyEigAllResult` is the part to correct.

8. **Effect of the fallback on callers.** `poly_range` and the counts of the companion package
   call `verifyeigall` without a method, so they now receive the fallback's result where the
   paper's algorithm leaves the spectrum uncovered. The recount of the 141 frames of the paper
   was not rerun after this change.

9. **Sizes above 64, ball input and BigFloat** were not run for the Schur variants; the pencil
   methods were run on one random pencil per size up to 128 (`probe_pencil_methods.jl`).

## Salvage from the companion's `main`

10. Moved on 2026-10-11 (`e310415`, `7e828ce`): `svd_frame_floor` (the `σ_min` surface from one
    singular value decomposition) and `pseudospectrum_cells` with `sigma_min_upper` (the cell
    sweep). Left behind: the use of the maximum principle on cells free of eigenvalues, the
    escalation to a cluster bound, and the record of which bound settled each cell. On the
    gallery (`probe_svd_frame_floor.jl`) the undecided cells at ε = 0.1 and `min_halfdiag = 0.05`
    covered 0.15% of the box for the normal matrix and 95% for Grcar 32, where neither floor is
    positive near the spectrum; a lower bound that is positive there at `O(n²)` for each `z` is
    the missing piece.
11. From the note on Miyajima's enclosure: the real Schur form with 2×2 blocks, individual radii
    and recentring by the diagonal of `S̃`. Not implemented.

## Left from the audit of 2026-10-10

12. Step 4, as decided on 2026-10-11.
    - **Rump and Ogita (2024) and Rump and Lange (2023), implemented** (2026-10-11, tip
      `bdceedf`): `src/decompositions/rump_ogita_2024.jl` has Algorithm 3.2, the LU
      decomposition for every shape (Sections 3.2 to 3.4), Cholesky (4), QR economy and full
      (5), Schur (8), polar (7) and Takagi (9); `src/eigenvalues/rump_lange_2023.jl` has the
      Hermitian eigenproblem, reached by `verifyeigall(A; method = :rumplange2023, kappa)`, and
      `verifysvdall(A; kappa)`, in place of the stub. Checked again on 2026-10-12 against
      Section 7 of Rump and Lange: the code has the listing of the paper, including the
      adjustment of the last radius of the left vectors for `m > n`, and Theorem 7.3 is stated
      for that listing. An enclosure of a basis of the left null space for `m > n` is not part
      of it: the paper says that this treatment "is omitted" from its algorithm and points to
      a full decomposition or to its reference [37] for the kernel of `A*`. It would be an
      addition, not a missing step. Not done: the paper's sizes (n = 100 to 1000) and tables
      were not reproduced, the tests using n up to 30 or 40.
    - **The exported decompositions, rewired** (2026-10-11, `dbec098` to `abc9c13`):
      `verified_lu`, `verified_qr`, `verified_cholesky`, `verified_polar` and `verified_takagi`
      are wrappers over the routines above, in Float64 or in BigFloat at `precision_bits`. On
      nighthawk (`probe_decomposition_containment.jl`, factors at 2048 bits, n = 10 and 40) the
      old `verified_lu` had 45 of 100 and 780 of 1600 entries of `L` outside its enclosure, the
      old `verified_qr` all entries of `Q`, the old `verified_cholesky` none; the new routines
      none. The new polar radii are larger than the old heuristic ones (n = 100, Float64:
      1.6e-10 against 2.7e-13 for `Q`), and the BigFloat runs are slower (n = 100, real: LU 4.5 s
      against 0.5 s, QR 8.0 s against 1.1 s, polar 11 s against 2.1 s;
      `probe_rewire_decompositions.jl`). RigorousInvariantMeasures calls `verified_cholesky`
      with `use_bigfloat = false` and reads `success` and `G`, which are unchanged; it was not
      rerun.
    - **The extension variants, rewired on 2026-10-12** (`ad54a98`).
      `verified_{lu,cholesky,qr,polar,takagi}_double64` and `_multifloat` are the routines
      without the suffix in BigFloat with the keyword `oracle = Double64` (or a MultiFloat):
      the approximate factorisation the method starts from is computed in that arithmetic,
      and the verification is the one of Rump and Ogita, unchanged. The `_gla` names are the
      BigFloat routines; `verified_svd_gla` is `verifysvdall` in BigFloat. Before, the
      perturbed identity was formed in Double64 or MultiFloat arithmetic with no bound of its
      rounding error, and five of the variants returned success with about half of the
      entries outside (`probe_extension_decompositions.jl`). `_lu_perturbed_identity`, which
      nothing calls any more, is removed with its 29 rounding-mode blocks; 2032 lines removed
      in all. Test: `test_extension_variants.jl`, every factor against references at 2048
      bits, real and complex, n = 6 and 15, Takagi at n = 4 and 9. None of the variants had a
      test before. The polar and Takagi variants with a MultiFloat need a generic `svd` and
      symmetric `eigen` for that type (GenericLinearAlgebra).
    - **The products in doubled precision, and `prodK`.** Rump and Ogita (2024) compute their
      products with INTLAB's `prodK`, `k = 2`, the scheme of M. Lange and S. M. Rump, "Accurate
      floating-point matrix residuals" (manuscript, in `~/Library/ToBeProcessed`).
      `_lange_rump_two_terms` (`src/types/MMul/lange_rump.jl`, 2026-10-11) implements, for
      Float32 and Float64, real and complex: Algorithms 1 to 4, the exactness of Lemma 4 and
      Corollary 5 with `λ = 1`, the balanced points (25), the evaluation (27) for any `k`, sums of
      products with the summands split separately, and an error matrix of our own (the paper does
      not list how `prodK` computes its second output). `_two_term_product_sum` and
      `_accurate_product_sum` call it and fall back on the Dot2 accumulation of Ogita, Rump and
      Oishi (2005) where it declines and for BigFloat. On nighthawk, `RA − I`, `R = inv(A)`,
      `cond(A) = 1e12`, 1000 sampled entries against exact values, no entry outside the bound in
      any run (`probe_prodk_timing2.jl`): one thread, n = 1000, time over the floating-point
      product 3.5, 7.3, 12.6 for `k = 1, 2, 3`, median error 4.6e-14, 1.2e-20, 0, median bound
      7.4e-10, 1.8e-16, 4.2e-23; the paper's Table 4 has the ratios 4.1 and 7.3 for `k = 1, 2`.
      With 16 threads the ratio at `k = 2` is 16.6 at n = 1000 and 12.1 at n = 4000 (the splitting
      and the summation are not threaded). The Dot2 accumulation: ratio 117 on one thread at
      n = 1000, median error 1.5e-22, median bound 5.7e-16 (`probe_prodk_timing.jl`,
      `probe_prodk_accuracy.jl`).
      **Left of the paper, to implement** (all of it, Isaia 2026-10-11):
      - `λ < 1` in Corollary 5, with the second bound (15) of Theorem 7;
      - unbalanced splittings, the points (23) and (26), and the call with `[k1, k2]`
        (Section 5), meant for matrix-vector products;
      - the Karatsuba evaluation (Lemma 14 and (28): 5 products for 6 at `k = 2`; Algorithm 6
        for general `k`, with its Table 6 of product counts), and the automatic choice between
        it and (27);
      - Toom-Cook (the products `T1` to `T5` at the points −1, 0, 1, 2, ∞ and condition (31));
        the paper says `prodK` does not use it;
      - the blocked summation of Section 8.1 (conditions (32), (33), Lemma 15), which also
        removes the limit on the dimension, and the reduced memory of its last paragraphs;
      - omitting the products with split parts that are zero (the option `NoReduction` of
        Table 9);
      - unevaluated sums of several matrices as one factor (Algorithm 7) and output in more than
        two terms with the normalisation (35) and its bound `8u|R_m|`;
      - sparse factors (Section 8.4, `spProdK`, and the option `δ` of Table 10); Section 8.4 has
        not been read yet;
      - SYRK for `A'A`, and TRMM for triangular factors;
      - the scaling of outliers of Section 8.2 beyond the separate summands already there;
      - threading of the splitting and of the final summation, which are `O(n²)` and single
        threaded (ours, not the paper's);
      - the paper's own tables (1 to 14) have not been reproduced beyond the one comparison
        above.
    - **Fixed:** `krawczyk_linear_system` is `verifylss` (`6751cf7`); RigorousInvariantMeasures
      calls it. `is_M_matrix` follows Definition 5.4 of Varga (2004) with its sign conditions,
      and `is_H_matrix` is the test on the comparison matrix (`2383207`). Sections 3.1 and 3.2
      of Rump and Ogita (2024) are in (`b698f68`, `5dcdfc0`).
    - **Fixed on 2026-10-11** (`7604ab4`, `1c58184`): `sep_clusters` (the distance from below),
      the two `σ_min` bounds of `oishi_2023_schur.jl` (rounding; `ψ` in closed form), the
      docstrings of `evbox`, `gevbox` and the two routines behind them (the union of the balls
      contains the eigenvalues, a single ball need not; this rests on the audit's reading of
      Miyajima 2012, which was not reread).
    - **Removed on 2026-10-11, at Isaia's word:** `miyajima_spectral_projectors` (both methods),
      `RigorousSpectralProjectorsResult`, `compute_invariant_subspace_basis`,
      `verify_projector_properties`, `projector_condition_number`, and the field `projector` of
      `gev_invariant_subspaces`. The spectral projector of a matrix is
      `compute_spectral_projector_schur`, by the resolvent on a contour as in the Gauss paper
      (arXiv:2602.19435). StatisticalPeriodicity was named as a user of the removed routine and
      was not checked. There is no projector for a pencil now; it would need a bound of
      `(zB − A)⁻¹` on a contour.
    - **Fixed on 2026-10-11** (`faa8f7f`): `adaptive_ogita_svd` and `ogita_svd_refine` on every
      shape, real and complex, each level at the precision it records. The refinement is
      Algorithm 1 of Ogita and Aishima (2020), stated there for a real matrix with `m ≥ n` and
      full factors; the conjugates in the two rectangular blocks for complex data are ours.
    - **From Horáček's thesis, done on 2026-10-11:** `regularity.jl` (`399a6cb` to `428ae6c`):
      Theorems 4.2, 11.12 (both conditions) and 11.13 (four conditions), Definition 4.24; tested
      against the vertex determinants in rational arithmetic on 800 interval matrices of order 2
      and 3. `determinant.jl` (`72c718d`): Sections 8.4.2 (Gerschgorin discs, Figures 8.3 and
      8.4), 8.4.3 (Hadamard, columns and rows) and 8.4.4 (Cramer's rule (8.3), by `verifylss`);
      the old `det_cramer` was a cofactor expansion and is now `det_cofactor`. **Left of
      Chapter 8:** monotonicity checking (8.4.5), preconditioning (8.4.6, Proposition 8.11),
      the verified determinant of a real matrix (8.5), the special cases of 8.6 (symmetric,
      symmetric positive definite, tridiagonal H-matrices).
    - **Chapter 5 of the thesis, done on 2026-10-11** (`af7dbac`, `d4261b3`):
      `interval_jacobi` and `interval_gauss_seidel` (Algorithms 5.15 and 5.17, stopping
      criterion of 5.7.2), `interval_initial_enclosure` (Proposition 5.14 and (5.4), new),
      `hbr_method` (Theorem 5.12, with `⟨A⟩⁻¹` enclosed by `verifylss`), `is_well_preconditioned`
      and `apply_preconditioner` on balls. Tested against the solutions of the vertex systems in
      rational arithmetic. Not implemented: extended interval arithmetic for a diagonal entry
      containing zero (5.7.4).
    - **`interval_shaving` (Section 5.9), done on 2026-10-11** (`e001b1c`): in the package, with
      the linear programme of 5.9.1 passed as a function and supplied by the extension
      `LinearProgrammingExt` (JuMP and HiGHS, weak dependencies, decided by Isaia). A slice is
      cut only after `verifylss` has enclosed the system built from the guess. In the suite on
      nighthawk a slice was cut in 270 of 270 runs from the initial enclosure of Proposition
      5.14, the largest radius after over before having median 0.075, and every shaved box
      contained the solutions of the vertex systems. Not implemented: the Sherman-Morrison
      update of the preconditioner in 5.9.3. The bound (5.16) was rederived, the printed one
      being illegible in the extracted text; the case where the system does not depend on the
      width is ours.
    - **Chapter 6 of the thesis, done on 2026-10-11** (`9321a2d`): `interval_least_squares`
      (the supersquare system of 6.2, and the interval normal equation), `subsquares_method`
      (Algorithm 6.12, intersection, unsolvability), `select_subsquares` (Algorithm 6.14, new),
      `multi_jacobi_method` (6.8.3; the preconditioning of each subsquare by its midpoint
      inverse is ours). Not implemented: the preconditioners of 6.3, Gaussian elimination for
      an overdetermined system (6.4), Rohn's method (6.6).
    - **`qi_intervals`, `qi_sqrt_intervals`, removed** (2026-10-11, at Isaia's word) with their
      rebalanced variants: Qi (1984) was tried and found not effective, and the audit found the
      interval `B_{n+1}` of its Theorem 2 missing for non-square input.
    - **Checked against Rump and Oishi's note, done on 2026-10-11** (`27cf3fa`):
      `rump_oishi_2024_triangular_bound` is Theorem 1.3 with a zero lower left block; its
      `:backward` and `:hybrid` methods, `backward_singular_value_bound` (a recursion and a
      "Theorem 3.2" that the note does not have), its `:psi` method, the copy
      `rump_oishi_triangular` and the commented-out `oishi_rump_bound` are removed.
    - **The Schur pair** (`c821268`): `certify_schur_decomposition` and `rigorous_schur_bigfloat`
      return the refined pair with radius zero and say that it is not an enclosure; its defects
      are measured by `ordschur_ball` and `compute_spectral_projector_schur`.
      `rigorous_symmetric_eigen_bigfloat` is removed (`fc6c817`, at Isaia's word);
      `verifyeigall(A; method = :rumplange2023)` is the verified routine.
    - **The eight `refine_*` routines, read against their sources and replaced on 2026-10-11**
      (`0237069`, `2ea0e74`, as Isaia decided). Four had a source whose algorithm starts from
      the matrix, while the code ran it from an approximate factor (and `refine_polar_qdwh` had
      two wrong formulas besides); they are now `polar_newton` (Algorithm Polar of Higham
      1986), `polar_newton_schulz` ((8.20) of Higham 2008), `polar_qdwh` (the QDWH algorithm of
      Nakatsukasa, Bai and Gygi 2010) and `cholesky_qr2` (Yamamoto et al. 2015), in
      `src/decompositions/polar_iterations.jl`; they are floating-point algorithms with no
      error bound and say so. Four had no source for what they did (`refine_lu`,
      `refine_cholesky`, `refine_qr_mgs`, `refine_takagi`) and are removed, with the result
      types and the Double64 and MultiFloat variants of all eight in `ext/`; the verified
      routines of Rump and Ogita cover those decompositions. The citations of the old file had
      a wrong journal, a wrong year and two DOIs of other papers. Not obtained: Wilkinson,
      *Rounding Errors in Algebraic Processes*, which nothing cites any more.
    - **`verified_linear_solve_hmatrix`, rewritten on 2026-10-11** after Minamihata, Ogita,
      Rump and Oishi (2020), which is in the library. The old code formed the residual, `RA`
      and every bound in plain floating point, and `:minamihata_2015` returned the bound of
      Theorem 3.1. Now: Theorems 3.1, 3.2, 3.3 and Corollary 3.2 (`method = :theorem_3_1`,
      `:theorem_3_2`, `:theorem_3_3`, `:corollary_3_2`, with `m` terms). Not reproduced: the
      two enclosures of `RA` of the paper (Methods (a) and (b)); there is one, the ball
      product. The residual is formed in working precision; an accurate residual (`Dot2` or
      Lange and Rump) is left to do.
    - **Removed on 2026-10-11:** `krawczyk_sylvester`, a stub that verified nothing; the
      verified routine is `sylvester_miyajima_enclosure`.
    - **`gev_invariant_subspaces`, removed on 2026-10-12.** Algorithms 1 and 2 of Miyajima
      (2014, SIAM J. Matrix Anal. Appl. 35; `Miyajima2014a` in the library) were already in
      the package, in `miyajima_2014a.jl` behind `verifyeigall(A, B)`, which is the one entry
      point. Added: a test that the enclosed basis and block contain those of a 512-bit
      eigendecomposition of a pencil, in `test_miyajima_2014a.jl`.
13. Step 5: citations and `docs/src/refs.bib`.
14. `setrounding` blocks that remain: BigFloat `upper_bound_L1_opnorm` and
    `upper_bound_L_inf_opnorm`, `block_enclosure`, `_vbd_block_data`, and those inside
    `rump_verifyeigall.jl` that predate the rule. `_vbd_kappa` is unrounded (a diagnostic).
15. A distributed certification run waits for ever if its workers fail.

## Further research

16. **Singular values under Gram conjugation without a verified Cholesky factor.** T. Terao,
    Y. Watanabe and K. Ozaki, "Verified error bounds for the singular values of structured
    matrices with applications to computer-assisted proofs for differential equations",
    arXiv:2502.09984 (2025), in `~/Library/ToBeProcessed`. Read: the abstract, Section 3 and
    the statements of the four theorems; not the proofs nor the numerical section. Its object
    is `σ_i(R⁻ᴴ A R⁻¹)` for `B = RᴴR` Hermitian positive definite. Theorem 1: from approximate
    factors `Û`, `Σ̂`, `V̂` of the decomposition `A V_B = B U_B Σ`, `U_Bᴴ B U_B = V_Bᴴ B V_B = I`,
    with `δ = ‖Ûᴴ A V̂ − Σ̂‖`, `‖Ûᴴ B Û − I‖ ≤ α < 1`, `‖V̂ᴴ B V̂ − I‖ ≤ β < 1`,
    `(σ̂_i − δ)/√((1+α)(1+β)) ≤ σ_i ≤ (σ̂_i + δ)/√((1−α)(1−β))`. Theorems 2 to 4 bound `σ_min`
    without a singular value decomposition, by positive definiteness of `AᴴA − θB²` and
    `AAᴴ − θB²`, or by the inertia of `[θB Aᴴ; A θB]` through an approximate `LDLᵀ`
    factorisation; they are aimed at sparse matrices.
    How it would meet `gram_transform` (algebra, not tested): with `G = LᴴL`, `R = L` and their
    `A` taken as `G(zI − A)`, `R⁻ᴴ[G(zI − A)]R⁻¹ = L(zI − A)L⁻¹`, so Theorem 1 would bound the
    singular values of the Gram-conjugated shifted matrix, hence the resolvent norm in the
    `G`-norm, from residuals formed with `G` and `G(zI − A)` alone, where `gram_transform`
    needs a verified Cholesky factor and a verified inverse of it.
    Run on 2026-10-11 (`probe_gram_terao.jl`, nighthawk, `d3de23d`, 16 threads): random `A`,
    `G = Q diag(1 … 1/κ) Qᵀ`, lower bounds of `σ_min(L(zI − A)L⁻¹)` at `z = 1.5` and
    `z = 0.3 + 0.2i`, n = 50, 100, 200, κ = 1e2, 1e6, 1e10; exact values at 512 bits for
    n ≤ 100. Four routes: ours with the BigFloat factor, ours with the Float64 factor,
    Theorem 1 with ball products, Theorem 1 with the products in doubled precision
    (`_two_term_product_sum`). Set-up of ours with the BigFloat factor: 0.9 to 13 s at n = 50,
    6 to 7 s at n = 100, 46 to 47 s at n = 200, once for all `z`; with the Float64 factor
    0.3 s at n = 200; Theorem 1 has no set-up. Per `z`, all four take 0.07 to 0.14 s at
    n = 200. At n = 100: κ = 1e2, all four agree with the exact value to the seven digits
    printed; κ = 1e6, `z = 0.3 + 0.2i` (exact 6.598093e-5): 6.598093e-5, 0 (no bound),
    6.565845e-5, 6.598087e-5; κ = 1e10, `z = 0.3 + 0.2i` (exact 3.638601e-6): 3.638377e-6, 0, 0,
    3.286427e-6. At n = 200, κ = 1e10, `z = 0.3 + 0.2i`: 5.887949e-7, 0, 0, 0. The defects of
    Theorem 1 at n = 100: `δ` 1.3e-10, 3.0e-7, 1.5e-3 with ball products and 5.7e-14, 9.6e-11,
    5.0e-7 in doubled precision, for κ = 1e2, 1e6, 1e10. Not run: the matrices of the paper,
    sparse input, Theorems 2 to 4, the upper bounds of Theorem 1 and its other singular values.

17. **Refinement of an approximate decomposition: the literature.** Searched on 2026-10-11,
    after the `refine_*` routines were removed for having no source. Bibliographic details were
    checked on Crossref or the arXiv where a DOI or an identifier is given; unless said
    otherwise only titles and abstracts were read.
    - **The Waseda line, on inverse factors with accurate dot products:** T. Ogita, "Accurate
      matrix factorization: inverse LU and inverse QR factorizations", SIAM J. Matrix Anal.
      Appl. 31(5) (2010) 2477-2497, doi 10.1137/090754376; T. Ogita and S. Oishi, "Accurate and
      robust inverse Cholesky factorization", NOLTA, IEICE 3(1) (2012) 103-111,
      doi 10.1587/nolta.3.103 (open on J-STAGE); Y. Yanagisawa, T. Ogita and S. Oishi,
      "Convergence analysis of an algorithm for accurate inverse Cholesky factorization", Japan
      J. Indust. Appl. Math. 31(3) (2014) 461-482, doi 10.1007/s13160-014-0154-4; S. Oishi,
      T. Ogita and S. M. Rump, "Iterative refinement for ill-conditioned linear systems", Japan
      J. Indust. Appl. Math. 26(2-3) (2009) 465-476, doi 10.1007/bf03186544. None is in the
      library. They refine an approximate inverse of a factor, the idea that Rump and Ogita
      (2024) verify; what they would add is accuracy for condition numbers beyond `1/u`, where
      our routines were not tested.
    - **Newton's method for decompositions in general:** J.-C. Yakoubsohn, "Unified Newton's
      method for matrix decompositions", Appl. Algebra Eng. Commun. Comput. 37(2) (2026)
      541-570, doi 10.1007/s00200-025-00720-7. Title only; no arXiv version found. The closest
      match to the question; to obtain and read first.
    - **Singular value decomposition:** Y. Uchino, T. Terao and K. Ozaki, "Acceleration of
      iterative refinement for singular value decomposition", Numer. Algorithms 95(2) (2024)
      979-1009 (in the library inbox as `s11075-023-01596-9.pdf`); W. Gao, Y. Ma and M. Shao,
      "A mixed precision Jacobi SVD algorithm", ACM Trans. Math. Softw. 51(1) (2025);
      D. Armentano and J.-C. Yakoubsohn, "High order numerical methods to approximate the
      singular value decomposition", arXiv:2309.06201.
    - **Symmetric eigendecomposition:** K. Shiroma and Y. Yamamoto, "Fixed-point analysis of
      Ogita-Aishima's symmetric eigendecomposition refinement algorithm for multiple
      eigenvalues", JSIAM Lett. 12 (2020) 5-8; T. Terao, T. Imamura and K. Ozaki, "Iterative
      refinement for an eigenpair subset of a real symmetric matrix", JSIAM Lett. 16 (2024)
      89-92; arXiv:2602.19090 and arXiv:2602.23778 (Terao with Ozaki, and with Ozaki, Imamura
      and Ogita, February 2026).
    - **Non-Hermitian:** Bujanović, Kressner and Schröder (the Schur refinement the package
      uses); T. Terao, "Iterative refinement for diagonalizable non-Hermitian
      eigendecompositions", arXiv:2604.02840 (2026); A. Dmytryshyn, M. Fasi, N. J. Higham and
      X. Liu, "Mixed-precision algorithms for solving the Sylvester matrix equation",
      arXiv:2503.03456.
    - **Survey:** N. J. Higham and T. Mary, "Mixed precision algorithms in numerical linear
      algebra", Acta Numerica 31 (2022) 347-414.
    - **LU, Cholesky and QR factors:** in the mixed-precision literature found, what is refined
      is the solution of the linear or least-squares problem, not the factors (Carson and
      Higham 2017 and the papers around it). One hit on the factors themselves was read and set
      aside (Isaia, 2026-10-11): M. P. Ahues and M. F. Hama, "Using Newton-Kantorovich method
      to compute QR and (L + I)U-factorizations", J. Math. Comput. Sci. 3(1) (2013) 332-351.
      Crossref and zbMATH have no record of it; it has four references; its convergence
      statement is the standard a posteriori one for Newton's method, quoted from Kantorovich
      and Krylov (1955); its examples are 4×4. Statements and examples were read, not the
      proofs. Not filed in the library.
    - **Polar and Takagi:** no paper on refining an approximate factor was found; for the
      polar decomposition only new iterations to compute it.
    - **How the citing papers were found:** the open API of zbMATH (its website is behind a
      bot check), searching reference lists for the titles of eleven seed papers; 59 distinct
      citing documents, a lower bound since zbMATH holds reference lists for part of the
      literature only and has none yet for Rump and Ogita (2024). Script and list:
      `zbmath_citing.py`, `zbmath_citing.bemtivi.tsv`, kept in `~/scratch` on nighthawk with
      the other probes.

18. **Return types, audited on 2026-10-12** at `ad2ffb2` (`audit_return_types.jl` and its
    output `audit_return_types.nighthawk.txt`, in `~/scratch` on nighthawk: every exported
    struct with its declared field types, and the return type that inference gives for each
    exported function on `Float64` arguments of nine shapes). Nothing was changed.
    - **Fields that are not concrete:** 28 fields in 13 of the 55 exported structs. Most are
      declared `BallMatrix{T, CT}`, `BallMatrix` or `BallVector{T}`, which leave the array
      parameters of `BallArray` free: `VerifyEigAllResult` (`subspaces`, `blocks`,
      `similarity`, `transformed`), `VerifySvdAllResult` (`U`, `V`),
      `SchurSpectralProjectorResult`, `SpectralCoefficientResult`, `RigorousBlockSchurResult`,
      `SVDFrameFloor`, `BlockResolventFloor`, `AlmostInvariantBasis`,
      `VerifiedLinearSystemResult`. The others: `Enclosure.λ::Any`,
      `PreconditionerResult.factorization::Any`, `PolyRange.diag::NamedTuple`,
      `SylvesterResolventResult.X` and `.R` (`Union{Nothing, …}`).
    - **Return type not inferred as concrete:** 47 calls of 42 functions out of 212 exported.
      `Any`: `verifyeigall(A, B)`, `rump_oishi_2024_triangular_bound`, `compute_beta_bound`.
      `Union{Float64, BigFloat}`: `upper_bound_L2_opnorm`, `collatz_upper_bound_L2_opnorm`,
      `sigma_min_upper`, `similarity_condition_number`. Untyped `Vector`: `evbox`, `gevbox`,
      `svdbox`. `Union{Nothing, BallVector}`: `interval_initial_enclosure`. A result type with
      free parameters: the linear-system and decomposition routines.
    - **The field that says whether the computation succeeded has five names:** `success`
      (12 structs), `converged` (5), `verified` (5), `certified` (2), and `spectrum_covered`,
      `is_regular` on their own.
    - **Failure is reported in different ways:** infinite radii, empty vectors, a `0 × 0`
      matrix (`VerifyEigAllResult.transformed`), `nothing`, a message string (`GEVResult`,
      `MiyajimaGEVResult`, `failure_reason`).
    - **Fields that carry nothing:** `DeterminantResult.computation_time` (zero, not
      measured) and `.tight` (always `false`); `RegularityResult.is_definitive` (always
      `false`).
    - **`Float64` written in:** `GEVResult`, `Enclosure` (which also has `radiuses`).
    - **Pairs with nearly the same fields:** `SchurGershgorinResult` and
      `Miyajima2014aSchurNewtonResult`; `Oishi2023Result` and `RumpOishi2024Result`.

19. **`verify_generalized_eigenpairs`, rewritten on 2026-10-12** after Miyajima, Ogita, Rump and
    Oishi (2010), Reliable Computing 14, 24-45 (in the library inbox as `MiyajimaEtAl2010`).
    What the old code did that the paper does not: it declined every problem with a cluster,
    where (39) of the paper gives the cluster the global bound δ; it did not check the
    hypothesis `ξ̂ᵢ < √gᵢ` of Theorem 7; δ and the ends of the intervals were formed in rounding
    to nearest; symmetry was tested on the midpoints up to `1e-10` with a warning; the order of
    the approximate eigenvalues was assumed; errors were caught and returned as a message;
    everything was `Float64`. Now: Algorithm 1 with Lemmas 2 and 3, Theorems 4, 5 and 7,
    Techniques 1, 3 and 4, for any floating-point type. Not reproduced: Technique 2 (an a
    priori rounding estimate, replaced by the ball product) and Theorem 10 (β from a Cholesky
    factor; `compute_beta_bound` uses an approximate inverse, which Remark 2 allows).
    **A second version, `method = :paper`** (the default for plain matrices), is Algorithm 1 as
    the paper computes it, for `Float32` and `Float64` matrices with radius zero: Theorem 10
    for β, the products `fl▽`, `fl△`, the a priori estimate (48), and δ by (26). One constant
    is not the paper's: its (51) has `γₙ` for the Cholesky backward error where Higham's
    Theorem 10.3, which it cites, has `γₙ₊₁`; the code uses `γₙ₊₁`.

20. **`certify_eigenpair`, `certify_eigenpairs`, audited on 2026-10-12.** The enclosure rests on
    a contraction argument that is now written, with its proof, in the docstring; the header
    used to cite a chapter of Kantorovich and Akilov as its authority. Found: the discriminant
    `(1 − q₀)² − 4cy` was evaluated with the rounding mode downward, which rounds the product
    `4cy` down and the discriminant up; `u` was divided by `uᴴṽ` where the conjugate is needed
    for `uᴴṽ = 1`, so a computed complex `u` gave a large residual (no wrong enclosure, a
    refusal). Added to `test_newton_kantorovich_eigenpair.jl`: tests against 256-bit
    eigendecompositions. The file existed, with 150 lines of tests from the commit that
    introduced the routine; on 2026-10-12 it was overwritten by the new tests, by a search
    that excluded its own name, and the statement that the routines had no test was wrong.
    The old tests were put back the same day, beside the new ones.
    GKWExperiments calls these routines in six scripts and one source file.

21. **`miyajima_gev_enclosure`, read on 2026-10-12** against Theorem 1 of Miyajima (2010),
    J. Comput. Appl. Math. 233, 2994-3004 (in the library): the statement and the code agree,
    and the individual radii are those of Corollary 3.2 of Miyajima (2014). Changed: the
    scalar steps use the directed operations and `_miyajima2014a_lem2_3` in place of six
    rounding-mode blocks, and the radii of a failed result are infinite, not an empty vector.
    The docstring claims Theorem 1, not Algorithm 1: the a priori bound of `‖R₂‖_∞`
    (Corollary 1) and Algorithms 2 to 7 of the paper are not what this routine runs; it
    encloses the residuals by ball products.

22. **`interval_gaussian_elimination`, rewritten on 2026-10-12** after Algorithms 5.9 and 5.10
    and Section 8.4.1 of Horáček (2019). Found: the field `singular`, documented as "provably
    singular", was set whenever a pivot contained zero, which proves nothing; the pivot was
    chosen by the absolute value of the lower end of the ball; the solution was a
    `Vector{Ball}` on success and a `BallVector` on failure; `L` held midpoints of the
    multipliers; failures were announced by `@warn`; the citation gave the thesis the year
    2012. Now: `certified` only, the pivot farthest from zero among those that do not contain
    it, `L` and `U` as ball matrices that contain the factors of every matrix of the ball,
    tests in rational arithmetic. The thesis says it selects the pivot "with the smallest
    mignitude"; the code takes the largest, which step 2 of Algorithm 5.9 allows.

23. **`compute_preconditioner(method = :ldlt)`, 2026-10-12:** it called `ldlt` on a dense
    symmetric matrix, which has no method, so the option failed on every input and fell back
    to LU with a warning. It now uses `bunchkaufman`.

24. **Exported functions with no test, 2026-10-12.** A search of the test directory for each of
    the 282 exported names found fifteen functions mentioned nowhere: `upper_bound_norm`,
    `ball_horner`, `pi_norm`, `compensated_terms2`, `inclusion_discs`, `det_cofactor`,
    `schur_sylvester_midpoint`, `clear_svd_cache!`, `set_svd_cache!`, `svd_cache_stats`,
    `ogita_iterations_for_precision`, `_certify_svd`, and three refinements in the MultiFloats
    extension (`ogita_svd_refine_multifloat`, `refine_schur_multifloat`,
    `refine_symmetric_eigen_multifloat`). The first eleven are now tested in
    `test_exported_primitives.jl`. `upper_bound_norm` computed `‖c‖ + ‖r‖` with
    `LinearAlgebra.norm` under the rounding mode; `probe_upper_bound_norm.jl` found it below
    the exact norm in none of 172,000 vectors of radius zero; it is now the norm of `|c| + r`
    by the directed operations, for `p = 1, 2, Inf`. Left without a test: `_certify_svd` (an
    exported name with an underscore, called by the companion packages) and the three
    MultiFloat refinements.

25. **The Schur refinements in Double64 and in a MultiFloat, 2026-10-12.**
    `probe_schur_refinement_inputs.jl` and `probe_multifloat_refinements.jl` (8 × 8, one matrix
    each, starting residual near 2e-15): `refine_schur_multifloat` returned residual 0.29 on a
    real Schur form with 2 × 2 blocks and 7e-16, 5e-16 on triangular input, each with
    `converged = true` (its loop re-orthogonalised `Q` and never solved the triangular
    equation); `refine_schur_double64` returned 1.4e215 on the real Schur form, threw a
    `MethodError` on a complex matrix, and 1.1e-16 on a real matrix with real eigenvalues; the
    BigFloat routine returned NaN on the real Schur form and 1e-77 on triangular input. Now the
    iteration of the package (Algorithm 4 of Bujanović, Kressner and Schröder) takes the high
    precision as a type, and the Double64, hybrid and MultiFloat routines call it. A Schur
    factor that is not upper triangular is an `ArgumentError` in all of them: the paper
    mentions an extension to the real Schur form and does not give it. Test:
    `test_schur_refinement_variants.jl`.
26. **`setprecision` without a block, 2026-10-12.** Eleven routines set the BigFloat precision
    by `setprecision(BigFloat, p)` and restored it in a `finally`. Inside a caller's
    `setprecision(BigFloat, q) do … end` that call has no effect (checked on Julia 1.12.7: the
    precision stays `q`), so the routine ran at the caller's precision and its own keyword was
    ignored. Found when `refine_schur_hybrid` did not converge inside a test's 512-bit block.
    All eleven now use the block form: `refine_schur_decomposition`, `refine_symmetric_eigen`,
    `ogita_svd_refine`, and eight in the DoubleFloats and MultiFloats extensions.

27. **`refine_schur_decomposition` by method, 2026-10-12.** Reading the paper against the code
    found that the code skipped the reordering of line 1 of Algorithm 4, took up to three
    initial Newton-Schulz steps where the paper takes one, and truncated the entries of `L` at
    `1e5` where the paper's Example 4 proposes `1e-5` (the code cited an "Example 10"; the
    paper has four); the citation gave volume 95 (2024), which is 92 (2023). Now, following
    the rule of `verifyeigall` that each algorithm is a function named for its paper and a
    deviation is visible in the name: `:bujanovicetal2023` (`_bujanovicetal2023_alg4`, the
    algorithm as published, with the reordering along a random line), `:bujanovicetal2023truncated`
    (the modification of Example 4), and `:bujanovicetal2023unordered` (what the routine did,
    the default, because `CertifScripts` orders the Schur form before refining it). The same
    keyword is on the Double64, hybrid and MultiFloat routines. Not the paper's in any of the
    three: `nmin = 32` in the recursive solve of the triangular equation (the paper's
    experiments use 4); the real Schur form is refused.

28. **To do: the Schur refinement on the MatrixMarket matrices of the paper.** Example 4 of
    Bujanović, Kressner and Schröder (2023) runs Algorithm 4 on matrices from MatrixMarket,
    double precision refined to quadruple. Its Table 3, for the real Schur form: `olm100`,
    `tols90`, `tub100`, `bwm200`, `olm500`, `olm1000`, `tub1000`, `bwm2000`, `dw2048` converge
    in 3 or 4 iterations, with `‖I − Q*Q‖_F` between 2.7e-32 and 1.3e-31 and
    `‖stril(Q*AQ)‖_F/‖A‖_F` between 2.7e-34 and 3.5e-33; `rdb200`, `rdb200l`, `tols1090`,
    `rdb1250` do not converge. Its Table 4, for the complex Schur form with the truncation at
    1e-5: those four converge in 4, 4, 5 and 4 iterations.
    To run: the three methods of `refine_schur_decomposition` on these thirteen matrices, with
    `Double64` as the high precision (the closest to the paper's quadruple) and with BigFloat,
    and compare with the two tables: which converge, in how many iterations, the two defects.
    The interest is `:bujanovicetal2023truncated`, which on the construction of Example 3
    (`probe_schur_refinement_methods.jl`, size 150, one matrix per row) did not converge in
    three of four cases where the untruncated algorithm did; the tables are where the paper
    reports that the truncation helps. Not comparable as they stand: Table 3 is for the real
    Schur form, which the routines here refuse, so the first nine would be run on the complex
    form. `rdb200` is already in the catalogue of the companion package's benchmarks; the
    others are to be fetched (data go to `~/ResearchData`, not into the repository).

29. **More copies in the extensions, 2026-10-12.** `refine_symmetric_eigen_double64`, `_hybrid`
    and `_multifloat` each had their own loop of RefSyEv; they now call the iteration of the
    package with the high precision as a type, as the Schur ones do. `ogita_svd_refine_multifloat`
    had its own copy of the Ogita-Aishima iteration; it calls the package's, as the Double64
    one did. Removed from `DoubleFloatsExt.jl`: `oishi_2023_solve_double64` and
    `oishi_2023_sigma_min_bound_fast`, which were plain functions of the extension module (not
    methods of a function of the package, so not reachable by name), with no caller and no
    test. The first returned a ball matrix described as an enclosure of the solution of
    `AX = B` from norms evaluated in rounding to nearest and a bound of `‖A⁻¹‖` that is not
    one; the second computed two solves, discarded them and called `oishi_2023_sigma_min_bound`.
    In all, 410 and 145 lines removed. `docs/src/literate/benchmark_warmstart_circle.tex` still
    names the second.

30. **`BallMatrix(Float64, A)` for a matrix of Arb numbers, 2026-10-12.** The radius was formed
    as `Float64(midpoint) + Float64(radius)` of an Arb sum under the rounding mode upward; the
    conversion of an Arb number to `Float64` rounds to nearest whatever the mode.
    `probe_arb_conversion.jl`: 9939 and 10053 of 20000 converted balls did not contain their Arb
    ball (Arb precision 128 and 512), by at most 1.1e-16 of the radius; the BigFloat conversion
    had none at 64, 128, 256 and 1024 bits. Now each converted number is moved to the next
    float when it fell below the exact one, and the complex radius is `hi √(1 + (lo/hi)²)`, the
    squares of tiny radii having underflowed in the direct formula. Test: containment at 4096
    bits, in `test_arbnumerics_ext.jl`.

31. **`fft` of a ball vector or matrix, 2026-10-12.** Read against Brisebarre, Muller and Picot
    (ARITH 2023; in the library). Found: the bound of the paper is proved for the radix-2
    transform of its Fig. 1 with the roots of unity rounded to nearest, and the midpoint was
    computed by FFTW, another algorithm, for which the paper proves nothing; a length that is
    not a power of two gave a warning and a result with the same radius formula; the constant
    was written with `(1 + u)` where the paper's (4) has `(1 + u)ⁿ`, with `u` taken as `eps`
    (twice the paper's unit roundoff), which made the value larger than the paper's except at
    `n = 2`, where it was smaller by a relative `u²`. Now the midpoint is computed by the
    algorithm of Fig. 1 (`_bmp_radix2_fft`), the constant is (4) with `u = eps/2`, bounded
    through `e^S − 1 ≤ S/(1 − S/2)` so that rounding `1 + u` up does not double it, and a
    length that is not a power of two is an `ArgumentError`. The bound along both dimensions
    of a matrix is derived in the docstring from the paper's bound in the Euclidean norm.
    `probe_fft_timing.jl` on nighthawk, one thread, complex `Float64`, second call: 0.9 ms
    against 0.72 ms for FFTW at `N = 2¹⁶`, 33 ms against 8.3 ms at `N = 2²⁰`; the first call
    at a size builds the roots in BigFloat (0.09 s and 1.3 s). RigorousInvariantMeasures calls
    this through its `interval_fft`; its tests were not run against this change.

32. **`ball_fft` in the package, 2026-10-12** (Isaia: keep FFTW as an extension, move the
    transform in). `src/fft/ball_fft.jl` has the radix-2 transform and the bound, with the
    methods `:brisebarreetal2023` and `:brisebarreetal2023fftw` (midpoint by FFTW, the radius
    as a hypothesis; needs the extension). The roots of unity of order 2¹⁶ for Float64 and
    Float32 are computed at precompilation and serve every smaller power of two.
    The audit of FFTW (log of 2026-09-04) and the note on the split-radix bound are in
    private repositories and are not on the arXiv; they are not copied or cited here (Isaia,
    2026-10-12). The README says that the audit is in progress and that the FFTW method is
    not a verified enclosure. To do when the audit is public: evaluate the bound that holds
    for FFTW's algorithms, with its roots, in place of the radix-2 constant, and cite it.

33. **Every exported function on real, complex and BigFloat balls, 2026-10-12.**
    `audit_input_types.jl` (scratch, nighthawk) calls each exported function that has a method
    for a ball matrix, alone, with a second one, or with a ball vector, on 4 × 4 inputs of the
    four kinds Float64, complex Float64, BigFloat, complex BigFloat: 340 calls per shape, for
    the shapes square, 6 × 4 and upper triangular. It reports the calls that throw and the
    calls on BigFloat input whose result holds Float64 numbers. Fixed from it:
    - `adaptive_ogita_svd` rounded a matrix given in BigFloat to the working precision of
      each level without enlarging the radii (at 53 bits, 12 of the 16 entries of a random
      matrix were outside the converted ball), so what it certified at Float64 was not about
      the matrix given. `_convert_to_precision` now puts the rounding error into the radii.
      The first level is the arithmetic of the matrix given (Isaia, 2026-10-12: a BigFloat
      matrix should have a BigFloat result), Float64 or BigFloat at the precision of the
      midpoints, and no level is below `max_precision_bits` when the matrix is at it.
      On a matrix with radii the precision is adapted on the midpoint and each enclosure is
      widened by a bound of `‖rad(A)‖₂` (Isaia, 2026-10-12: the singular values are Lipschitz
      in the 2-norm, and there was a helper for it): `_weyl_widening`, taken out of
      `_miyajima2014_thm10_weyl`, which did the same step with two `setrounding` blocks.
      Before, the radii went through the residual at every level and the driver ran to its
      limit when the tolerance was below what they impose.
      Comparison of 9666e4e with ac017d8 (`regression_digest.jl`, scratch, nighthawk: 131
      calls on fixed 8 × 8 and 10 × 6 inputs, each revision run twice with identical
      output): besides the calls changed on purpose, `oishi_2023_optimal_block_size` and
      `rump_oishi_2024_optimal_block_size` on the real ball changed in the last digit, and
      the change is at ac017d8, in `_miyajima2014_thm10_weyl`. On the 7 × 7 block concerned,
      the widened ball of σ₇ at 76e943c has its upper end below `mid + rad + delta` of the
      inner ball computed in BigFloat; at ac017d8 both ends enclose it. One input; the
      cause in the `setrounding` version was not looked for.
    - `compute_preconditioner` threw on a complex matrix; `PreconditionerResult{CT, T, F}`.
    - `oishi_2023_sigma_min_bound`, `rump_oishi_2024_sigma_min_bound` and the two block size
      searches threw on a complex matrix; the lower bound of `|d|` on the diagonal was
      rounded to nearest for a complex `d` and is rounded down.
    - `rigorous_generalized_eigenvalues` and `gevbox` threw in BigFloat for want of
      `eigen(A, B)`; they take the approximation that `verifyeigall(A, B)` uses.
    - `triangular_inverse_inf/one/two_norm_bound` on a ball matrix gave a `DomainError` or
      "U must be upper triangular" for a triangular one; they have a method for ball
      matrices, with the proof unchanged.
    Left as they are: the routines stated for real interval matrices refuse complex ones
    with an `ArgumentError` (Horáček's methods, `det_gershgorin`,
    `verified_linear_solve_hmatrix`, Theorem 11.13); the routines for plain matrices given a
    ball matrix (`polar_newton`, `polar_newton_schulz`, `polar_qdwh`, `cholesky_qr2`,
    `triangular_eigenvectors`, `verified_lu`, `verified_qr`, `verified_cholesky`,
    `verified_polar`, `verified_takagi` and their `_gla`, `_double64` and `_multifloat`
    forms) failed with "This conversion breaks rigour" or a `MethodError`; they throw an
    `ArgumentError` that says so (Isaia, 2026-10-12).
    **To do** (Isaia, 2026-10-12): some of these can probably be extended to ball matrices
    by treating midpoint and radius separately, the factorisation computed on the midpoint
    and the radii carried into the residual that is bounded. Not examined routine by
    routine; `verifysvdall` and `rigorous_svd` already take a ball matrix in this way. Not covered by the sweep: functions whose first argument
    is not a ball matrix, keyword options, and whether the results are correct, which the
    sweep does not check.
    The nineteen `@assert` of the package are explicit errors now, since an assertion may be
    compiled out: the defect below 1 in `rigorous_eigenvalues` and
    `rigorous_generalized_eigenvalues`, the triangular matrix of the four substitutions, the
    sizes and indices of the Riesz projections (`ArgumentError`, `DimensionMismatch`).

34. **Rounding-mode blocks are not reliable; the SVD theorems no longer use them, 2026-10-12.**
    Base of Julia 1.12.7 has no `setrounding(Float64, mode)`; the package takes it from
    SetRounding.jl, which calls `fesetround`. Measured on nighthawk, Julia 1.12.7:
    - `setrounding_patterns3.jl`: a function with `setrounding(T, RoundDown) do s.c - s.r - δ end`
      inside a loop or comprehension over a vector of structs gave a result above the exact
      value in 106,249 of 200,000 cases, and the mirror block with RoundUp a result below it
      in the same number (forms H1 to H5, including a single block per iteration). With each
      block around a whole comprehension (H6), or one RoundUp block with the lower end
      written as −((−c + r) + δ) (H7), 0 of 200,000.
    - `setrounding_patterns.jl`: six other scalar forms, 0 failures in 200,000 each.
    - a real matrix product inside a block, with the package loaded (its OpenBLAS with
      consistent rounding across threads, 64 threads): 0 entries on the wrong side for
      n = 64, 300, 1000. Without the package, stock OpenBLAS on 64 threads, n = 300: 42,542
      and 42,594 of 90,000; n = 8 and 64: 0.
    - a complex matrix product inside a block is not a bound of the real and imaginary
      parts (about half the entries on the wrong side at n = 64 and 300); whether the package
      relies on one anywhere was not checked.
    Rewritten with the directed operations (8e76e3d): Theorems 4, 7, 10, 11 of Miyajima
    (2014), Theorem 3.1 of Rump (2011), `_compute_svd_bounds`, `_ball_from_bounds`. Three
    formulas were wrong whatever the rounding: in Theorems 10 and 11 a quotient was bounded
    below with its denominator rounded down and above with it rounded up; in Rump's theorem
    the modulus of a complex diagonal entry was rounded to nearest, and a negative numerator
    was divided by the upper bound of the denominator. Comparison of ac017d8 with 8e76e3d
    (`regression_digest.jl`): 29 of 131 lines differ, all in `svdbox`, `rigorous_svd`,
    `svd_bound_L2_opnorm_inverse` and the σ_min bounds built on them; in each the sum of the
    radii is larger or the lower bound smaller, by amounts of the order of 1e-15 relative.
    Converted since (4a84a38, bee49b1 and the commit of this entry): the block
    diagonalisation and Gershgorin bounds, Algorithms 1 and 2 of Miyajima (2014a),
    `verifyeigall`, the 1- and ∞-norm bounds and `upper_abs` in arithmetics other than
    Float32 and Float64, the sums of ball vectors, `bdd_R2` (where `n u/(1 − n u)` was
    evaluated inside a RoundUp block, so with the denominator rounded up). The matrix
    products that stay under a rounding mode in those files go through `_product_up`,
    beside `_product_down_up`. `proceduresMiyajima2010.jl` is removed: its two functions
    `_encR1ci` and `_encR1ccr` used a type `T` that is not defined in them, and nothing
    called them. `bdd_R2` was then checked against the paper: it is inequality (8)
    of Corollary 1 of Miyajima (2010), with `eps` for the unit roundoff and γₙ for γₙ₋₁, both
    larger. The two terms that the corollary states as exact quantities,
    `‖(|Y_r| + |Y_i|) Z_r e‖_∞` and `‖|Y| |Z_c| e‖_∞`, were products formed in round-to-nearest
    with no account of their rounding; the γ terms of (8) pay for the product `Y Z_c` only.
    They are evaluated with every operation rounded up. The two functions removed were the
    paper's Procedures 1 and 2 for the enclosure of R₁.
    RoundingEmulator exports the ten scalar operations and no `setrounding` (read in its
    source, version 0.2.1), so a matrix product cannot be converted in the same way: to
    decide (Isaia), whether the default `*` moves from `MMul4` to the error-free route
    `mmul_ogita_rump_oishi_2005`, and whether its radius is then bounded in round-to-nearest
    as the note in `ogita_rump_oishi.jl` proposes, which needs the error model of gemm and
    an underflow term first.
    **To do:** the remaining lines (145 at 8aa59dc) with `setrounding` in src and ext, scalar blocks first
    (`schur_gershgorin.jl`, `miyajima_2014a.jl`, `rump_verifyeigall.jl`, `rump_lange_2023.jl`,
    `gram_transform.jl`, `rigorous_residual.jl`, `rigorous_opnorm_bounds.jl`, the Miyajima
    procedures), then the matrix products of `types/MMul` and `types/matrix.jl`, where the
    question is the complex case and broadcasts inside a block.


35. **Product bounds without a rounding mode, by Dot2Err, 2026-10-12** (Isaia: the trick of
    RoundingEmulator, a value with its error term rounded to either side, applied to a product
    kept in two terms). `_dot2err_ogita_rump_oishi_2005(x, y)` is Algorithm 5.8 of Ogita, Rump
    and Oishi (2005) with the `res`, `err` of Corollary 5.9. `_product_down_up_dot2err(A, B)`
    keeps each entry as the unevaluated `p + s` of that algorithm, bounds its distance from
    the exact entry by `β = fl(δe + 3eta/eps)`, which the proof of the corollary shows with
    underflow included, and rounds `p + (s − β)` and `p + (s + β)` down and up with
    `add_down`, `sub_down`, `add_up`. `Float32` and `Float64` only. Nothing calls it yet:
    `_product_down_up` and `_product_up` still evaluate the BLAS product under `RoundDown`
    and `RoundUp`.
    Timing, `time_product_down_up.jl` on nighthawk at 53a5a3a, 16 Julia threads, 64 BLAS
    threads, `A`, `B` real `k × k` with standard normal entries, minimum of 3 runs in seconds;
    "split" is `_two_term_product_sum` (the Lange and Rump splitting on the BLAS) with the
    same directed sums of its two terms and bound; width is the median of `(up − down)/|up|`:

    | k | BLAS, two modes | Dot2Err | split | width BLAS | width Dot2Err | width split |
    |---|---|---|---|---|---|---|
    | 100 | 0.000050 | 0.00028 | 0.00071 | 1.1e-14 | 1.6e-16 | 1.6e-16 |
    | 200 | 0.00018 | 0.0010 | 0.0027 | 2.2e-14 | 1.6e-16 | 1.6e-16 |
    | 400 | 0.00045 | 0.0060 | 0.011 | 4.4e-14 | 1.6e-16 | 1.6e-16 |
    | 800 | 0.0017 | 0.046 | 0.041 | 6.3e-14 | 1.6e-16 | 1.6e-16 |
    | 1600 | 0.0082 | 0.30 | 0.19 | 9.7e-14 | 1.6e-16 | 1.6e-16 |

    At every `k` each Dot2Err interval lay inside the interval of the two BLAS products.
    Not measured: complex factors, non-negative factors (the radius products, where only the
    upper bound is used), one thread, sizes above 1600, matrices with cancellation.
    Suite at b06c31d on nighthawk: 183,039 passed, none failed (`rigour_suite59.out`).
    **Decided (Isaia, 2026-10-12):** `MMul4` is the default, called by `*`, and the code uses
    `*` only; the Dot2Err route stays as an alternative that nothing calls.

36. **The MMul routines read against their sources, 2026-10-12.** Sources: S. M. Rump, Fast
    interval matrix multiplication, Numer. Algorithms 61 (2012) 1-34 (Theorem 2.1, Algorithms
    4.3, 4.5, 4.7, 4.9); Revol and Théveny (2013), Algorithms 1 and 2; Théveny's thesis (2014).
    `MMul4` is Algorithm 4.5 with the conversion to midpoint and radius. `MMul3` is Algorithm
    4.7 (and 4.3 for a point factor) with `(k + 2)·2⁻⁵²` where the paper has `(k + 2)·2⁻⁵³`.
    `MMul5` is Algorithm 4.9; its error bound had half of the paper's `realmin`, and it threw
    on every `Float32` input (a `Float64` literal in the radius): both corrected at 59422a8,
    suite 183,048 passed, none failed (`rigour_suite60.out`). `MMul3` and `MMul5` accepted
    complex ball matrices, for which the algorithms are not stated: restricted to real at
    6b7dc28, and both check the paper's bound on the inner dimension. Enclosure check before
    the changes, `check_mmul.jl` on nighthawk: 0 entries outside the ball for `MMul3`, `MMul4`,
    `MMul5` and `*` in `Float64` (651,520 each), `MMul3`, `MMul4`, `*` in `Float32` (162,880
    each), `MMul3`, `MMul5`, `*` in `ComplexF64` (651,520 each).
    The matrix of `rounding_test`, run when the module is loaded, is the example of Section
    5.1.3 of the thesis; the INTLAB page that gives CONSISTENT_FPCSR as a remedy is cited in
    the README and beside the import.
    `blas_rounding_cases.jl` on nighthawk: 0 entries of `_product_down_up` on the wrong side
    for dense `Float64` and `Float32`, n from 1 to 150, and for views, adjoint, transpose,
    triangular, diagonal, symmetric and rectangular factors at n = 2, 3, 8, 64.
    `MMul2`, a stub that warned "Not Implemented" (thesis, Algorithm 9), is removed (Isaia,
    2026-10-12).
    **Open:** `MMul3` and `MMul5` rest on the products `mA·mB` and `|mA|·|mB|` being computed
    in the same order (Remark 1 of Rump's paper).

37. **Routes to the real ball product compared, 2026-10-12** (Isaia: whether to substitute the
    default with the prodK alternative; the centre and the radius may have different
    precision). `ball_product_routes.jl` on nighthawk at c067ed2, 16 Julia threads, 64 BLAS
    threads, real `k × k` factors with standard normal midpoints and radius `relrad·|mid|`;
    minimum of 3 runs (2 at 3200), seconds; the radius is the median of `rad/|mid|` of the
    product. "mixed" is `MMul4` with the two radius products in `Float32` (Ozaki, Ogita,
    Bünger and Oishi, NOLTA 6 (2015) 364-376, doi 10.1587/nolta.6.364, Algorithm 4), written
    in the script and not in the package; "split1", "split2" take the midpoint from
    `_lange_rump_two_terms` with one and two splitting points and the radius products of
    `MMul4`; "dot2" is `mmul_ogita_rump_oishi_2005`.

    | k | gemm | mmul4 | mixed | dot2 | split1 | split2 |
    |---|---|---|---|---|---|---|
    | 100 | 0.000036 | 0.00015 | 0.00018 | 0.00069 | 0.00027 | 0.00037 |
    | 400 | 0.00028 | 0.0016 | 0.0021 | 0.013 | 0.0030 | 0.0061 |
    | 800 | 0.00089 | 0.013 | 0.0095 | 0.061 | 0.011 | 0.016 |
    | 1600 | 0.0036 | 0.044 | 0.044 | 0.42 | 0.071 | 0.11 |
    | 3200 | 0.024 | 0.25 | 0.34 | not run | 0.53 | 0.72 |

    (times at `relrad = 0`; at 800 and 1600 `mmul4` took 0.0065 and 0.027 with `relrad > 0`).
    Median relative radius of the product:

    | k | relrad | mmul4 | dot2 | split1 | split2 |
    |---|---|---|---|---|---|
    | 100 | 0 | 5.5e-15 | 1.1e-16 | 3.9e-17 | 3.8e-17 |
    | 1600 | 0 | 4.9e-14 | 1.1e-16 | 4.7e-17 | 3.8e-17 |
    | 3200 | 0 | 7.5e-14 | not run | 7.0e-17 | 3.8e-17 |
    | 1600 | 1e-15 | 1.25e-13 | 7.56e-14 | 7.55e-14 | 7.55e-14 |
    | 1600 | 1e-12 | 7.5538e-11 | 7.5489e-11 | 7.5489e-11 | 7.5489e-11 |
    | 1600 | 1e-8 | 7.54360e-7 | 7.54360e-7 | 7.54360e-7 | 7.54360e-7 |

    The radius of "mixed" agreed with that of `mmul4` to 5 digits in every row. Not measured:
    complex factors, a chain of products (where the output radius of one is the input radius
    of the next), the certification runs of the paper with either route.
    **Decided (Isaia, 2026-10-12):** the package keeps `MMul4` on the OpenBLAS built with
    CONSISTENT_FPCSR for the time being; the splitting, Dot2 and mixed-precision routes are
    not substituted for it.

38. **Decisions of 2026-10-12 (Isaia), carried out.**
    - `MMul2` removed; `rigorous_residual.jl` stays removed.
    - `consolidate-rp` fast-forwarded to f50b4e8, where the suite (182,957 passed,
      `rigour_suite64.out`) and the companion's tests (300 of 300) passed. No tag.
    - `refine_schur_decomposition` and its Double64, hybrid and MultiFloat forms: the default
      method is `:bujanovicetal2023`, the algorithm as published; `CertifScripts`, the
      certification routine and the second stage of the hybrid routine pass
      `:bujanovicetal2023unordered`.
    - `_certify_svd` is `certify_svd`, with a docstring.
    - The field that says the proof went through is `certified`: `RegularityResult.is_regular`
      and `SylvesterResolventResult.precomputation_success` renamed. Kept, because they say
      something else: `VerifyEigAllResult.spectrum_covered`, `PreconditionerResult.success`.
    - `verified_lu`, `verified_qr`, `verified_cholesky`, `verified_polar`, `verified_takagi`,
      `verified_svd` take the arithmetic of the enclosure as a leading type argument,
      `verified_lu(Float64, A)`; without it, BigFloat. `use_bigfloat`, `use_double_precision`
      and the names with `_gla`, `_double64`, `_multifloat` are removed (`oracle = Double64`,
      `oracle = Float64x4` say the same); `verified_svd_gla` is `verified_svd`.
      `gram_transform` takes the keyword `arithmetic` in place of `use_bigfloat`.
    - `rigorous_eigenvalues`, `evbox`, `rigorous_generalized_eigenvalues`, `gevbox`: a ball
      matrix always gives discs with complex centres. They gave real balls when the computed
      eigenvalues of a real matrix were real, and a real ball is an interval, while a matrix
      inside the input ball may have complex eigenvalues. Real intervals are returned for a
      point matrix that is `Hermitian`, or real `Symmetric`, by its type. A ball matrix is
      an open set of matrices and is not Hermitian: no such method.
    - The dependent packages are audited later: RigorousPseudospectra's tests pass at 3da1547
      (300 of 300) with the renames; RigorousInvariantMeasures and GKWExperiments not run.
    **Open in the return types** (`audit_return_types.jl`, rerun at f50b4e8: no struct field
    is flagged as not concrete; 22 calls with a return type that is not inferred): the type
    depends on a keyword in `rigorous_svd` (`apply_vbd`) and `compute_preconditioner`
    (`method`), and on the data in `certify_eigenpairs`; not traced: `rigorous_block_schur`,
    `schur_gershgorin_enclosure`, `triangular_sylvester_miyajima_enclosure`,
    `adaptive_ogita_svd`. To rerun after the type argument and the eigenvalue dispatch.
    `evbox(Hermitian(M))` returns a `Vector{Ball{Float64, Float64}}` and is inferred as `Vector`
    (the test asserted the inference and errored at 0aeeef3; the assertion is removed, the
    type of the value is still tested).

39. **RigorousInvariantMeasures against this branch, 2026-10-12.** `run_rim_tests.sh` on
    nighthawk, BallArithmetic at 96dff9d (suite 182,949 passed, none failed,
    `rigour_suite68.out`; RigorousPseudospectra 300 of 300), RIM at afd7284 on its branch
    `ballarithmetic-0.4` with its uncommitted changes, on a local clone without
    `test/Manifest.toml`, which pins the registered BallArithmetic 0.3.0 (a first run with it
    loaded 0.3.0 and says nothing about this branch).
    RIM as it is: 1,545 passed, 6 errored. Five from `res.verified` on a `KrawczykResult`
    (`src/Basis/NewChebyshev.jl:739`), now `certified`; one from an FFT of 9 points
    (`ext/FFTWExt/FourierObservables.jl`), which `ball_fft` refuses.
    With the edits of `rim_patch.py`: 1,582 passed, none failed (`rim_tests5.nighthawk.out`).
    The edits: `res.certified`; `verified_cholesky(Float64, …)` for
    `verified_cholesky(…; use_bigfloat = false)` at three places, with `chol.certified` for
    `chol.success` (also in `test/TestChebyshev.jl`); and in `_fourier_coeffs_no_aliasing` the
    transform on a grid of `nextpow(2, M)` points, keeping the frequencies
    0, 1, …, K, −K, …, −1 (Isaia: RIM always takes a bigger FFT and saves those). They are in
    `~/scratch/rim_ballarithmetic_names.patch` on nighthawk and are not applied to RIM.
    Not looked at: `interval_fft` in RIM says FFTW computes the midpoints, and the default
    method of `fft(::BallVector)` is now the package's own transform; the compat entry
    `BallArithmetic = "0.3, 0.4"`; GKWExperiments.

40. **The `hermitian` keywords, 2026-10-12** (after Isaia's remark that a ball matrix with
    positive radii is an open set of matrices and cannot be Hermitian).
    - `schur_gershgorin_enclosure(A; hermitian = true)` and `rigorous_block_schur(A; hermitian =
      true)` returned real intervals for a ball matrix under a docstring that states the
      spectrum of every matrix of the ball. `hermitian_keyword_cases.jl` on nighthawk at
      96dff9d: for the ball of centre `[1 0; 0 1+1e-8]` and radius `1e-6` both returned
      `1 ± 2.0e-6` and `1.00000001 ± 2.0e-6`; the matrix `[1 1e-6; −1e-6 1+1e-8]` of that ball
      has the eigenvalues `1.000000005 ± 1.0e-6 i`. Now (c99e81f): a `BallMatrix` gives discs of
      the complex plane and has no keyword; `Hermitian(M)`, or `Symmetric(M)` with `M` real,
      for a point matrix gives real intervals; so does a real ball matrix stored as `Diagonal`
      in midpoint and radius, every matrix of which is a real diagonal matrix. That last type
      is what the singular value routines pass: after the certification, with `apply_vbd`,
      they group the enclosures by calling the routine on `Σ*Σ`, `Σ` the diagonal ball matrix
      of the singular value enclosures. No Bool selects the algorithm: the kind of enclosure
      is a type (`_ComplexDiscs`, `_RealIntervals`) obtained from the type of the argument.
      (A first version of this change, c99e81f, added an unexported routine "for the Hermitian
      matrices contained in the ball", with a docstring saying that the singular value
      routines enclose the eigenvalues of `S*S` for `S` in a ball matrix. They do not: Isaia,
      "the svd routines certify a numerical svd, they never build S'*S". The routine is
      removed.)
    - `certify_eigenpairs(B; hermitian = true)` threw a `MethodError` for a complex Hermitian
      `B`; corrected. The keyword only selected the eigendecomposition the approximations come
      from; since 37d796c it is removed and the Hermitian eigendecomposition is taken whenever
      the midpoint is Hermitian (Isaia, 2026-10-13: "we are certifying an oracle. If the
      midpoint is hermitian, we certify the best possible oracle"). The certification is for
      every matrix of the ball.
    - Read, not changed: `compute_spectral_projector_hermitian` (the Hermitian midpoint gives
      the approximate basis; the statement is the general one).
    - **To do (the last `hermitian` keyword of the package):** in `riesz_projections.jl` the keyword `hermitian` of `project_onto_eigenspace`,
      `compute_eigenspace_projector` and `verified_project_onto_eigenspace` means that the
      projection is the orthogonal one onto the span of the chosen columns, and
      `verified_project_onto_eigenspace(...; hermitian = false)` throws "not yet implemented":
      an exposed option that is not implemented, and a keyword whose name says something
      else than what it does.
      **Written, 2026-10-13, not yet run:** `verified_project_onto_eigenspace` has no keyword
      and no unimplemented branch; its docstring says what it encloses, the orthogonal
      projection onto the span of the chosen columns for every member of the two balls, and
      that this is the Riesz projection for a normal matrix only. The test samples members.
      **Removed, 2026-10-13** (Isaia: "the verified routines are correctly referenced in
      literature? If not, remove them"): `verified_project_onto_eigenspace` cited no result
      for its enclosure; the references of the file (Kato; Stewart and Sun; Golub and Van
      Loan) are on the routines that are not verified and are for the definition of the
      projection. No caller in `src`, `ext`, or in the clones under `~/Code` and `~/Write` on
      bemtivi. The routine, its export, its API entry and its tests are deleted.
      **The rest of the file removed too, 2026-10-13** (Isaia: "remove them also"):
      `project_onto_eigenspace`, `project_onto_schur_subspace`,
      `compute_eigenspace_projector`, `compute_schur_projector`, floating-point
      approximations with no bound and no caller; `src/eigenvalues/riesz_projections.jl`,
      its test file, its exports and its entries in the docs and the README are deleted. The
      question below no longer arises.
      **Was open:** in the two routines that are not verified,
      `project_onto_eigenspace` and `compute_eigenspace_projector`, the keyword `hermitian`
      still selects the orthogonal projection against the oblique one built from the left
      eigenvectors; both branches are implemented.
    - Dependents: GKWExperiments calls `rigorous_block_schur(A; hermitian = hermitian)`
      (`src/EigenspaceCertification.jl:249`) and exports `miyajima_vbd`; its compat entry is
      `BallArithmetic = "0.1.1"`. Not run.

41. **Sampling check of the spectral enclosures, 2026-10-13.** `check_spectral_enclosures.jl`
    on nighthawk, for `schur_gershgorin_enclosure`, `miyajima2014a_schurnewton`,
    `rigorous_block_schur` and `rigorous_block_schur(vbd_method = :nsd)`. Ball matrices with
    midpoint general, symmetric, symmetric with two clusters, triangular with a clustered
    diagonal, complex general, complex Hermitian; n = 3, 6, 12; radius 0, 1e-12, 1e-8, 1e-4,
    1e-2 times the largest entry; 12 ball matrices for each, 5 matrices of each ball (for a
    real ball one with the skew sign pattern, two corners, two random), eigenvalues at 256 bits.
    At 7e1cfce and again at 21bb3ec, the same counts: 4,320 ball matrices, none threw, none
    with an infinite radius; 151,200 eigenvalues, 0 outside the union of `cluster_intervals`,
    0 outside the union of the discs of `block_enclosure`; 21,600 checks of the groups of
    overlapping block discs, 0 with a number of eigenvalues other than the summed
    multiplicities; 680,400 entries of `Q⁻¹XQ`, 0 outside `similar`. With every radius of
    `cluster_intervals` divided by 10, 9,706 of the 151,200 eigenvalues are outside at 7e1cfce
    and 9,782 at 21bb3ec.
    By type, at 21bb3ec: `Hermitian(M)` real and complex, `Symmetric(M)`, and a real ball
    matrix stored as `Diagonal`, 1,440 cases for the first and third routine: every result
    has real intervals, 20,160 eigenvalues, 0 outside.
    Not covered: n above 12, radii above 1e-2, defective midpoints, BigFloat input, and the
    quantities of the results that are norms (`remainder_norm`, `projected_residual_norm`,
    `perturbation_norm`).
    Since 21bb3ec the basis is chosen from the midpoint (eigenvectors when it is Hermitian)
    and the kind of enclosure from the type.
    Suite at 21bb3ec: 182,986 passed, none failed (`rigour_suite73.out`); RigorousPseudospectra
    300 of 300.

42. **Citations against Crossref (step 5), 2026-10-13.** `check_bib.py` and
    `check_docstring_dois.py` (scratch scripts, run on bemtivi).
    - The 22 DOIs written in `src`, `ext`, the README and the documentation pages each resolve
      to the paper named beside them, with the same journal, volume, pages and year.
    - `docs/src/refs.bib`, which the references page prints in full, had 29 entries. Wrong:
      `RumpOishi2001` gave Rump and Oishi as authors of the paper on rounding mode controlled
      computation, which is by Oishi alone, with a DOI that does not exist (now `Oishi2001`);
      `BujanovicKressnerSchroder2022` gave volume 95 (2024), it is 92(1) (2023), with a DOI
      that does not exist; `MiyajimaSylvester2013`, `MartinPetersWilkinson1971` and
      `KnightKaiser1979` carried the DOI of another paper; `RumpOgita2024Verified` had the
      title "Error-free transformations and verified matrix decompositions" in Numerical
      Algorithms, the paper is "Verified Error Bounds for Matrix Decompositions", SIAM J.
      Matrix Anal. Appl. 45(4) (2024) 2155-2183 (now `RumpOgita2024`);
      `NakatsukasaBaiGygi2010` said SIAM J. Sci. Comput. 32, it is SIAM J. Matrix Anal. Appl.
      31(5); `Yamamoto2015` said Numer. Math. 131, 297-322, it is Electron. Trans. Numer.
      Anal. 44 (2015) 306-326 (checked on the journal's page; now `YamamotoEtAl2015`);
      `Higham1996` was the second edition, of 2002 (now `Higham2002`);
      `FukayaKannanBallardDongata2020`, "LU-Cholesky QR Algorithms for Thin QR Decomposition"
      by Fukaya, Kannan, Ballard and Dongarra in SIAM J. Sci. Comput. 42, matches no paper:
      that title is by Terao, Ozaki and Ogita, Parallel Computing 92 (2020) 102571. The last
      one, `MartinPetersWilkinson1971` and `KnightKaiser1979` are cited nowhere in the package
      and are removed; the others are corrected, and DOIs added to `Rump2010` and
      `BrisebarreMullerPicot2023`. 26 entries; those with a DOI agree with Crossref.
    - `rigorous_eigenvalues` and `rigorous_generalized_eigenvalues` cited "Miyajima, JCAM 246,
      9 (2012)", which is no paper, under the key of Miyajima (2012), J. Comput. Appl. Math.
      236, 2545-2552. That paper gives one radius for each eigenvalue (its Theorem 3). What
      the routines compute, one common radius `‖R₁‖∞/(1 − ‖R₂‖∞)`, is Theorem 1 of Miyajima
      (2010), J. Comput. Appl. Math. 233, 2994-3004 (read in the library copy); the docstrings
      now cite and state it.
    - Collatz (1942) now carries its DOI.
    - `types/MMul/oishi_mmul.jl` read against Algorithms 2 to 7 of Miyajima (2010) in the
      library copy: `_cprod`, `_ccr`, `_ccrprod`, `_cr`, `_iprod`, `_ciprod` are those
      algorithms line by line (in the printed Algorithm 2 the real part reads
      `F_r G_r + (−F_r) G_i` in the text extracted from the PDF; the code has `(−F_i) G_i`);
      `_ccrprod_prime`, `_iprod_right`, `_ciprod_prime` are the same with the factors in the
      other order and are ours. The docstring of `_ccr` described a centre rounded to nearest
      and a radius from the half-widths, which the code no longer does; corrected. The product
      was called "Oishi-Rump" in four places; the paper is by Oishi alone (Crossref), and
      Miyajima attributes the algorithm to Oishi. Oishi (2001) itself was not read.
    Not checked: that each theorem or algorithm number quoted in a docstring is the one in the
    paper, beyond those read during the audit; `RevolTheveny2013`, which is not on Crossref.

43. **The README and the documentation pages against the package, 2026-10-13.**
    `check_doc_names.jl` on nighthawk: of 145 names that the pages document, link to or call,
    five in the README are not defined: `miyajima_eigenvalue_bounds`,
    `rump_2022a_eigenvalue_bounds`, `rump_lange_2023`, `verified_gev_enclosure` (its table of
    eigenvalue routines) and `krawczyk_linear_solve` (its example). The table now lists the
    routines that exist with the result each cites, and the example, run on nighthawk, uses
    `rigorous_eigenvalues` and `krawczyk_linear_system`.
    **Not done:** the pages under `docs/src` were not read against the code; the documentation
    was not built; `docs/src/decompositions.md` has timing tables headed "(GLA)" whose
    numbers were not rerun; `docs/src/literate/benchmark_warmstart_circle.tex` names
    `oishi_2023_solve_double64`, to be checked.
    **To decide (Isaia):** `verify_generalized_eigenpairs(A, B, …)` takes ball matrices and
    states its bounds "for every symmetric `A` and every symmetric `B` in the given balls",
    the theorem being for a symmetric definite pencil. By the rule of item 40 a ball matrix is
    not symmetric; the routine could take `Symmetric` point matrices, or keep the statement
    about the symmetric matrices of the balls, which is what Miyajima, Ogita, Rump and Oishi
    (2010) need when the entries are only known within a radius.

44. **Return types, rerun 2026-10-13** (`audit_return_types.jl` on nighthawk at cdcc7e8;
    positional calls on `Float64` arguments of nine shapes). 198 exported functions (212 at
    f50b4e8). The field that says the proof went through is `certified` in 20 structs;
    `converged` in the 5 floating-point iterations; `spectrum_covered` and
    `PreconditionerResult.success` as in item 38. Calls whose return type is not inferred as
    concrete: 16 (22 at f50b4e8). `evbox` and `gevbox` are no longer among them. Remaining:
    `certify_eigenpairs` (real or complex by the data); `compute_preconditioner` (by its
    method); `rigorous_svd`, `rigorous_svd_m4` (with or without the block diagonalisation);
    `verified_cholesky`, `verified_lu`, `verified_polar`, `verified_qr`, `verified_svd`, which
    the leading type argument did not make inferable; `adaptive_ogita_svd`,
    `rigorous_block_schur`, `schur_gershgorin_enclosure`,
    `triangular_sylvester_miyajima_enclosure`, `rigorous_eigenvalues`,
    `rigorous_generalized_eigenvalues`. None of these was traced to the statement that loses
    the type.

45. **Matrix sums under the rounding mode again, 2026-10-13** (Isaia: "back to the rounding
    mode; we have decided to trust OpenBLAS with the thread rounding flag").
    776c18a had replaced the rounding-mode block of `A ± B` by `add_up` and `mul_up` entry
    by entry. `sum_radius_forms.jl` on nighthawk, one Julia thread, random dense matrices,
    minimum of 5 runs: the two forms give the same radius in every entry for n = 100, 300,
    1000, 3000 (10,100,000 entries); the time of the radius is 0.000029, 0.00014, 0.0014,
    0.050 s under the mode and 0.00019, 0.0017, 0.019, 0.18 s entry by entry.
    Now the radius of `A ± B` is formed under `RoundUp` (`_sum_radius`), and so are the sums
    `X + PQ` of a radius and a product in `rump_ogita_2024.jl` and `rump_lange_2023.jl`
    (`_sum_product_up`, beside `_product_up`). Kept with the directed operations: scalar
    formulas, the diagonal updates of `A ± λI`, the scalar multiple of a ball matrix, and
    the routines converted before 776c18a (the theorems of Miyajima (2014) and of Rump
    (2011), `schur_gershgorin.jl`, `miyajima_2014a.jl`, `rump_verifyeigall.jl`, the norm
    bounds), where a formula was also corrected at the time.
    A sum or a difference keeps a `Symmetric` radius when the midpoints are `Hermitian` or
    `Symmetric` and the radii `Symmetric` (2ed7865, 621783b): broadcasting had returned a
    dense radius.
    What this rests on is item 46.

46. **A matrix sum under a rounding mode is not covered by the OpenBLAS build** (Isaia,
    2026-10-13: to be kept in view).
    The package trusts two different things when it sets a rounding mode, and only the first
    has a mechanism behind it.
    - **Products** (`A * B` on `Float32`, `Float64` matrices) are OpenBLAS calls. The build
      loaded by the package, with `CONSISTENT_FPCSR`, makes every BLAS thread take the
      rounding mode of the caller; the module checks one product when it is loaded, and the
      probes of items 36 and 45 found no entry on the wrong side.
    - **Sums and other array operations** (`A + B`, `ϵ * M`, `M - N`, `(C1 + C2) / 2` inside
      a `setrounding` block: `_sum_radius`, `_sum_product_up`, the last lines of `MMul4`,
      `MMul3`, `MMul5`, the Oishi product) are Julia's own loops, compiled by Julia, and run
      in the thread that set the mode. The OpenBLAS flag has no part in them. That they
      round in the mode that was set is a property of the code the Julia compiler emits, and
      Julia does not document it as guaranteed: the compiler assumes the default rounding
      mode when it folds constants and rewrites expressions.
    What was measured, on nighthawk only, Julia 1.12.7:
    - a rounding-mode block inside a loop or a comprehension over a vector of structs:
      wrong side in 106,249 of 200,000 (2026-10-12); such forms were removed from the package;
    - six scalar forms: 0 of 200,000 (2026-10-12);
    - the whole-array form of the radius of a sum: the same value as the directed operations
      in all 10,100,000 entries (item 45).
    Not measured: another CPU or operating system, another Julia version, a sum whose
    operands are views, structured or sparse matrices, `BigFloat` (where the mode is MPFR's
    and is read at every operation), and code where the compiler can see constant operands.
    **Done, 7a0f6e0** (Isaia: the check belongs to `using BallArithmetic`, not only to the
    test suite): `NumericalTest.array_rounding_test` compares these array forms with the
    directed operations entry by entry, for `Float64` and `Float32`, dense and `Symmetric`;
    `__init__` runs it after the OpenBLAS check and says the outcome. On nighthawk the forms
    agree, the check takes 0.0009 s and the package loads in 2.2 s (`load_check.jl`). Where
    it fails, the sums of ball matrices fall back to the entrywise directed operations and an
    error is logged.
    The products `MMul3`, `MMul4`, `MMul5` and `oishi_MMul` add and subtract matrices under
    a rounding mode after their BLAS products and had no fallback. **Written, 2026-10-13,
    not yet run:** every such sum goes through `_add_up`, `_add_down`, `_sub_up`,
    `_sub_down`, `_scale_up`, `_centre_radius_up` and `_sum_product_up` in
    `src/types/MMul/MMul4.jl`, which use the entrywise directed operations for `Float32` and
    `Float64` where the check failed; the three `MMul4` methods share `_mmul4`. The test
    sets the flag to `false` and compares with the rounding-mode form.
    The 16 other `setrounding` blocks outside `src/types/MMul`, read 2026-10-13: 12 are the
    scalar `BigFloat` operations of `rounding.jl`, one is commented out, one is `_sum_radius`
    (has the fallback), one is the last product of `_collatz_quotient_up`, now passed by its
    two callers through `_product_up`, and one is in `ext/CUDAExt.jl`, a comprehension of
    scalar differences under `RoundUp`, **not changed**.
    **The BLAS itself (Isaia, 2026-10-13: "if the load-time multiplication fails, we could
    fall back to the prodK route"). Written, not yet run:** `NumericalTest.BLAS_ROUNDING_OK`
    is set to `false` by `__init__` when `rounding_test` fails on all the threads and on one
    (before, three warnings and nothing else). `_product_down_up`, `_product_up`,
    `_product_down` and `_sum_product_up` then return the ends of `_accurate_product_sum`,
    the splitting of Lange and Rump with Dot2 where it declines, for `Float32` and
    `Float64`; the threads are left as they were. When the test fails on all the threads and
    passes on one, the BLAS stays on one thread and the rounding mode is used, as before.
    `rounding_test` is one `Float64` `gemm` of size 1024; it does not exercise `gemv`,
    `Float32` or the triangular and diagonal products.
    **To do:** the check does not cover `BigFloat`, views or sparse matrices.
    Suite at 6c3ccb6 on nighthawk: 183,158 passed, none failed (`rigour_suite78.out`);
    RigorousPseudospectra 300 of 300.

47. **Export audit, 2026-10-13** (`export_inventory`, a scan on bemtivi of the 298 exported
    names at b50fdcc: definition, mentions in `test/`, mentions in the other clones).
    - `eigenvalue_intervals_disjoint` was exported and defined nowhere since 12349c7, which
      rewrote `verify_generalized_eigenpairs`; the export is removed.
    - `run_certification_ogita` was exported and named in no test (its evaluator is tested
      through `dowork_ogita_bigfloat`). A test is added: its bound against the maximum of the
      resolvent on the polygon. **Not yet run.**
    - Exported functions of `CertifScripts` named in no test, **not examined**: `dowork`,
      `adaptive_arcs!`, `save_snapshot!`, `configure_certification!`, `points_on`,
      `polyconv` (they may be exercised through `run_certification` and the Distributed
      extension; that has not been checked by coverage).
    - `CertifScripts` exports three names that begin with an underscore:
      `_clear_bf_ogita_cache!`, `_bf_ogita_cache_stats`, `_set_center_svd_cache!`, and
      `_parametric_cache_stats`, `_clear_parametric_cache!`. **Open.**
    - `polyconv` and `poly_from_roots` are polynomial helpers exported from the certification
      module; GKWExperiments imports `polyconv` from there. **Open.**
    - The scan's docstring column (164 of 298 names with no citation in the docstring) counts
      result types and enum values and has not been read name by name. **Open.**

48. **The precision cascade of the SVD removed, 2026-10-13** (found by searching the source
    for "placeholder" and "stub"). `ogita_svd_cascade` (MultiFloats extension) and
    `ogita_svd_cascade_gla` (GenericLinearAlgebra extension) returned a
    `PrecisionCascadeSVDResult` whose field `σ_min`, documented as "Certified minimum singular
    value" and in `docs/src/decompositions.md` as "a rigorous lower bound", was
    `Σ[end] - residual_norm` with `residual_norm = sqrt(sum(abs2, A - U*Diagonal(Σ)*V'))`
    evaluated in `BigFloat` rounding to nearest. No rounding is bounded, and the inequality
    `σ_min(A) ≥ σ̃_n − ‖A − ŨΣ̃Ṽ*‖` needs `Ũ`, `Ṽ` unitary, which computed factors are not;
    their defects did not enter. Neither routine was called in `src`, nor in the clones under
    `~/Code` and `~/Write` on bemtivi. Deleted: `src/decompositions/svd/precision_cascade_svd.jl`
    (`PrecisionCascadeSVDResult`, `_ogita_iteration!`), the two routines and
    `_ogita_iteration_gla!` in the extensions, `test_precision_cascade_svd.jl`,
    `test_precision_cascade_core.jl`, the cascade testsets of `test_gla_svd.jl`, the section
    "SVD Certification Methods" of `docs/src/decompositions.md` with its timing tables.
    `svd_bigfloat` stays, declared in `svd.jl` and documented as an approximation.
    Certified singular values in `BigFloat` are `rigorous_svd` on a `BigFloat` ball matrix.
    - In the same pass: `rigorous_svd_m4` returned point zero matrices in the fields `U` and
      `residual`, which Theorem 11 of Miyajima (2014) does not provide; they are `nothing`
      now. Two comment blocks were stale: one said the Theorem 11 method "falls back to M1
      bounds" (it throws and points to the right routine), the other stated Parlett's bound
      with `2ρᵢ`, the error that the code under it had been corrected for.
    - **Open:** `docs/src/literate/warmstart_investigation.md` still names the two removed
      routines in a code block. The exported refiners named for an arithmetic
      (`refine_schur_double64`, `refine_schur_hybrid`, `refine_schur_multifloat`,
      `refine_symmetric_eigen_double64`, `_hybrid`, `_multifloat`, `ogita_svd_refine_fast`,
      `_hybrid`, `_multifloat`) are approximations, say so in most docstrings, have no caller
      outside tests except `ogita_svd_refine_multifloat` and
      `refine_symmetric_eigen_multifloat` (one each), and the SVD three carry timing claims
      ("~30× faster") with no script named. Not changed.

49. **The refiners named for an arithmetic removed, 2026-10-13** (Isaia, asked how far to go
    after item 48: all nine and the benchmarks; and "remove the norm bound in Multifloat").
    Deleted: `refine_schur_double64`, `refine_schur_hybrid`, `refine_schur_multifloat`,
    `refine_symmetric_eigen_double64`, `refine_symmetric_eigen_hybrid`,
    `refine_symmetric_eigen_multifloat`, `ogita_svd_refine_fast`, `ogita_svd_refine_hybrid`,
    `ogita_svd_refine_multifloat`; with them `ext/DoubleFloatsExt.jl` and
    `ext/MultiFloatsExt.jl`, which held nothing else (`_spectral_norm_bound_mf`, a
    floating-point `√(‖·‖₁‖·‖_∞)` under the name of a bound, went with the second);
    `test_schur_refinement_variants.jl` and `test_doublefloats_ext/`.
    DoubleFloats and MultiFloats were in `[deps]` and `src` never loaded them: they are removed
    from `[deps]`, `[compat]` and `[extensions]` (they stay in `test/Project.toml`, where the
    tests pass `Double64` and a MultiFloat as `oracle` to the verified decompositions).
    Docs: `benchmark_warmstart_circle.*`, `benchmark_test1.*`, `warmstart_investigation.md`
    deleted; the Double64, hybrid and MultiFloat helpers cut from `benchmark_common.jl`; the
    section "Recommended Extensions" cut from `docs/src/decompositions.md`.
    **Not run at the time of writing.**
    - **Open, found while cutting:** the helpers that remain in
      `docs/src/literate/benchmark_common.jl` and `resolvent_benchmark.jl`, `certify_ogita`
      and `certify_bigfloat`, report `1/(Σ[end] − residual_norm)` from `ogita_svd_refine` as a
      bound, under the comment "Rigorous lower bound"; `residual_norm` there is
      `_spectral_norm_bound` of `adaptive_ogita_svd.jl`, a floating-point
      `√(‖·‖₁‖·‖_∞)`, and the factors are not unitary. The tracked `.tex` and `.pdf` of
      `benchmark_test2` to `4` and `benchmark_spectral_gap` carry columns computed that way.
      The scripts are docs only; nothing in `src` uses that formula for a statement
      (`_spectral_norm_bound` is called twice inside `ogita_svd_refine`, for its stopping
      test and its reported `residual_norm`).

50. **Exports of `CertifScripts`, 2026-10-13** (continuing item 47; each export counted in
    `src`, `ext`, `test` and the clones under `~/Code` on bemtivi).
    - The five names beginning with an underscore are no longer exported:
      `_clear_bf_ogita_cache!`, `_bf_ogita_cache_stats`, `_set_center_svd_cache!`,
      `_parametric_cache_stats`, `_clear_parametric_cache!`. No other clone names them; the
      tests call them qualified.
    - `dowork`, `dowork_parametric`, `configure_certification!`, `adaptive_arcs!`,
      `save_snapshot!`, `points_on`, `set_parametric_config!` are named in no test; they are
      the pieces of the distributed driver in `ext/DistributedExt.jl`, which the suite ran
      with the plain worker only, checking that the log was not empty. A test is added for
      the four kinds of worker against the maximum of the resolvent on the polygon. **Not
      run at the time of writing.**
    - **Open:** GKWExperiments does `using BallArithmetic.CertifScripts`, so every export of
      the module enters its namespace, and it uses `dowork`, `adaptive_arcs!`,
      `bound_res_original`, `choose_snapshot_to_load`, `save_snapshot!`,
      `configure_certification!`, `set_schur_matrix!`, `points_on`, `poly_from_roots`,
      `polyconv` and the unexported `_polynomial_matrix`; PlateauExperiment names `dowork`.
      Whether the worker functions and the two polynomial helpers should stay exported is
      tied to that package and is left for the audit of the dependents.
    - **Open:** `circle_resolvent_bound`, `dowork_ogita`, `dowork_parametric`,
      `set_parametric_config!`, `polygon_sagitta`, `GramTransform`, `apply_gram_transform`
      are used in no other clone.

51. **Readers of a refinement's `residual_norm`, 2026-10-14.** Runs first: 81 at b174536,
    184,307 passed, none failed; 82 at 10a2e01, 183,816 passed, none failed
    (`rigour_suite81.out`, `rigour_suite82.out`); RigorousPseudospectra 300 of 300 in both;
    `structured_ball_ops.nighthawk.tsv` at 10a2e01, 336 rows, none with a member outside, an
    exception or a result that is not a ball matrix. (The audit after run 82 had failed to
    load: the untracked `Manifest.toml` of the worktree on nighthawk still listed the two
    deleted extensions; `Pkg.resolve()` there.)
    - `CertifScripts.jl`, the Ogita worker: `refined.residual_norm` decides whether the
      refined SVD is used as the approximation; `certify_svd` then certifies it. Read, not
      changed.
    - `_compute_schur_bigfloat_refined` built ball matrices of the Schur factors whose radii
      were the refinement's floating-point diagnostics ("Create BallMatrix with refinement
      errors as radii") and passed them to `_schur_defects`. The defects computed that way
      hold for the midpoints, so nothing false was returned, and the radii only enlarged them.
      The routine was called nowhere in `src` (`compute_schur_and_error` uses
      `_compute_schur_bigfloat_direct`); it and its tests are deleted.
    - `_spectral_norm_bound` (`adaptive_ogita_svd.jl`): its docstring said "Compute an upper
      bound on the spectral norm"; it is a floating-point evaluation. The docstring and the
      comment of the field `residual_norm` now say estimate. Its two calls are the stopping
      test and that diagnostic.

52. **Private functions that nothing calls, 2026-10-14** (a scan of the 656 function
    definitions of `src` and `ext` for names that are not exported and occur nowhere else in
    the code, docstrings and comments excluded).
    - Deleted, no caller and no test: `src/linear_system/Krawczyk.jl` (a file that was not
      included: `krawczyk` computed `inv(A.c) * v.c` and stopped);
      `src/types/triangularize.jl` (`_triangularize` wrapped midpoint and radius in
      `UpperTriangular`, which drops the lower triangle, and its docstring said it kept "the
      original enclosure guarantees"); `_working_type` and `_to_working` in `verified_lu.jl`.
    - **Open, Isaia's:** `ext/CuTileExt.jl` (commit 4860f9d, "rigorous GPU GEMM
      extensions"): its one entry point `_cutile_mmul4` is called by nothing, named in no
      test and in no docs page. Not touched.
    - **Open:** not exported, not called in `src` or `ext`, named in tests, used in no other
      clone: `compute_enclosure` and `bound_resolvent` (`rigorous_contour.jl`, whose type
      `Enclosure` is exported), `_compute_exclusion_circle_level_set_ode` and `_priori` (same
      file), `svd_bound_L2_resolvent`, `r2_infty_bound_by_blocks`, `off_diagonal_abs`,
      `miyajima_algorithm_1_procedure_1`, and the four kernels `_ccrprod`, `_ccrprod_prime`,
      `_ciprod`, `_ciprod_prime` of Miyajima (2010) in `oishi_mmul.jl`. Each is either an
      entry point that should be exported and documented or code to remove; not decided.

53. **The docs benchmark scripts removed, 2026-10-14** (Isaia: "I think we can remove the docs
    benchmark scripts"). `docs/src/literate` is deleted, 36 tracked files:
    `benchmark_common.jl`, `benchmark_test2` to `5`, `benchmark_spectral_gap`,
    `resolvent_benchmark.jl`, `run_benchmarks.jl`, with the `.tex`, `.pdf` and LaTeX build
    files of four of them. Their helpers `certify_ogita` and `certify_bigfloat` reported
    `1/(Σ[end] − residual_norm)` as a rigorous bound (item 49). No docs page, `docs/make.jl`
    or the README referred to the directory.
    Kept (Isaia, the same day): `ext/CuTileExt.jl` and the twelve functions of item 52, work
    in progress.

54. **The twelve kept functions, read 2026-10-14** (item 52; Isaia: work in progress, to be
    checked and kept). Run 83 at 42470c2 first: 183,807 passed, none failed
    (`rigour_suite83.out`), RigorousPseudospectra 300 of 300.
    - `miyajima_algorithm_1_procedure_1`, read against the paper (Miyajima 2010, Section 4):
      it did not compute Procedure 1, which is the enclosure of `R₁` (`encR1ci`); it computed
      Steps 1 to 3 of Algorithm 1 (`Z_c`, `Z_r`, `Y`, `ε₂`). And it took ball matrices but used
      `mid(B)` only, so its `Z` and its bound were for the midpoint. Renamed
      `miyajima2010_algorithm1_steps1to3`, with a docstring that lists the steps and the
      reference; `Z` is now the enclosure of `⟨mid(B), rad(B)⟩X̃` by `_ccrprod_prime`, the
      kernel `ccrprod′` of Procedure 2, which is our use of it. `_rigorous_matrix_product`,
      its one helper, is gone. Test: members of the ball of `B` in BigFloat. **Not run at the
      time of writing.**
    - Not implemented anywhere with the paper's kernels: Steps 5 and 6, that is Procedure 1
      (`cprod`, `ciprod′`, `ciprod`) and Procedure 2 (`cprod`, `ccrprod′`, `ccr`, `ccrprod`).
      `rigorous_generalized_eigenvalues` applies Theorem 1 with products of ball matrices.
      The four kernels `_ccrprod`, `_ccrprod_prime`, `_ciprod`, `_ciprod_prime` exist and are
      tested on sampled members (`test_MMul.jl`).
    - `rigorous_contour.jl` (`compute_enclosure`, `bound_resolvent`, the two exclusion
      circles): read; the discs' bounds are `certify_svd` on `T − B(c, r)I`, the closure of a
      chain is decided with distances rounded up and sums of radii rounded down, and the
      docstring says that the bounds are for the Schur factor of the midpoint and not for `A`.
      Not changed.
    - `svd_bound_L2_resolvent`: `svd_bound_L2_opnorm_inverse(A − λI)`, the reciprocal of the
      lower end of the smallest singular value rounded up, `Inf` when that end is not
      positive. Not changed.
    - `off_diagonal_abs`: its docstring said "Returns a matrix containing the off-diagonal
      elements"; it returns upper bounds of their moduli as a point ball matrix. Docstring
      corrected.
    - **Open:** `r2_infty_bound_by_blocks` has no docstring and no reference. It returns
      `max_p Σ_{q≠p} ‖|H|_{pq}‖_∞ / sep(p, q)`, with `sep` the distance between the hulls of
      two groups of balls. What that quantity bounds is stated nowhere; a comment in
      `test_vbd_remainder_norm.jl` records that it was once taken as a bound of the
      remainder norm and underestimated it. It cannot be checked until it has a statement.

