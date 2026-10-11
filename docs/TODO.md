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
      `verifysvdall(A; kappa)`, in place of the stub. Left to do:
      the left null space for `m > n` in `verifysvdall`, which
      the paper describes and leaves out of its listing; the paper's sizes (n = 100 to 1000)
      and tables were not reproduced, the tests using n up to 30 or 40.
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
    - **The extension variants, not rewired:** `verified_{lu,cholesky,qr,polar,takagi}_double64`,
      `_multifloat` and `verified_{lu,cholesky,qr,polar}_gla` still run their own code on
      `_lu_perturbed_identity`. `probe_extension_decompositions.jl` (n = 10, 40, real, factors at
      2048 bits): `lu_double64`, `lu_multifloat`, `lu_gla`, `cholesky_double64` and
      `cholesky_multifloat` return `success = true` with about half of the entries outside
      (errors near 1e-31 against radii near 1e-47 and 1e-62; for `lu_gla` 5.5e-17 against
      9.7e-77); the three polar variants had no entry outside; for the QR variants and
      `cholesky_gla` the midpoints differ from the reference by order one, so the probe compared
      different normalisations and says nothing. Left as they are for the moment (Isaia,
      2026-10-11).
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
    - **No source found yet:** `verified_linear_solve_hmatrix` (cites Minamihata 2015, not in
      the library), `krawczyk_sylvester` (a stub; the Sylvester enclosures are now those of Miyajima 2013),
      `gev_invariant_subspaces`.
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
