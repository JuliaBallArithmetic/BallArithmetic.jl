# To do and further research

Opened 2026-10-11 on the branch `rigour-fixes`. Each item says what was run, where the output is,
and what was not tried. The probe scripts are one-off and live in `~/scratch` on nighthawk, with
their outputs beside them (`<script>.nighthawk.txt`).

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
      different normalisations and says nothing. To decide: remove them, or give the
      Rump and Ogita routines an approximate factorisation computed in the extended type.
    - **Fixed:** `krawczyk_linear_system` is `verifylss` (`6751cf7`); RigorousInvariantMeasures
      calls it. `is_M_matrix` follows Definition 5.4 of Varga (2004) with its sign conditions,
      and `is_H_matrix` is the test on the comparison matrix (`2383207`). Sections 3.1 and 3.2
      of Rump and Ogita (2024) are in (`b698f68`, `5dcdfc0`).
    - **To fix, having users or callers:** `miyajima_spectral_projectors` (StatisticalPeriodicity),
      `evbox` and `gevbox` (SelfConsistentExperiments), `sep_clusters`, the two `σ_min` bounds of
      `oishi_2023_schur.jl`, `adaptive_ogita_svd`.
    - **Implement from the thesis, do not remove** (decided 2026-10-11): the routines of
      `src/linear_system` and `src/matrix_properties` that cite J. Horáček, *Interval linear and
      nonlinear systems* (the thesis is in `references/`, in twelve parts, dated 2019 where the
      files say 2012): `hbr_method` (the audit names Theorem 5.12), `interval_shaving`
      (Chapter 5), `interval_gauss_seidel`, `interval_jacobi`, `is_well_preconditioned` and the
      preconditioners, `interval_least_squares` and `subsquares_method` (Chapter 6),
      `det_gershgorin` and `det_hadamard` (Chapters 7 and 8), `is_regular*` and
      `is_singular_sufficient_condition` (Chapter 11, Theorems 11.12 and 11.13). Each is to be
      read against its chapter and implemented as stated there.
    - **`qi_intervals`, `qi_sqrt_intervals`:** Qi (1984) is a singular value enclosure theorem
      that was tried and found not effective (Isaia, 2026-10-11), and the audit found the
      interval `B_{n+1}` of its Theorem 2 missing for non-square input. Candidate for removal;
      not removed.
    - **Sources to check before deciding:** the `:backward`, `:hybrid` and `:psi` methods of
      `rump_oishi_2024_triangular_bound` against Rump and Oishi (2024), in the library;
      `refine_polar_qdwh`, `refine_cholesky`, `refine_lu`, `refine_takagi`,
      `rigorous_symmetric_eigen_bigfloat` against Ogita and Aishima and Bujanović, Kressner and
      Schröder (2022), in `references/`.
    - **No source found yet:** `verified_linear_solve_hmatrix` (cites Minamihata 2015, not in
      the library), `krawczyk_sylvester` (a stub; the Sylvester enclosures are now those of Miyajima 2013),
      `gev_invariant_subspaces`.
13. Step 5: citations and `docs/src/refs.bib`.
14. `setrounding` blocks that remain: BigFloat `upper_bound_L1_opnorm` and
    `upper_bound_L_inf_opnorm`, `block_enclosure`, `_vbd_block_data`, and those inside
    `rump_verifyeigall.jl` that predate the rule. `_vbd_kappa` is unrounded (a diagnostic).
15. A distributed certification run waits for ever if its workers fail.
