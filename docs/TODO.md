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
   transformation accepted; the docstring of the field says it "must be below one". Which of the
   two is wrong was not looked into.

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

12. Step 4: `verified_lu`, `verified_qr`, `hbr_method`, `shaving`, `rump_lange_2023`, the Qi
    intervals, the bounds of the smallest singular value, `verified_gev`, regularity and
    determinant, the extensions, the `krawczyk_sylvester` stub: fix or unexport.
13. Step 5: citations and `docs/src/refs.bib`.
14. `setrounding` blocks that remain: BigFloat `upper_bound_L1_opnorm` and
    `upper_bound_L_inf_opnorm`, `block_enclosure`, `_vbd_block_data`, and those inside
    `rump_verifyeigall.jl` that predate the rule. `_vbd_kappa` is unrounded (a diagnostic).
15. A distributed certification run waits for ever if its workers fail.
