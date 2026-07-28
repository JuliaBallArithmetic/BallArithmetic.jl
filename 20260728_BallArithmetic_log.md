# BallArithmetic.jl — 2026-07-28

## Gram-weighted resolvent certification (route b) + verified_cholesky rigor fixes

### What was run

Added `gram` / `gram_factor` / `gram_factor_inv` / `gram_kwargs` keywords to the
resolvent certificators (`run_certification`, `run_certification_ogita`,
`run_certification_parametric`, and the `DistributedExt` path), implemented as a
change of variables `Ã = L A L⁻¹` with `G = L*L` — so `‖(A-z)⁻¹‖_G = ‖(Ã-z)⁻¹‖₂`
exactly, with no `√κ₂(G)` penalty. Transform is applied once on the driver
before the Schur reduction; no worker state changed.

While validating rigor, found and fixed three pre-existing correctness bugs.

### verified_cholesky was not rigorous (two independent bugs)

Test: compare the returned factor enclosure against a 512-bit reference
Cholesky of `(A+A*)/2`, entrywise.

Before, `n=12`, `use_bigfloat=false`: **2 of 78 entries failed to enclose** the
true factor, worst by 6× the radius (entry (1,12): dev 8.85e-17, radius
1.44e-17). With the default `use_bigfloat=true` essentially *every* upper entry
failed (radii ~1e-31 around a center only Float64-accurate).

Causes:
1. `I_E = X_G' * A_sym * X_G` computed in plain floating point, then
   `E = I_E - I` passed to `_lu_perturbed_identity` as an exact point matrix —
   the ~eps‖X_G‖²‖A‖ rounding of the triple product was never bounded.
2. Step 5 evaluates `G = G_E X_G⁻¹` as `G_E G̃`, which holds only if the
   preconditioner is exactly `G̃⁻¹`; `X_G = inv(G_approx)` is not, losing
   ~eps·κ(G̃).

Fixes: `_lu_perturbed_identity` gained an `E_rad` keyword (the §3.1 bounds are
monotone in `|E|`, so they are evaluated at `|E|+E_rad` with `E_rad` added to
the offset radii) plus directed rounding on all Δ arithmetic; `verified_cholesky`
encloses `I_E = X_G* A X_G` with ball products and assembles `G_E`/`G` with ball
products instead of the hand-rolled Revol–Théveny formula.

Bug 2 was first fixed by applying the preconditioner as two rigorous
`forward_substitution` solves against `G̃` itself (making `X_G⁻¹ = G̃` true by
construction), then replaced by the **Miyajima–Rump inversion bound** from
`~/Code/RigPseudospectra.jl` (`implementation.md:37`, `miyajima_rump.jl:8`:
"never form a certified inverse; `X_G` is *free*, only the residual matters"):

    G̃X_G = I - R,  R = I - G̃X_G   ⟹   X_G⁻¹ = (I - R)⁻¹ G̃
    (I - R)⁻¹ = I + R + R²(I-R)⁻¹,   ‖R²(I-R)⁻¹‖₂ ≤ ‖R‖₂²/(1-‖R‖₂)

so `G = G_E (I + R + T) G̃` with `T` a ball matrix of that uniform radius
(`|Tᵢⱼ| ≤ ‖T‖₂`), and `‖R‖₂ < 1` proves `G̃`, `X_G` nonsingular. All BLAS-level
ball products. In the BigFloat path `X_G` is first sharpened by one point Newton
step `X ← X(2I - G̃X)`, which squares `‖R‖` for two matrix products. The Neumann
correction is a full matrix, so the strictly-lower zeroing is no longer a no-op:
it now checks `0` really lies in each lower enclosure and fails otherwise.

After: **0 bad entries** across n = 5/20/60, cond(G) = 1e0…1e8, complex
Hermitian, BigFloat input, and both precision paths.

`residual_norm` is now a rigorous bound on the enclosure —
`upper_bound_L2_opnorm(G*G - A)` over `‖A‖₂ ≥ maxᵢ Aᵢᵢ` — rather than a
midpoint-only diagnostic. Values move from a fake ~1e-16 to honest
~1e-30 (BigFloat path) / ~1e-13 (Float64 path); still under all existing
test tolerances.

### Ball complex inv broken for BigFloat

`inv(y::Ball{T,Complex{T}})` used the `Float64` constants `ϵp`/`η`, so
`T = BigFloat` hit `MethodError: mul_up(::BigFloat, ::Float64)`. Switched to
`machine_epsilon(T)` / `subnormal_min(T)`, matching the real-ball method at
`ball.jl:313`.

### Numbers

Norm identity check (n=12, random SPD G, cond ≈ 8.3):

| z | ‖R(z)‖_G | ‖(Ã-z)⁻¹‖₂ | √κ₂(G)·‖R‖₂ (cheap route) |
|---|---|---|---|
| 1.3+0.7i | 1.9528246269692076 | 1.9528246269692076 | 3.3219 |
| 0.2-0.9i | 5.118952813747308 | 5.118952813747307 | 8.2356 |

End-to-end on the same matrix, circle r=0.6, 16 samples, η=0.9:
plain 2-norm 5018.4 · gram (G-norm) 6479.1 · cheap `Cbound=√κ₂(G)` route 9790.1.
Transform agrees with certifying `apply_gram_transform(A, gt)` by hand to 1e-8.

`verified_cholesky` cost, triangular-solve version → Miyajima–Rump version:

| n | BigFloat | Float64 |
|---|---|---|
| 100 | 4.49 s → 4.27 s | 0.087 s → 0.010 s |
| 200 | 36.1 s → 34.8 s | 0.492 s → 0.066 s |

Float64 path **7.5× faster** at n=200; the BigFloat path is unchanged because it
is dominated by BigFloat ball matmuls either way (the two ball triangular solves
it replaced cost 22.2 s at n=200, but four BigFloat ball matmuls cost about the
same). Enclosure widths are unchanged — the Newton step keeps the BigFloat path
at ~1e-31 for well-conditioned G, and 0 bad entries across the whole sweep.

### Tests

Full suite: **4084 pass, 15 broken (pre-existing), 0 fail**, 6m56s.
New `test/test_pseudospectra/test_gram_transform.jl`: 48 tests covering the
diagonal and Cholesky paths, enclosure of the exact conjugation, the norm
identity, both `inverse_method` variants, user-supplied factor/inverse
(including rejection of a bogus inverse), argument validation, and the
certification wiring.

### Caveat

`use_double_precision` in `verified_cholesky` is now a no-op (documented as
retained for compatibility); `_double_precision_triple_product_symmetric` and
its `DoubleFloatsExt` override are consequently dead code.
