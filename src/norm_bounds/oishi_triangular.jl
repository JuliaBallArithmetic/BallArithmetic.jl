# NOTE: backward_singular_value_bound is now defined in rump_oishi_2024.jl
# The implementation there is more general and better documented.
# This old version has been removed to avoid method overwriting during precompilation.

# DISABLED — NON-RIGOROUS for block size k < n.
#
# `oishi_rump_bound(T, k)` is meant to UPPER-bound ‖inv(T)‖₂, but for k < n on a
# non-diagonal upper-triangular matrix it can return a value BELOW the true norm.
# Example (found 2026-06-29):
#     Tmid = [2.0 0.15 0.05; 0.0 1.3 0.2; 0.0 0.0 0.7]   # ‖inv‖₂ = 1.451948
#     k=1: 1.428571  (< true → NON-rigorous)
#     k=2: 1.428571  (< true → NON-rigorous)
#     k=3 (=n): 1.451948 ≥ true  (rigorous)
# Only the full k == n case (which delegates to svd_bound_L2_opnorm_inverse) is safe.
# The block-bound formula below lacks the precondition guard that
# `rump_oishi_triangular` enforces (‖offdiag·inv(diag)‖ < 1), so it silently
# returns a non-rigorous value instead of refusing. Commented out until the
# missing guard / fallback is implemented; callers must use a rigorous estimator.
#
# function oishi_rump_bound(T::BallMatrix, k::Int)
#     if k == size(T)[1]
#         return svd_bound_L2_opnorm_inverse(T)
#     end
#
#     @assert istriu(T.c)&&istriu(T.r) "Matrix T must be upper triangular"
#
#     # Extract blocks
#     A = T[1:k, 1:k]
#     B = T[1:k, (k + 1):end]
#     D = T[(k + 1):end, (k + 1):end]
#
#     # Estimate ||A⁻¹|| using the SVD bound or backward substitution
#     Ainv_norm = svd_bound_L2_opnorm_inverse(A)
#
#     # Compute BD⁻¹ via solving D x = bᵗ and then transposing result
#     BDinv = (backward_substitution(D, B')')  # Solve D x = bᵗ for each row of B
#
#     # Estimate norm of BD⁻¹
#     BDinv_norm = collatz_upper_bound_L2_opnorm(BDinv)
#
#     # Estimate ||D⁻¹||
#     Dinv_norm = sup(backward_singular_value_bound(D)[1])
#
#     # Final bound
#
#     α = Ball(Ainv_norm, 0.0)
#     #@info α
#
#     β = Ball(BDinv_norm, 0.0)
#     #@info β
#
#     γ = Dinv_norm
#     #@info γ
#
#     return max(sup(α * (1 + β)), γ)
# end
