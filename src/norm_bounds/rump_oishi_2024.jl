# The bound of Rump and Oishi for the norm of the inverse, in the case of an upper triangular
# matrix:
#
#   S. M. Rump and S. Oishi, "A note on Oishi's lower bound for the smallest singular value of
#   linearized Galerkin equations", Theorem 1.3 and Lemma 1.2 (numbering of the preprint read).
#
# The general case, with a lower left block, is `rump_oishi_2024_sigma_min_bound` in
# `oishi_2023_schur.jl`.

"""
    rump_oishi_2024_triangular_bound(T::BallMatrix, k::Int) -> upper bound of ‖T⁻¹‖₂

Theorem 1.3 of Rump and Oishi for an upper triangular `T`, split as `T = [A B; 0 D]` with `A` of
order `k`, and `D = D_d + D_f` into its diagonal and the rest. The lower left block is zero, so
the theorem reads: if `‖D_d⁻¹ D_f‖₂ < 1`, then `T` is invertible and

    ‖T⁻¹‖₂ ≤ max{ ‖A⁻¹‖₂, ‖D_d⁻¹‖₂ / (1 − ‖D_d⁻¹ D_f‖₂) } · ψ(‖A⁻¹B‖₂),

with `ψ(μ)` the norm of `[I −N; 0 I]` for `‖N‖₂ = μ` (Lemma 1.2,
[`psi_schur_factor`](@ref)). The bound holds for every matrix of the ball `T`. `Inf` is returned
where the hypothesis is not established or a diagonal entry of `D` may vanish. For `k` equal to
the order the result is [`svd_bound_L2_opnorm_inverse`](@ref) of `T`.

How each quantity is bounded: `‖A⁻¹‖₂` by `svd_bound_L2_opnorm_inverse`; `A⁻¹B` is enclosed by
[`backward_substitution`](@ref) and its norm, like that of `D_d⁻¹ D_f` (formed by ball
divisions), bounded by [`collatz_upper_bound_L2_opnorm`](@ref); `‖D_d⁻¹‖₂` is the largest
`1/mig(d_ii)`; the final operations are rounded upwards.

The note states the theorem for real matrices. Complex triangular matrices are accepted here:
the block factorisation, the Neumann series and Lemma 1.2 use only the norm.

# Reference
S. M. Rump and S. Oishi, *A note on Oishi's lower bound for the smallest singular value of
linearized Galerkin equations*, Japan J. Indust. Appl. Math. **41**(2) (2024), 1097-1104,
doi 10.1007/s13160-024-00645-7, Lemma 1.2 and Theorem 1.3.
"""
function rump_oishi_2024_triangular_bound(T::BallMatrix{RT}, k::Int) where {RT}
    n = size(T, 1)
    n == size(T, 2) || throw(DimensionMismatch("T must be square"))
    (istriu(mid(T)) && istriu(rad(T))) || throw(ArgumentError("T must be upper triangular"))
    1 <= k <= n || throw(ArgumentError("k must be between 1 and $n"))
    k == n && return svd_bound_L2_opnorm_inverse(T)

    A, B, D = T[1:k, 1:k], T[1:k, (k + 1):n], T[(k + 1):n, (k + 1):n]
    m = n - k
    norm_Ainv = svd_bound_L2_opnorm_inverse(A)
    ψ = psi_schur_factor(collatz_upper_bound_L2_opnorm(backward_substitution(A, B)))

    Dc, Dr = mid(D), rad(D)
    norm_Ddinv = zero(RT)
    Fc, Fr = zeros(eltype(Dc), m, m), zeros(RT, m, m)
    for i in 1:m
        mig = sub_down(abs_down(Dc[i, i]), Dr[i, i])
        mig > 0 || return RT(Inf)
        norm_Ddinv = max(norm_Ddinv, div_up(one(RT), mig))
        d = Ball(Dc[i, i], Dr[i, i])
        for j in (i + 1):m
            f = Ball(Dc[i, j], Dr[i, j]) / d
            Fc[i, j], Fr[i, j] = mid(f), rad(f)
        end
    end
    norm_F = m == 1 ? zero(RT) : collatz_upper_bound_L2_opnorm(BallMatrix(Fc, Fr))
    norm_F < 1 || return RT(Inf)
    β = div_up(norm_Ddinv, sub_down(one(RT), norm_F))
    bound = mul_up(max(RT(norm_Ainv), β), RT(ψ))
    return (isfinite(bound) ? bound : RT(Inf))::RT
end

export rump_oishi_2024_triangular_bound
