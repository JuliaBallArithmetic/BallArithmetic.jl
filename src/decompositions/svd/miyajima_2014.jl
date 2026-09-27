# Verified bounds for ALL the singular values of a matrix, by
#
#   @Article{Miyajima2014,
#     author  = {Miyajima, Shinya},
#     title   = {Verified bounds for all the singular values of matrix},
#     journal = {Japan Journal of Industrial and Applied Mathematics},
#     year    = {2014},
#     volume  = {31},
#     pages   = {513--539},
#     doi     = {10.1007/s13160-014-0145-5},
#   }
#
# Section 3 of that paper collects five enclosures, and it is careful about whose each one is:
# Theorem 4 is Oishi's, Theorems 5 and 6 are Rump's (Theorems 3.1 and 3.2 of Rump (2011), see
# `rump_2011.jl`), and Theorems 7 and 11 are Miyajima's own. Implemented here are the three that
# the paper proves as enclosures of every singular value:
#
#   _miyajima2014_thm4   Oishi's, square frames:      Sigma_ii -+ (Sigma_ii max(|F|,|G|) + |E|)
#   _miyajima2014_thm7   the economy enclosure, which the paper itself calls `svdbox`
#   _miyajima2014_thm11  one frame, from the eigenvectors of A'A, with Gershgorin and Parlett
#
# With q = min(m,n), U, Sigma, V a numerical SVD of A, and
#
#     E := U Sigma V^T - A,     F := V^T V - I_n,     G := U^T U - I_m,
#
# Theorem 4 needs ||F||_2 < 1 and ||G||_2 < 1 and gives, for i = 1..q,
#
#     Sigma_ii - delta_i <= sigma_i(A) <= Sigma_ii + delta_i,
#     delta_i := Sigma_ii max(||F||_2, ||G||_2) + ||E||_2.
#
# Theorem 7 takes the first q columns Uhat, Vhat and the q by q Sigmahat, with
# Ehat := Uhat Sigmahat Vhat^T - A, Fhat := Vhat^T Vhat - I_q, Ghat := Uhat^T Uhat - I_q, and gives
#
#     Sigmahat_ii sqrt((1-||Fhat||)(1-||Ghat||)) - ||Ehat||  <=  sigma_i(A)
#                                                           <=  Sigmahat_ii sqrt((1+||Fhat||)(1+||Ghat||)) + ||Ehat||.
#
# Theorem 8 proves these are at least as tight as Theorem 4's, and Remark 3 is the warning that
# makes this the right economy enclosure to use: Rump's split U^T A V = D + E carried to an economy
# frame keeps the lower bound but the UPPER bound can fail, the counterexample being m = 2n,
# A = [I_n; 0], U the block swap and V = I_n, where every truncated residual vanishes while every
# singular value is 1. So an economy frame must have its residual measured against A, as Ehat is
# here, and not through Rump's split.
#
# Theorem 11 uses one frame only. With V the eigenvectors of A^T A when m >= n (of A A^T otherwise),
# F := V^T V - I, and Dhat + Ehat := (A V)^T (A V) with Dhat diagonal, put f := |Ehat| 1 (the
# Gershgorin row sums of the off-diagonal part) and assume ||F||_2 < 1. Then for each i:
#
#   * if <Dhat_ii, f_i> is isolated from the other intervals, with
#     rho_i := min_{j != i} (|Dhat_ii - Dhat_jj| - f_j),  g_i := ||Ehat e_i||_2^2 / rho_i  and
#     h_i := min(f_i, g_i), the bounds are
#
#         zeta_i := sqrt((Dhat_ii - h_i)/(1 + ||F||_2))   when Dhat_ii >= h_i, else 0,
#         zetabar_i := sqrt((Dhat_ii + h_i)/(1 - ||F||_2));
#
#     g_i is Parlett's Theorem 3, |theta - alpha| <= ||A y - theta y||^2 / gap, so the denominator
#     is rho_i and NOT 2 rho_i;
#
#   * otherwise, with {i_1..i_k} the connected component of overlapping intervals containing i,
#
#         xi_i := max( min_{j in cluster} (Dhat_jj - f_j),  Dhat_ii - ||Ehat||_inf ),
#         xibar_i := min( max_{j in cluster} (Dhat_jj + f_j),  Dhat_ii + ||Ehat||_inf ),
#
#     and zeta_i := sqrt(xi_i/(1 + ||F||_2)) when xi_i >= 0 else 0, zetabar_i := sqrt(xibar_i/(1 - ||F||_2)).
#     The cluster branch is what makes the clustered case sound: an individual eigenvalue of
#     Dhat + Ehat need not lie in its OWN Gershgorin interval, only the cluster's eigenvalues in
#     the union of the cluster's intervals, so h_i = f_i is not available there.
#
# Every bound below is computed with the rounding mode set outward.

using LinearAlgebra

# The numerical SVD of the midpoint, which is a candidate and needs no verification of its own:
# every theorem here verifies whatever frame it is handed.
function _svd_candidate(A::BallMatrix{T}; full::Bool = false) where {T}
    Am = mid(A)
    F = try
        svd(Am; full = full)
    catch e
        e isa LinearAlgebra.LAPACKException || e isa SingularException || rethrow(e)
        return nothing
    end
    all(isfinite, F.U) && all(isfinite, F.S) && all(isfinite, F.V) || return nothing
    return F
end

"""
    _miyajima2014_thm4(A::BallMatrix) -> Vector{Ball}

Theorem 4 of Miyajima (2014), which that paper attributes to Oishi: with `U`, `Σ`, `V` a
numerical SVD of `A`, `E = UΣVᵀ − A`, `F = VᵀV − Iₙ` and `G = UᵀU − Iₘ`, and provided
`‖F‖₂ < 1` and `‖G‖₂ < 1`,

    Σᵢᵢ − δᵢ ≤ σᵢ(A) ≤ Σᵢᵢ + δᵢ,     δᵢ = Σᵢᵢ max(‖F‖₂, ‖G‖₂) + ‖E‖₂.

Both frames are square, so this is the expensive enclosure; Theorem 8 of the same paper proves
[`_miyajima2014_thm7`](@ref) is never worse. Kept because it is the baseline the paper compares
against. Returns `Inf` radii where the orthogonality condition fails.
"""
function _miyajima2014_thm4(A::BallMatrix{T}) where {T}
    m, n = size(A)
    q = min(m, n)
    # Theorem 4 is stated for U in R^{m x m} and V in R^{n x n}, so the FULL factors are needed;
    # the thin ones would not satisfy G = U^T U - I_m
    F = _svd_candidate(A; full = true)
    F === nothing && return [Ball(zero(T), T(Inf)) for _ in 1:q]
    S = F.S
    Ub, Vb = BallMatrix(F.U), BallMatrix(F.V)
    # the Sigma of Theorem 4 is m by n
    Sig = zeros(T, m, n)
    for i in 1:q
        Sig[i, i] = S[i]
    end
    E = Ub * BallMatrix(Sig) * Vb' - A
    normE = upper_bound_L2_opnorm(E)
    normF = upper_bound_L2_opnorm(Vb' * Vb - I)
    normG = upper_bound_L2_opnorm(Ub' * Ub - I)
    (normF < 1 && normG < 1) || return [Ball(S[i], T(Inf)) for i in 1:q]
    mx = max(normF, normG)
    return [_ball_from_bounds(
                setrounding(T, RoundDown) do
                    S[i] - (setrounding(T, RoundUp) do
                        S[i] * mx + normE
                    end)
                end,
                setrounding(T, RoundUp) do
                    S[i] + S[i] * mx + normE
                end, T) for i in 1:q]
end

"""
    _miyajima2014_thm7(A::BallMatrix) -> Vector{Ball}

Theorem 7 of Miyajima (2014), the economy enclosure that the paper itself calls `svdbox`. With
`Û`, `V̂` the first `q = min(m,n)` columns of a numerical SVD, `Σ̂` the `q × q` diagonal,
`Ê = Û Σ̂ V̂ᵀ − A`, `F̂ = V̂ᵀV̂ − I_q`, `Ĝ = ÛᵀÛ − I_q`, and provided `‖F̂‖₂ < 1` and `‖Ĝ‖₂ < 1`,

    Σ̂ᵢᵢ √((1−‖F̂‖₂)(1−‖Ĝ‖₂)) − ‖Ê‖₂  ≤  σᵢ(A)  ≤  Σ̂ᵢᵢ √((1+‖F̂‖₂)(1+‖Ĝ‖₂)) + ‖Ê‖₂.

Theorem 8 proves these bounds are at least as tight as Theorem 4's, and the residual is measured
against `A` rather than through Rump's split `UᵀAV = D + E`, which Remark 3 shows can lose the
upper bound on an economy frame: for `m = 2n`, `A = [Iₙ; 0]`, `U` the block swap and `V = Iₙ`
every truncated residual vanishes while every singular value is 1.

This is the default of [`svdbox`](@ref). Returns `Inf` radii where the orthogonality condition
fails.
"""
function _miyajima2014_thm7(A::BallMatrix{T}) where {T}
    m, n = size(A)
    q = min(m, n)
    F = _svd_candidate(A)
    F === nothing && return [Ball(zero(T), T(Inf)) for _ in 1:q]
    S = F.S
    Uh = BallMatrix(F.U[:, 1:q])
    Vh = BallMatrix(F.V[:, 1:q])
    Sh = BallMatrix(Diagonal(S[1:q]))
    Eh = Uh * Sh * Vh' - A
    normE = upper_bound_L2_opnorm(Eh)
    normF = upper_bound_L2_opnorm(Vh' * Vh - I)
    normG = upper_bound_L2_opnorm(Uh' * Uh - I)
    (normF < 1 && normG < 1) || return [Ball(S[i], T(Inf)) for i in 1:q]
    fac_lo = setrounding(T, RoundDown) do
        sqrt((one(T) - normF) * (one(T) - normG))
    end
    fac_hi = setrounding(T, RoundUp) do
        sqrt((one(T) + normF) * (one(T) + normG))
    end
    return [_ball_from_bounds(
                setrounding(T, RoundDown) do
                    S[i] * fac_lo - normE
                end,
                setrounding(T, RoundUp) do
                    S[i] * fac_hi + normE
                end, T) for i in 1:q]
end

"""
    _miyajima2014_thm11(A::BallMatrix) -> Vector{Ball}

Theorem 11 of Miyajima (2014): the enclosure from one frame only, the eigenvectors `V` of `AᵀA`
when `m ≥ n` and of `AAᵀ` otherwise. With `F = VᵀV − I`, `D̂ + Ê = (AV)ᵀ(AV)` and `D̂` diagonal,
`f = |Ê|𝟙` the Gershgorin row sums of the off-diagonal part, and `‖F‖₂ < 1`:

for an `i` whose interval `⟨D̂ᵢᵢ, fᵢ⟩` is isolated from the others,

    ρᵢ = min_{j≠i}(|D̂ᵢᵢ − D̂ⱼⱼ| − fⱼ),   gᵢ = ‖Ê eᵢ‖₂²/ρᵢ,   hᵢ = min(fᵢ, gᵢ),
    √((D̂ᵢᵢ − hᵢ)/(1+‖F‖₂)) ≤ σᵢ(A) ≤ √((D̂ᵢᵢ + hᵢ)/(1−‖F‖₂)),

the lower bound being `0` when `D̂ᵢᵢ < hᵢ`; and for an `i` inside a connected component
`{i₁,…,i_k}` of overlapping intervals,

    ξᵢ = max(min_{j∈cluster}(D̂ⱼⱼ − fⱼ), D̂ᵢᵢ − ‖Ê‖_∞),
    ξ̄ᵢ = min(max_{j∈cluster}(D̂ⱼⱼ + fⱼ), D̂ᵢᵢ + ‖Ê‖_∞),

with the same division by `1 ± ‖F‖₂` under the square root.

`gᵢ` is Parlett's bound, Theorem 3 of the paper, `|θ − α| ≤ ‖Ay − θy‖₂²/gap(A,θ)`, so its
denominator is `ρᵢ` and not `2ρᵢ`. The cluster branch is what makes the clustered case sound:
an individual eigenvalue of `D̂ + Ê` need not lie in its own Gershgorin interval, only the
cluster's eigenvalues in the union of the cluster's intervals, so `hᵢ = fᵢ` is not available
there.

Only `V` is needed, so this costs less than the two-frame enclosures and says nothing about the
singular vectors. Returns `Inf` radii where `‖F‖₂ < 1` fails.
"""
function _miyajima2014_thm11(A::BallMatrix{T}) where {T}
    m, n = size(A)
    q = min(m, n)
    # the frame: eigenvectors of A'A (m >= n) or A A' (m < n), largest first
    W = m >= n ? A' * A : A * A'
    ev = try
        eigen(Hermitian(mid(W)))
    catch e
        e isa LinearAlgebra.LAPACKException ? (return [Ball(zero(T), T(Inf)) for _ in 1:q]) :
        rethrow(e)
    end
    Vm = ev.vectors[:, end:-1:1]
    V = BallMatrix(Vm)
    normF = upper_bound_L2_opnorm(V' * V - I)
    normF < 1 || return [Ball(zero(T), T(Inf)) for _ in 1:q]

    AV = m >= n ? A * V : A' * V
    H = AV' * AV                                  # Dhat + Ehat
    k = size(H, 1)
    Dm = T[real(mid(H)[i, i]) for i in 1:k]
    Dr = T[rad(H)[i, i] for i in 1:k]
    absH = upper_abs(H)
    # f_i: the Gershgorin row sums of the OFF-diagonal part, and ||Ehat||_inf is the largest
    f = setrounding(T, RoundUp) do
        T[sum(absH[i, j] for j in 1:k if j != i; init = zero(T)) for i in 1:k]
    end
    normEinf = isempty(f) ? zero(T) : maximum(f)
    # ||Ehat e_i||_2^2 over the i-th column, off the diagonal
    colsq = setrounding(T, RoundUp) do
        T[sum(absH[j, i]^2 for j in 1:k if j != i; init = zero(T)) for i in 1:k]
    end

    # the connected components of the intervals <Dhat_ii, f_i>, with the distance bounded BELOW
    # and the radii above, so a singleton component is proved isolated
    parent = collect(1:k)
    find(x) = (while parent[x] != x
        x = parent[x]
    end;
    x)
    for i in 1:k, j in (i + 1):k
        overlap = setrounding(T, RoundDown) do
            abs(Dm[i] - Dm[j]) - Dr[i] - Dr[j]
        end <= setrounding(T, RoundUp) do
            f[i] + f[j]
        end
        if overlap
            a, b = find(i), find(j)
            a != b && (parent[a] = b)
        end
    end
    comp = Dict{Int, Vector{Int}}()
    for i in 1:k
        push!(get!(comp, find(i), Int[]), i)
    end

    lo2 = Vector{T}(undef, k)
    hi2 = Vector{T}(undef, k)
    for i in 1:k
        cl = comp[find(i)]
        if length(cl) == 1
            # isolated: Gershgorin sharpened by Parlett, rho_i = min_j (|D_ii - D_jj| - f_j)
            rho = T(Inf)
            for j in 1:k
                j == i && continue
                rho = min(rho, setrounding(T, RoundDown) do
                    abs(Dm[i] - Dm[j]) - Dr[i] - Dr[j] - f[j]
                end)
            end
            h = f[i]
            if rho > 0
                g = setrounding(T, RoundUp) do
                    colsq[i] / rho
                end
                h = min(h, g)
            end
            lo2[i] = setrounding(T, RoundDown) do
                Dm[i] - Dr[i] - h
            end
            hi2[i] = setrounding(T, RoundUp) do
                Dm[i] + Dr[i] + h
            end
        else
            # clustered: the union of the cluster's intervals, intersected with the Weyl bound
            lo2[i] = setrounding(T, RoundDown) do
                max(minimum(Dm[j] - Dr[j] - f[j] for j in cl), Dm[i] - Dr[i] - normEinf)
            end
            hi2[i] = setrounding(T, RoundUp) do
                min(maximum(Dm[j] + Dr[j] + f[j] for j in cl), Dm[i] + Dr[i] + normEinf)
            end
        end
    end

    out = Vector{Ball{T, T}}(undef, q)
    for i in 1:q
        s2lo = setrounding(T, RoundDown) do
            max(lo2[i] / (one(T) + normF), zero(T))
        end
        s2hi = setrounding(T, RoundUp) do
            max(hi2[i], zero(T)) / (one(T) - normF)
        end
        out[i] = _ball_from_bounds(setrounding(T, RoundDown) do
                sqrt(s2lo)
            end,
            setrounding(T, RoundUp) do
                sqrt(s2hi)
            end, T)
    end
    return out
end
