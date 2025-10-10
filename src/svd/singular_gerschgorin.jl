using BallArithmetic
export qi_intervals, qi_sqrt_intervals

using Base.Rounding: RoundDown, RoundUp

function helper_i(v::BallVector, i)::BallVector
    if i == 1
        return BallVector(v.c[2:end], v.r[2:end])
    elseif i == length(v)
        return BallVector(v.c[1:(end - 1)], v.r[1:(end - 1)])
    else
        return BallVector(
            [v.c[1:(i - 1)]; v.c[(i + 1):end]], [v.r[1:(i - 1)]; v.r[(i + 1):end]])
    end
end

"""
Qi1984 intervals for singular values (Theorem 2).
Returns intervals B_i as a Vector{Ball}.
"""
function qi_intervals(A::BallMatrix)
    m, n = size(A)
    N = min(m, n)
    B = Ball[]
    for i in 1:N
        ai = abs(A[i, i])
        ri = upper_bound_norm(helper_i(A[i, :], i), 1)
        ci = upper_bound_norm(helper_i(A[:, i], i), 1)
        si = max(ri, ci)

        if inf(ai - si) < 0
            c = sup((ai + si) / 2)
            r = c
            push!(B, Ball(c, r))
        else
            push!(B, ai + Ball(0, si))
        end
    end
    return B
end

"""
Sharper square-root intervals (Theorem 3).
"""
function qi_sqrt_intervals(A::BallMatrix)
    m, n = size(A)
    N = min(m, n)
    outer = qi_intervals(A)
    G = Vector{Ball}(undef, N)

    for i in 1:N
        ai = abs(A[i, i])
        ri = upper_bound_norm(helper_i(A[i, :], i), 1)
        ci = upper_bound_norm(helper_i(A[:, i], i), 1)

        ci_ball = Ball(ci)
        ri_ball = Ball(ri)
        T = typeof(ri)

        Δc_lower = ai^2 - ai * ci_ball + (ci_ball^2) / 4
        Δr_lower = ai^2 - ai * ri_ball + (ri_ball^2) / 4

        lc = setrounding(T, RoundDown) do
            sqrt(max(inf(Δc_lower), zero(T))) - ci / 2
        end
        lr = setrounding(T, RoundDown) do
            sqrt(max(inf(Δr_lower), zero(T))) - ri / 2
        end

        Δc_upper = ai^2 + ai * ci_ball + (ci_ball^2) / 4
        Δr_upper = ai^2 + ai * ri_ball + (ri_ball^2) / 4

        uc = setrounding(T, RoundUp) do
            sqrt(max(sup(Δc_upper), zero(T))) + ci / 2
        end
        ur = setrounding(T, RoundUp) do
            sqrt(max(sup(Δr_upper), zero(T))) + ri / 2
        end

        outer_lower = @up outer[i].c - outer[i].r
        outer_upper = @down outer[i].c + outer[i].r

        lower = max(lc, lr, zero(T), outer_lower)
        upper = min(uc, ur, outer_upper)

        if upper < lower
            upper = lower
        end

        # Pull the upper bound slightly inside the containing Qi interval to
        # compensate for subsequent outward rounding when forming the ball.
        clamped_upper = if isfinite(outer_upper) && upper == outer_upper
            prevfloat(outer_upper)
        else
            upper
        end
        clamped_lower = lower

        if clamped_upper < clamped_lower
            clamped_upper = clamped_lower
        end

        center = T((clamped_lower + clamped_upper) / 2)
        radius = setrounding(T, RoundUp) do
            max(clamped_upper - center, center - clamped_lower)
        end

        if radius < zero(T)
            radius = zero(T)
        end

        G[i] = Ball(center, radius)
    end

    return G
end

"""
Rebalanced (Theorem 2).
"""
function qi_intervals_rebalanced(A::BallMatrix)
    D, Dinv = _balancing_diagonals(A)
    resA = Dinv * (A * D)
    return qi_intervals(resA)
end

"""
Rebalanced (Theorem 3).
"""
function qi_sqrt_intervals_rebalanced(A::BallMatrix)
    D, Dinv = _balancing_diagonals(A)
    resA = Dinv * (A * D)
    return qi_sqrt_intervals(resA)
end

function _balancing_diagonals(A::BallMatrix)
    row_norms = [norm(row, 1) for row in eachrow(A.c)]
    col_norms = [norm(col, 1) for col in eachcol(A.c)]

    real_type = promote_type(eltype(row_norms), eltype(col_norms))
    ratios = col_norms ./ row_norms

    k = collect(real_type.(ratios))
    inv_k = collect(real_type.(1 ./ ratios))

    return Diagonal(k), Diagonal(inv_k)
end
