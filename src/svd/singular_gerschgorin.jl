using BallArithmetic
export qi_intervals, qi_sqrt_intervals

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
    qi_intervals(A::BallMatrix)

Qi (1984, Theorem 2) intervals for singular values. Returns the
intervals `Bᵢ` as a `Vector{Ball}`. See Ref. [Qi1984](@cite).
"""
function qi_intervals(A::BallMatrix)
    m, n = size(A)
    N = min(m, n)
    N == 0 && return Ball[]

    intervals = Ball[]
    for i in 1:N
        ai = abs(A[i, i])
        ri = upper_bound_norm(helper_i(A[i, :], i), 1)
        ci = upper_bound_norm(helper_i(A[:, i], i), 1)
        si = max(ri, ci)

        a_lower = inf(ai)
        a_upper = sup(ai)

        lower_raw = @down a_lower - si
        lower = max(lower_raw, zero(lower_raw))
        upper = @up a_upper + si

        center = (lower + upper) / 2
        upper_radius = @up upper - center
        lower_radius = @up center - lower
        radius = max(upper_radius, lower_radius)

        push!(intervals, Ball(center, radius))
    end

    return intervals
end

"""
    qi_sqrt_intervals(A::BallMatrix)

Sharper square-root intervals for the singular values (Qi 1984,
Theorem 3). See Ref. [Qi1984](@cite).
"""
function qi_sqrt_intervals(A::BallMatrix)
    m, n = size(A)
    N = min(m, n)
    N == 0 && return Ball[]

    intervals = Ball[]
    for i in 1:N
        ai = abs(A[i, i])
        ri = upper_bound_norm(helper_i(A[i, :], i), 1)
        ci = upper_bound_norm(helper_i(A[:, i], i), 1)

        a_lower = inf(ai)
        a_upper = sup(ai)

        ci_half_down = @down ci * 0.5
        ci_half_up = @up ci * 0.5
        ri_half_down = @down ri * 0.5
        ri_half_up = @up ri * 0.5

        zero_val = zero(a_lower)
        lc = if a_upper ≤ ci_half_down
            @down zero_val - a_upper
        elseif a_lower ≥ ci_half_up
            @down a_lower - ci
        else
            @down zero_val - ci_half_up
        end

        lr = if a_upper ≤ ri_half_down
            @down zero_val - a_upper
        elseif a_lower ≥ ri_half_up
            @down a_lower - ri
        else
            @down zero_val - ri_half_up
        end

        uc = @up a_upper + ci
        ur = @up a_upper + ri

        lower = max(min(lc, lr), zero(lc))
        upper = max(uc, ur)

        center = (lower + upper) / 2
        upper_radius = @up upper - center
        lower_radius = @up center - lower
        radius = max(upper_radius, lower_radius)

        push!(intervals, Ball(center, radius))
    end

    return intervals
end

"""
    qi_intervals_rebalanced(A::BallMatrix)

Qi (1984, Theorem 2) intervals after the balancing transformation
suggested in the paper. See Ref. [Qi1984](@cite).
"""
function qi_intervals_rebalanced(A::BallMatrix)
    norm_r = [norm(v, 1) for v in eachrow(A.c)]
    norm_c = [norm(v, 1) for v in eachcol(A.c)]
    k = norm_c ./ norm_r

    D = Diagonal(k)
    Dinv = Diagonal(1 ./ k)

    resA = Dinv * (A * D)

    return qi_intervals(resA)
end

"""
    qi_sqrt_intervals_rebalanced(A::BallMatrix)

Qi (1984, Theorem 3) square-root intervals after the balancing
transformation suggested in the paper. See Ref. [Qi1984](@cite).
"""
function qi_sqrt_intervals_rebalanced(A::BallMatrix)
    norm_r = [norm(v, 1) for v in eachrow(A.c)]
    norm_c = [norm(v, 1) for v in eachcol(A.c)]
    k = norm_c ./ norm_r

    D = Diagonal(k)
    Dinv = Diagonal(1 ./ k)

    resA = (Dinv * A) * D

    return qi_sqrt_intervals(resA)
end
