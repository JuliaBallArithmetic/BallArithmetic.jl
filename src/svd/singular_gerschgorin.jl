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
    G = Ball[]
    for i in 1:N
        ai = abs(A[i, i])
        ri = upper_bound_norm(helper_i(A[i, :], i), 1)
        ci = upper_bound_norm(helper_i(A[:, i], i), 1)

        Δc = ai^2 - ai * ci + (ci^2) / 4
        Δr = ai^2 - ai * ri + (ri^2) / 4

        if inf(Δc) < 0
            Δc = 0
        end
        if inf(Δr) < 0
            Δr = 0
        end
        lc = inf(sqrt(Δc) - Ball(ci) / 2)
        lr = inf(sqrt(Δr) - Ball(ri) / 2)

        Δc = ai^2 + ai * ci + (ci^2) / 4
        Δr = ai^2 + ai * ri + (ri^2) / 4
        uc = sup(sqrt(Δc) + Ball(ci) / 2)
        ur = sup(sqrt(Δr) + Ball(ri) / 2)

        l = max(min(lc, lr), 0)
        u = max(uc, ur)

        c = (l + u) / 2
        r = @up (u - l) / 2.0

        push!(G, Ball(c, r))
    end
    return G
end

"""
Rebalanced (Theorem 2).
"""
function qi_intervals_rebalanced(A::BallMatrix)
    norm_r = [norm(v, 1) for v in rows(A.c)]
    norm_c = [norm(v, 1) for v in cols(A.c)]
    k = norm_c ./ norm_r

    D = Matrix(Diagonal(k))
    Dinv = Matrix(Diagonal(1 ./ k))

    resA = Dinv * (A * D)

    return qi_intervals(resA)
end

"""
Rebalanced (Theorem 3).
"""
function qi_sqrt_intervals_rebalanced(A::BallMatrix)
    norm_r = [norm(v, 1) for v in eachrow(A.c)]
    norm_c = [norm(v, 1) for v in eachcol(A.c)]
    k = norm_c ./ norm_r

    D = Matrix(Diagonal(k))
    Dinv = Matrix(Diagonal(1 ./ k))

    resA = (Dinv * A) * D

    return qi_sqrt_intervals(resA)
end
