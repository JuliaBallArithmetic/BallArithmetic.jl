# The three verified block decompositions against each other and against a full eigen
# certification.
#
#   none       schur_newton_vbd(refine = :none)       W = the unitary Schur Q, no transformation
#              (this is what schur_gershgorin_enclosure does: the block-diagonal part of Z*AZ, the whole
#              strictly-off-block part left as remainder)
#   entrywise  schur_newton_vbd(refine = :entrywise)  Newton on X, W <- W(I+X), gated per pair
#   block      schur_newton_vbd(refine = :block)      Sylvester / Bavely-Stewart elimination
#   rump2022a  rump_2022a_eigenvalue_bounds           one ball per eigenvalue, not per cluster
#
# rump2022a carries a `verified` flag and declines when its coupling defect reaches one, which it
# does on a defective matrix; the balls it returns in that case are the floating-point estimates
# and enclose nothing. The flag is honoured below.
#
# Reported per matrix: whether every eigenvalue of the midpoint lies in the union returned, the
# largest and median enclosure radius, the number of clusters, and the wall clock. The radii of
# the first three are the discs of block_enclosure; for rump2022a they are the eigenvalue ball
# radii, which enclose one eigenvalue each and are not the same object, so the comparison is of
# what each method delivers rather than of like with like.
#
#   julia --project=. bench/vbd_compare.jl

using BallArithmetic
using LinearAlgebra, Random, Printf, Statistics
using BallArithmetic: mid, rad

const ROUTES = (:none, :entrywise, :block)

"A disc union covers every eigenvalue of `M` when each lies in some disc."
function covers(discs, λs)
    for λ in λs
        any(abs(λ - d.center) <= d.radius * (1 + 1e-12) for d in discs) || return false
    end
    return true
end

function families(n, rng)
    out = Pair{String, Matrix{ComplexF64}}[]
    push!(out, "random" => randn(rng, ComplexF64, n, n) ./ sqrt(n))
    # two well-separated clusters, mildly non-normal
    D = Diagonal(vcat(fill(1.0 + 0im, n ÷ 2), fill(-2.0 + 0im, n - n ÷ 2)))
    S = I + 0.3 * triu(randn(rng, ComplexF64, n, n), 1)
    push!(out, "two clusters" => Matrix(S * D * inv(S)))
    # a Jordan-like defective block
    J = diagm(0 => fill(0.5 + 0im, n), 1 => fill(1.0 + 0im, n - 1))
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    push!(out, "defective" => U * J * U')
    # graded, strongly non-normal
    G = triu(randn(rng, ComplexF64, n, n))
    for i in 1:n, j in i:n
        G[i, j] *= 2.0^(-(j - i))
    end
    push!(out, "graded" => Matrix(G))
    return out
end

function main(; n = 24, seed = 20260927)
    rng = MersenneTwister(seed)
    @printf("%-14s %-11s %7s %10s %12s %12s %10s\n", "family", "route", "clust",
        "covers", "max radius", "median r", "seconds")
    for (name, M) in families(n, rng)
        A = BallMatrix(M)
        λs = eigvals(M)
        for r in ROUTES
            local res, t
            try
                t = @elapsed res = schur_newton_vbd(A; refine = r)
            catch err
                @printf("%-14s %-11s %7s %10s %12s %12s %10s   (%s)\n", name, String(r),
                    "-", "-", "-", "-", "-", sprint(showerror, err)[1:min(end, 40)])
                continue
            end
            d = block_enclosure(res)
            rr = [x.radius for x in d]
            @printf("%-14s %-11s %7d %10s %12.4e %12.4e %10.4f\n", name, String(r),
                length(d), covers(d, λs), maximum(rr), median(rr), t)
        end
        try
            t = @elapsed rres = rump_2022a_eigenvalue_bounds(A)
            if !rres.verified
                @printf("%-14s %-11s %7s %10s %12s %12s %10.4f   (declined, coupling defect %.2e)\n",
                    name, "rump2022a", "-", "declined", "-", "-", t, rres.coupling_defect)
            else
                rr = [rad(b) for b in rres.eigenvalues]
                cov = all(any(abs(λ - mid(b)) <= rad(b) * (1 + 1e-12) for b in rres.eigenvalues)
                          for λ in λs)
                @printf("%-14s %-11s %7d %10s %12.4e %12.4e %10.4f\n", name, "rump2022a",
                    length(rr), cov, maximum(rr), median(rr), t)
            end
        catch err
            @printf("%-14s %-11s %7s %10s %12s %12s %10s   (%s)\n", name, "rump2022a",
                "-", "-", "-", "-", "-", sprint(showerror, err)[1:min(end, 40)])
        end
        println()
    end
end

main(; n = something(tryparse(Int, get(ARGS, 1, "")), 24))
