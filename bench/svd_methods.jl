# The singular-value enclosures of `svdbox`, compared on width and on cost.
#
# What is measured, per matrix and method: the largest radius returned, the wall clock, and whether
# every interval contains the corresponding singular value of 50 members of the ball computed at 256
# bits. The containment check is index by index in decreasing order, which is what the theorems
# claim; a method that fails it is a bug and is reported as such.
#
# The question the table answers is which route to take when the input carries a radius. Theorem 10
# is the tighter of the two economy enclosures on an exact matrix and the cheaper of the two always,
# but its residual is `A V̂ − Û Σ̂`, so an input radius `R` reaches the bound as a ball product and is
# bounded entrywise by `R|V̂|`, losing the cancellation that makes `‖R V̂‖₂ ≤ ‖R‖₂` at a cost growing
# like `√n`. `:miyajima2014_thm10_weyl` keeps `R` out of the residual instead: Theorem 10 on
# `mid(A)`, each interval widened by `‖R‖₂` through Weyl's inequality. The numbers in
# `_svd_auto_theorem`'s docstring are the first six rows of this.
#
# Run: julia --project=. bench/svd_methods.jl
# Writes bench/svd_methods.<hostname>.tsv

using BallArithmetic
using LinearAlgebra
using Random
using Printf

BA = BallArithmetic
setprecision(256)

const METHODS = (:miyajima2014_thm10_weyl, :miyajima2014_thm7, :miyajima2014_thm10)

"largest returned radius, and the number of index-wise containment failures over `trials` members"
function assess(A::BallMatrix, method; trials = 50, rng = MersenneTwister(7))
    s = svdbox(A; method)
    mx = maximum(rad(x) for x in s)
    bad = 0
    Am, Ar = mid(A), rad(A)
    for _ in 1:trials
        X = Am .+ Ar .* (2 .* rand(rng, size(Am)...) .- 1)
        sv = svdvals(Complex{BigFloat}.(X))
        for i in eachindex(s)
            lo = BigFloat(mid(s[i])) - BigFloat(rad(s[i]))
            hi = BigFloat(mid(s[i])) + BigFloat(rad(s[i]))
            (lo <= sv[i] <= hi) || (bad += 1)
        end
    end
    return mx, bad
end

best(f, reps) = minimum(begin
                            t = time_ns()
                            f()
                            (time_ns() - t) / 1e9
                        end for _ in 1:reps)

function cases()
    out = Pair{String, BallMatrix}[]
    # the six rows quoted in `_svd_auto_theorem`'s docstring
    for n in (6, 20), r in (0.0, 1e-16, 1e-12)
        M = randn(MersenneTwister(2026 + n), n, n)
        push!(out, "randn n=$n rad=$r" => BallMatrix(M, fill(r, n, n)))
    end
    # graded, so sigma_min is far below sigma_max and an absolute widening is felt
    n = 20
    G = Matrix(Diagonal(10.0 .^ range(0, -8, length = n))) *
        Matrix(qr(randn(MersenneTwister(5), n, n)).Q)
    push!(out, "graded n=20 rad=1e-14" => BallMatrix(G, fill(1e-14, n, n)))
    # the shape the pseudospectra work hands over: a square compression shifted off its spectrum
    k = 40
    H = triu(randn(MersenneTwister(11), k, k), -1)
    Hz = H - 1.2 * opnorm(H, 2) * I
    push!(out, "compression k=40 rad=1e-13" => BallMatrix(Hz, fill(1e-13, k, k)))
    push!(out, "compression k=40 rad=1e-16" => BallMatrix(Hz, fill(1e-16, k, k)))
    return out
end

function bench(; reps = 20)
    host = gethostname()
    path = joinpath(@__DIR__, "svd_methods.$(host).tsv")
    open(path, "w") do io
        println(io, "host\tcase\tmethod\tmax_radius\tcontainment_failures\tseconds")
        @printf("%-28s %-14s %-14s %-14s  %s\n", "case", "thm10+Weyl", "thm7(ball)",
            "thm10(ball)", "weyl/thm7")
        for (name, A) in cases()
            widths = Float64[]
            for m in METHODS
                mx, bad = assess(A, m)
                push!(widths, mx)
                bad == 0 || @printf("  !! %s %s: %d containment failures\n", name, m, bad)
                svdbox(A; method = m)
                t = best(() -> svdbox(A; method = m), reps)
                println(io, "$host\t$name\t$m\t$mx\t$bad\t$t")
            end
            @printf("%-28s %-14.4g %-14.4g %-14.4g  %.3g\n", name, widths[1], widths[2],
                widths[3], widths[1] / widths[2])
            flush(io)
        end
    end
    println("\nwrote ", path)
end

bench()
