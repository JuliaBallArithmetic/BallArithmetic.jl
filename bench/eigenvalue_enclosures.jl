# Comparison of every exposed route to an enclosure of ALL eigenvalues of a ball matrix.
#
# What is measured, per matrix and method: the wall clock; how many of the n eigenvalues the
# method places inside one of its returned regions, checked against a 256-bit reference; and the
# largest radius it returns. A method that declines is recorded as declining, not as failing.
#
# The reference is `eigvals` at 256 bits on the midpoint. For a non-normal matrix that reference
# is itself only as good as BigFloat makes it, so the comparison is against a high-precision
# computation and not against the exact spectrum; where a method's radius is near or below the
# reference's own error the "enclosed" count says more about the reference than about the method,
# and the triangular families are included because for them the spectrum is the diagonal exactly.
#
# Run: julia --project=. bench/eigenvalue_enclosures.jl
# Writes bench/eigenvalue_enclosures.<hostname>.tsv

using BallArithmetic
using LinearAlgebra
using Random
using Printf

BA = BallArithmetic
setprecision(256)

# ---------------------------------------------------------------------------------------------
# the methods: each returns a vector of (center, radius, count) regions, or nothing if it declines
# ---------------------------------------------------------------------------------------------

function m_verifyeigall(A, method)
    r = verifyeigall(A; method)
    isempty(r.clusters) && return nothing
    return [(complex(r.centers[i]), r.radii[i], length(r.clusters[i]))
            for i in eachindex(r.clusters)]
end

function m_block(f, A; kwargs...)
    r = try
        f(A; kwargs...)
    catch e
        return nothing
    end
    return [(complex(b.center), b.radius, b.mult) for b in block_enclosure(r)]
end

function m_evbox(A)
    ev = try
        evbox(A)
    catch e
        return nothing
    end
    return [(complex(mid(z)), rad(z), 1) for z in ev]
end

function m_certify_eigenpairs(A)
    r = try
        certify_eigenpairs(A)
    catch e
        return nothing
    end
    # one Newton-Kantorovich enclosure per eigenpair; unverified ones carry no claim
    regions = [(complex(mid(p.eigenvalue)), rad(p.eigenvalue), 1)
               for p in r.results if p.verified]
    return isempty(regions) ? nothing : regions
end

function m_rump_lange(A)
    r = try
        rump_lange_2023_cluster_bounds(A)
    catch e
        return nothing
    end
    r.verified || return nothing
    # the per-eigenvalue balls, read as discs in the plane; their centres are real even when the
    # spectrum is not, so the radius has to cover the imaginary part as well
    return [(complex(mid(z)), rad(z), 1) for z in r.eigenvalues]
end

const METHODS = [
    "verifyeigall :rump2022a" => A -> m_verifyeigall(A, :rump2022a),
    "verifyeigall :rump2022aneumann" => A -> m_verifyeigall(A, :rump2022aneumann),
    "verifyeigall :rump2022adiscclusters" => A -> m_verifyeigall(A, :rump2022adiscclusters),
    "verifyeigall :miyajima2014a" => A -> m_verifyeigall(A, :miyajima2014a),
    "schur_gershgorin_enclosure" => A -> m_block(schur_gershgorin_enclosure, A),
    "miyajima2014a_schurnewton" => A -> m_block(miyajima2014a_schurnewton, A),
    "evbox" => m_evbox,
    "certify_eigenpairs" => m_certify_eigenpairs,
    "rump_lange_2023" => m_rump_lange,
]

# ---------------------------------------------------------------------------------------------
# the matrices
# ---------------------------------------------------------------------------------------------

function families(n, rng)
    out = Pair{String, Matrix{Float64}}[]
    push!(out, "randn" => randn(rng, n, n))
    # triangular: the spectrum is the diagonal, exactly
    d = randn(rng, n)
    push!(out, "triangular" => triu(randn(rng, n, n), 1) + Diagonal(d))
    # symmetric: normal, well conditioned eigenvectors
    S = randn(rng, n, n)
    push!(out, "symmetric" => (S + S') / 2)
    # a k-fold cluster, rotated to be non-triangular
    k = 3
    dd = Float64[]
    while length(dd) < n
        l = randn(rng)
        for _ in 1:k
            push!(dd, l)
        end
    end
    Q = Matrix(qr(randn(rng, n, n)).Q)
    push!(out, "cluster k=3" => Q * (triu(randn(rng, n, n) .* 1e-6, 1) + Diagonal(dd[1:n])) * Q')
    # a single Jordan block, the defective extreme
    J = diagm(0 => fill(0.7, n), 1 => ones(n - 1))
    Qj = Matrix(qr(randn(rng, n, n)).Q)
    push!(out, "jordan n" => Qj * J * Qj')
    # Grcar, the standard non-normal benchmark
    G = zeros(n, n)
    for i in 1:n, j in 1:n
        (j == i - 1) && (G[i, j] = -1.0)
        (0 <= j - i <= 3) && (G[i, j] = 1.0)
    end
    push!(out, "grcar" => G)
    # graded, so the eigenvalues span many magnitudes
    push!(out, "graded" => Matrix(Diagonal(10.0 .^ range(0, -8, length = n))) *
                           Matrix(qr(randn(rng, n, n)).Q))
    return out
end

# ---------------------------------------------------------------------------------------------

"how many of `lams` lie in some returned region, and the largest radius"
function score(regions, lams)
    isnothing(regions) && return (nothing, nothing)
    inside = 0
    for l in lams
        for (c, r, _) in regions
            isfinite(r) || continue
            if abs(l - Complex{BigFloat}(c)) <= BigFloat(r)
                inside += 1
                break
            end
        end
    end
    finite = [r for (_, r, _) in regions if isfinite(r)]
    return (inside, isempty(finite) ? nothing : maximum(finite))
end

function bench(; sizes = (10, 30), seed = 20260927, reps = 3)
    host = gethostname()
    path = joinpath(@__DIR__, "eigenvalue_enclosures.$(host).tsv")
    open(path, "w") do io
        println(io, "host\tn\tfamily\tmethod\tenclosed\tof\tmax_radius\tseconds")
        for n in sizes
            rng = MersenneTwister(seed)
            for (fam, M) in families(n, rng)
                A = BallMatrix(M)
                lams = eigvals(Complex{BigFloat}.(M))
                @printf("\n%-14s n=%-4d  (reference: %d eigenvalues at 256 bits)\n", fam, n,
                    length(lams))
                @printf("  %-32s %-12s %-12s %s\n", "method", "enclosed", "max radius", "seconds")
                for (name, f) in METHODS
                    f(A)                                  # warm
                    t = Inf
                    regions = nothing
                    for _ in 1:reps
                        s = time_ns()
                        regions = f(A)
                        t = min(t, (time_ns() - s) / 1e9)
                    end
                    inside, mx = score(regions, lams)
                    if isnothing(inside)
                        @printf("  %-32s %-12s %-12s %.4f\n", name, "declined", "-", t)
                        println(io, "$host\t$n\t$fam\t$name\tdeclined\t$(length(lams))\t\t$t")
                    else
                        @printf("  %-32s %-12s %-12s %.4f\n", name,
                            "$inside/$(length(lams))",
                            isnothing(mx) ? "all Inf" : @sprintf("%.3g", mx), t)
                        println(io,
                            "$host\t$n\t$fam\t$name\t$inside\t$(length(lams))\t$(isnothing(mx) ? "" : mx)\t$t")
                    end
                    flush(io)
                end
            end
        end
    end
    println("\nwrote ", path)
end

bench()
