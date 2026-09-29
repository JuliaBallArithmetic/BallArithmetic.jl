# Two routes to a certified lower bound on σ_min(G₁₁ − zI) over a contour, compared on cost and
# on tightness:
#
#   full      one verified economy SVD per sample, `svdbox(G₁₁ − zI)`, which is what
#             `_evaluate_sample` in src/pseudospectra/CertifScripts.jl does today. O(k³) per z.
#
#   verifyeig one transform up front, then O(k) per sample. `verifyeigall` returns an enclosure
#             A ⊇ W⁻¹G₁₁W together with the basis W. Writing A = D + N with D diagonal,
#             σ_min(A − zI) ≥ minᵢ|Aᵢᵢ − z| − ‖N‖₂ by Weyl, and since
#             (G₁₁ − zI)⁻¹ = W(A − zI)⁻¹W⁻¹,
#
#                 σ_min(G₁₁ − zI) ≥ (minᵢ|Aᵢᵢ − z| − ‖N‖₂) / κ(W),
#
#             with ‖N‖₂ and κ(W) = σ_max(W)/σ_min(W) certified once by `svdbox`. Per sample this
#             is a minimum over k scalars.
#
#   schurnewt the same idea over the block-diagonalising frame of `miyajima2014a_schurnewton`,
#             which is the route that already exists for this; its floor is the better of a global
#             Weyl term and a block-dominance term.
#
# The fast routes cannot be tighter than the full SVD: they pay κ(W) and ‖N‖₂, where the SVD
# computes σ_min directly. What they buy is the per-sample order, so the question is the crossover
# in the number of samples and how much floor is given up.
#
# Run: julia --project=. bench/resolvent_routes.jl
# Writes bench/resolvent_routes.<hostname>.tsv

using BallArithmetic
using LinearAlgebra
using Random
using Printf

BA = BallArithmetic

# ---------------------------------------------------------------------------------------------

"σ_min(M) ≥ this, certified, by one verified economy SVD"
function full_floor(G::BallMatrix{T}, z) where {T}
    Σ = svdbox(G - Ball(Complex{T}(z), zero(T)) * I)
    s = Σ[end]
    return setrounding(T, RoundDown) do
        mid(s) - rad(s)
    end
end

"precompute for the verifyeigall route: the transformed diagonal, ‖N‖₂ and κ(W)"
function verifyeig_precompute(G::BallMatrix{T}) where {T}
    r = verifyeigall(G)
    W = BallMatrix(r.basis)
    sv = svdbox(W)
    smax = maximum(mid(s) + rad(s) for s in sv)
    smin = minimum(mid(s) - rad(s) for s in sv)
    smin > 0 || return nothing
    kappa = setrounding(T, RoundUp) do
        smax / smin
    end
    # the transform itself: an enclosure of W⁻¹ G W, reconstructed as Y*G*W with the Neumann slack
    Y = BallMatrix(inv(r.basis))
    R2 = Y * W - I
    d2 = upper_bound_L2_opnorm(R2)
    d2 < 1 || return nothing
    A = Y * G * W
    slack = setrounding(T, RoundUp) do
        d2 * upper_bound_L2_opnorm(A) / (one(T) - d2)
    end
    k = size(A, 1)
    dg = [Ball(mid(A)[i, i], rad(A)[i, i]) for i in 1:k]
    Nm = copy(mid(A))
    Nr = copy(rad(A))
    for i in 1:k
        Nm[i, i] = zero(eltype(Nm))
        Nr[i, i] = zero(T)
    end
    nN = setrounding(T, RoundUp) do
        upper_bound_L2_opnorm(BallMatrix(Nm, Nr)) + slack
    end
    return (diag = dg, nN = nN, kappa = kappa)
end

function verifyeig_floor(p, z::Complex{T}) where {T}
    gmin = setrounding(T, RoundDown) do
        minimum(abs(z - mid(d)) - rad(d) for d in p.diag)
    end
    return setrounding(T, RoundDown) do
        max(zero(T), (gmin - p.nN) / p.kappa)
    end
end

"precompute for the block route"
function schurnewt_precompute(G::BallMatrix{T}) where {T}
    res = miyajima2014a_schurnewton(G)
    centres = res.block_centers
    nn = res.block_nonnormality
    rad_ = res.block_coupling
    return (centres = centres, nn = nn, rad = rad_, kappa = res.kappa)
end

function schurnewt_floor(p, z::Complex{T}) where {T}
    s = [setrounding(T, RoundDown) do
             abs(z - p.centres[j]) - p.nn[j]
         end for j in eachindex(p.centres)]
    gmin = minimum(s)
    fglob = setrounding(T, RoundDown) do
        (gmin - maximum(p.rad)) / p.kappa
    end
    fdom = T(-Inf)
    if all(>(zero(T)), s)
        tau2 = setrounding(T, RoundUp) do
            sum((p.rad[j] / s[j])^2 for j in eachindex(s))
        end
        tau2 < 1 && (fdom = setrounding(T, RoundDown) do
            gmin * (one(T) - sqrt(tau2)) / p.kappa
        end)
    end
    return max(zero(T), fglob, fdom)
end

# ---------------------------------------------------------------------------------------------

best(f, reps) = minimum(begin
                            s = time_ns()
                            f()
                            (time_ns() - s) / 1e9
                        end for _ in 1:reps)

function run(; ks = (16, 32, 64), nsamples = 64, reps = 3)
    host = gethostname()
    path = joinpath(@__DIR__, "resolvent_routes.$(host).tsv")
    open(path, "w") do io
        println(io,
            "host\tfamily\tk\troute\tprecompute_s\tper_sample_s\tmin_floor\tmedian_floor\tratio_to_full")
        for k in ks
            rng = MersenneTwister(2026)
            fams = ("grcar" => [(j == i - 1) ? -1.0 : (0 <= j - i <= 3 ? 1.0 : 0.0)
                                for i in 1:k, j in 1:k],
                "randn" => randn(rng, k, k),
                "triangular" => triu(randn(rng, k, k), 1) + Diagonal(randn(rng, k)),
                "symmetric" => (x -> (x + x') / 2)(randn(rng, k, k)))
            for (fam, M) in fams
                G = BallMatrix(M)
                R = 1.2 * opnorm(M, 2)                     # a circle outside the spectrum
                zs = [R * cis(2π * j / nsamples) for j in 0:(nsamples - 1)]

                fulls = Float64[]
                t_full = best(() -> (for z in zs
                    full_floor(G, z)
                end), 1) / nsamples
                for z in zs
                    push!(fulls, full_floor(G, z))
                end
                med_full = sort(fulls)[nsamples ÷ 2]
                @printf("\n%-11s k=%-4d  circle |z| = %.4g\n", fam, k, R)
                @printf("  %-11s %-12s %-13s %-12s %-12s %s\n", "route", "precompute",
                    "per sample", "min floor", "median", "median/full")
                @printf("  %-11s %-12s %-13.3g %-12.4g %-12.4g %s\n", "full", "-", t_full,
                    minimum(fulls), med_full, "1")
                println(io,
                    "$host\t$fam\t$k\tfull\t\t$t_full\t$(minimum(fulls))\t$med_full\t1.0")

                for (name, pre, flo) in (("verifyeig", verifyeig_precompute, verifyeig_floor),
                    ("schurnewt", schurnewt_precompute, schurnewt_floor))
                    p = try
                        pre(G)
                    catch e
                        @printf("  %-11s declined (%s)\n", name,
                            first(sprint(showerror, e), 34))
                        continue
                    end
                    if p === nothing
                        @printf("  %-11s declined\n", name)
                        continue
                    end
                    t_pre = best(() -> pre(G), 1)
                    t_s = best(() -> (for z in zs
                        flo(p, z)
                    end), reps) / nsamples
                    vals = [flo(p, z) for z in zs]
                    med = sort(vals)[nsamples ÷ 2]
                    ratio = med_full > 0 ? med / med_full : NaN
                    @printf("  %-11s %-12.3g %-13.3g %-12.4g %-12.4g %.3g\n", name, t_pre, t_s,
                        minimum(vals), med, ratio)
                    println(io,
                        "$host\t$fam\t$k\t$name\t$t_pre\t$t_s\t$(minimum(vals))\t$med\t$ratio")
                end
                flush(io)
            end
        end
    end
    println("\nwrote ", path)
end

run()
