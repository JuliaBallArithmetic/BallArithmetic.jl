# The shaving method of Section 5.9 of Horáček's thesis with its linear programme solved by
# HiGHS through JuMP (the extension LinearProgrammingExt).
using Test, LinearAlgebra, Random
using BallArithmetic
using JuMP, HiGHS

@testset "Shaving method with a linear programming solver" begin
    rng = MersenneTwister(20261015)
    # the hull of the solution set of a regular interval system is the hull of the solutions of
    # its vertex systems, here in rational arithmetic
    function vertex_solutions(Ac, AΔ, bc, bΔ)
        n = size(Ac, 1)
        C, D, c, d = Rational{BigInt}.(Ac), Rational{BigInt}.(AΔ), Rational{BigInt}.(bc),
            Rational{BigInt}.(bΔ)
        sols = Vector{Vector{Rational{BigInt}}}()
        for mask in 0:(2^(n * n + n) - 1)
            V = [C[i, j] + (isodd(mask >> ((j - 1) * n + i - 1)) ? D[i, j] : -D[i, j])
                 for i in 1:n, j in 1:n]
            r = [c[i] + (isodd(mask >> (n * n + i - 1)) ? d[i] : -d[i]) for i in 1:n]
            push!(sols, V \ r)
        end
        return sols
    end
    contains_all(x, sols) = all(
        all(big(mid(x)[i]) - big(rad(x)[i]) <= s[i] <= big(mid(x)[i]) + big(rad(x)[i])
            for i in eachindex(s)) for s in sols)

    @testset "the shaved box contains the solution set" begin
        cut, tried, gained = 0, 0, Float64[]
        for n in (2, 3), _ in 1:(n == 2 ? 80 : 10)
            Ac = randn(rng, n, n) + 3I
            AΔ = rand(rng, n, n) .* 10.0^rand(rng, -3:0.5:-0.5)
            bc = randn(rng, n)
            bΔ = rand(rng, n) .* 10.0^rand(rng, -3:0.5:-0.5)
            A, b = BallMatrix(Ac, AΔ), BallVector(bc, bΔ)
            is_regular(A).is_regular || continue
            x0 = interval_initial_enclosure(A, b)
            all(isfinite, rad(x0)) || continue
            sols = vertex_solutions(Ac, AΔ, bc, bΔ)
            for (passes, improvements) in ((1, 0), (1, 3), (2, 3))
                r = interval_shaving(A, b, x0; passes, improvements)
                tried += 1
                @test !r.empty
                @test contains_all(r.solution, sols)
                if r.components_shaved > 0
                    cut += 1
                    push!(gained, maximum(rad(r.solution)) / maximum(rad(x0)))
                end
            end
        end
        @info "shaving: a slice was cut in $cut of $tried runs; median ratio of the largest radius after and before: $(isempty(gained) ? NaN : sort(gained)[cld(length(gained), 2)])"
        @test cut > 0
    end

    @testset "also on the enclosure of the HBR method" begin
        A = BallMatrix([3.0 -0.5; 0.5 3.0], fill(1.0, 2, 2))
        b = BallVector([0.0, 0.0], [4.0, 4.0])
        r0 = hbr_method(A, b)
        if r0.certified
            r = interval_shaving(A, b, r0.solution; passes = 2)
            @test !r.empty
            @test contains_all(r.solution, vertex_solutions(mid(A), rad(A), mid(b), rad(b)))
            @test all(rad(r.solution) .<= rad(r0.solution) .* (1 + 1e-12))
        end
    end

    @testset "a box away from the solution set" begin
        A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.01, 2, 2))
        b = BallVector([5.0, 4.0], fill(0.01, 2))
        far = BallVector([10.0, 10.0], [0.5, 0.5])
        @test interval_shaving(A, b, far).empty
    end
end
