# `adaptive_arcs!`'s arc budget and its three stop reasons.
#
# Ported from the `log-io-reset` branch, where the budget was keyed on processed arcs and reported
# `:maxarcs`; in `main` the ceiling counts EVALUATIONS and the reason is `:cap`, and the
# non-refinable guard lives inside `_side_decision` rather than in a `_refinable` predicate, so the
# expectations below are stated in main's terms.
#
# Four things are asserted: that the uncapped path walks the number of sides the chord geometry
# predicts; that the budget is honoured, reported, and not off by one at the exact cost of the run;
# that the refinement stops with a reason instead of looping when no side can ever be accepted; and
# that each stop reason is reached by the condition its docstring names.

using Test
using BallArithmetic
const CS = BallArithmetic.CertifScripts

# A deterministic evaluator: sigma = |z| - 1/2, so every sample on the unit circle is exactly 1/2.
# A side of chord l is accepted when sup(l/sigma) <= eta.
unit_eval(z) = (; z = ComplexF64(z), val = Ball(abs(ComplexF64(z)) - 0.5, 0.0),
    second_val = Ball(0.0, 0.0))

# An evaluator the acceptance test can never satisfy: sigma tiny but POSITIVE, since a lower end of
# zero is caught by `:nonpositive` before any bisection is attempted.
tiny_eval(z) = (; z = ComplexF64(z), val = Ball(1.0e-20, 0.0), second_val = Ball(0.0, 0.0))

# A lower end of zero: the `:nonpositive` branch.
zero_eval(z) = (; z = ComplexF64(z), val = Ball(0.0, 0.0), second_val = Ball(0.0, 0.0))

@testset "adaptive_arcs! uncapped path" begin
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
    @test length(CS._initial_arcs(circle)) == 8
    # the chord of an eighth of the unit circle
    @test abs(cis(2π / 8) - 1.0) ≈ 0.7653668647301795

    # eta = 2: 0.76537/0.5 = 1.5307 <= 2, so every initial side is accepted and nothing is bisected
    arcs = CS._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 2.0; evaluator = unit_eval)
    @test res.ok
    @test res.processed == 8
    @test res.reason === :complete
    @test isnan(res.z)
    @test res.evaluations == 8
    @test res.remaining == 0
    @test length(cache) == 8
    @test isempty(arcs)

    # eta = 1: every initial side is rejected (1.5307 > 1) and both halves are accepted
    # (0.38268/0.5 = 0.7654 <= 1), so 8 bisections then 16 acceptances
    arcs2 = CS._initial_arcs(circle)
    cache2 = Dict{ComplexF64, Any}()
    res2 = CS.adaptive_arcs!(arcs2, cache2, Dict{Int, Any}(), 1.0; evaluator = unit_eval)
    @test res2.ok
    @test res2.processed == 24
    @test res2.reason === :complete
    @test isnan(res2.z)
    @test res2.evaluations == 16
    @test length(cache2) == 16
end

@testset "adaptive_arcs! honours maxarcs" begin
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
    # the eta = 1 run costs 16 evaluations; a budget of 5 must stop at exactly 5 and say why
    arcs = CS._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 1.0; evaluator = unit_eval,
        maxarcs = 5)
    @test !res.ok
    @test res.reason === :cap
    @test res.evaluations == 5
    @test res.z isa ComplexF64
    @test !isnan(res.z)
    @test res.remaining > 0

    # a budget at the true cost leaves the result complete, so the ceiling is not off by one
    arcs2 = CS._initial_arcs(circle)
    cache2 = Dict{ComplexF64, Any}()
    res2 = CS.adaptive_arcs!(arcs2, cache2, Dict{Int, Any}(), 1.0; evaluator = unit_eval,
        maxarcs = 16)
    @test res2.ok
    @test res2.processed == 24
    @test res2.evaluations == 16
    @test res2.reason === :complete

    # maxarcs = 0 is the uncapped default, not a budget of nothing
    arcs3 = CS._initial_arcs(circle)
    cache3 = Dict{ComplexF64, Any}()
    res3 = CS.adaptive_arcs!(arcs3, cache3, Dict{Int, Any}(), 1.0; evaluator = unit_eval,
        maxarcs = 0)
    @test res3.ok
    @test res3.evaluations == 16
end

@testset "adaptive_arcs! terminates when no side can be accepted" begin
    # sigma = 1e-20 against chords of order 1: the test rejects at every scale a Float64 midpoint
    # can reach, so only the non-refinable guard can stop the loop. Without it this does not return.
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 4)
    arcs = CS._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 0.5; evaluator = tiny_eval,
        io = devnull, check_interval = 100_000)
    @test !res.ok
    @test res.reason === :not_refinable
    @test res.z isa ComplexF64
    @test res.processed > 0
end

@testset "adaptive_arcs! stops on a non-positive bound" begin
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 4)
    arcs = CS._initial_arcs(circle)
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 0.5; evaluator = zero_eval,
        io = devnull)
    @test !res.ok
    @test res.reason === :nonpositive
    @test res.z isa ComplexF64
    @test res.processed == 0
end

@testset "_side_decision" begin
    tiny = Ball(1.0e-20, 0.0)
    half = Ball(0.5, 0.0)
    # a side whose ratio is below eta is accepted whatever the budget
    @test CS._side_decision(0.0 + 0.0im, 1.0e-20 + 0.0im, half, 0.5, 0, 0)[1] === :accept
    # a rejected side with room to bisect
    @test CS._side_decision(0.0 + 0.0im, 1.0 + 0.0im, tiny, 0.5, 0, 0)[1] === :split
    # adjacent floats: the midpoint cannot separate from both ends
    @test CS._side_decision(1.0 + 0.0im, nextfloat(1.0) + 0.0im, tiny, 0.5, 0, 0)[1] ===
          :not_refinable
    # a side of length zero has ratio zero, so it is accepted rather than found non-refinable
    @test CS._side_decision(1.0 + 0.0im, 1.0 + 0.0im, tiny, 0.5, 0, 0)[1] === :accept
    # a non-positive lower end is caught before the budget and before the midpoint
    @test CS._side_decision(0.0 + 0.0im, 1.0 + 0.0im, Ball(0.0, 0.0), 0.5, 0, 0)[1] ===
          :nonpositive
    @test CS._side_decision(0.0 + 0.0im, 1.0 + 0.0im, Ball(-1.0, 0.0), 0.5, 0, 0)[1] ===
          :nonpositive
    # the budget is tested after the acceptance test and before the midpoint
    @test CS._side_decision(0.0 + 0.0im, 1.0 + 0.0im, tiny, 0.5, 3, 3)[1] === :cap
    @test CS._side_decision(0.0 + 0.0im, 1.0 + 0.0im, tiny, 0.5, 2, 3)[1] === :split
end
