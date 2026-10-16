# adaptive_arcs!'s termination guard and its arc budget.
#
# Three things are asserted: that the default path is bit-identical to the behaviour before the
# budget existed, against expected values captured by running the pre-change code rather than by
# re-running the new code; that the budget is honoured and reported; and that the refinement
# terminates with a reported outcome, instead of looping, when the acceptance test can never be
# satisfied.

using Test
using BallArithmetic
const CS = BallArithmetic.CertifScripts

# A deterministic evaluator: sigma = |z| - 1/2, so every sample on the unit circle is exactly 1/2.
# An arc of chord l is accepted when sup(l/sigma) <= eta.
unit_eval(z) = (; z = ComplexF64(z), val = Ball(abs(ComplexF64(z)) - 0.5, 0.0),
    second_val = Ball(0.0, 0.0))

# An evaluator the test can never satisfy: sigma tiny but POSITIVE, since a ball containing zero
# has no reciprocal and l/sigma would raise instead of rejecting.
tiny_eval(z) = (; z = ComplexF64(z), val = Ball(1.0e-20, 0.0), second_val = Ball(0.0, 0.0))

@testset "adaptive_arcs! default path unchanged" begin
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
    @test length(collect(CS._initial_arcs(circle))) == 8
    # the chord of an eighth of the unit circle
    @test abs(cis(2π / 8) - 1.0) ≈ 0.7653668647301795

    # eta = 2: 0.76537/0.5 = 1.5307 <= 2, so every initial arc is accepted and nothing is bisected
    arcs = collect(CS._initial_arcs(circle))
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 2.0; evaluator = unit_eval)
    @test res.ok
    @test res.processed == 8
    @test res.reason === :complete
    @test res.z === nothing
    @test length(cache) == 8
    @test isempty(arcs)

    # eta = 1: every initial arc is rejected (1.5307 > 1) and both halves are accepted
    # (0.38268/0.5 = 0.7654 <= 1), so 8 rejections then 16 acceptances
    arcs2 = collect(CS._initial_arcs(circle))
    cache2 = Dict{ComplexF64, Any}()
    res2 = CS.adaptive_arcs!(arcs2, cache2, Dict{Int, Any}(), 1.0; evaluator = unit_eval)
    @test res2.ok
    @test res2.processed == 24
    @test res2.reason === :complete
    @test res2.z === nothing
    @test length(cache2) == 16
end

@testset "adaptive_arcs! honours maxarcs" begin
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 8)
    # the eta = 1 run costs 24 arcs; a budget of 5 must stop at exactly 5 and say why
    arcs = collect(CS._initial_arcs(circle))
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 1.0; evaluator = unit_eval,
        maxarcs = 5)
    @test !res.ok
    @test res.processed == 5
    @test res.reason === :maxarcs
    @test res.z isa ComplexF64

    # a budget at the true cost leaves the result complete, so the bound is not off by one
    arcs2 = collect(CS._initial_arcs(circle))
    cache2 = Dict{ComplexF64, Any}()
    res2 = CS.adaptive_arcs!(arcs2, cache2, Dict{Int, Any}(), 1.0; evaluator = unit_eval,
        maxarcs = 24)
    @test res2.ok
    @test res2.processed == 24
    @test res2.reason === :complete
end

@testset "adaptive_arcs! terminates when no arc can be accepted" begin
    # sigma = 1e-20 against chords of order 1: the test rejects at every scale a Float64 midpoint
    # can reach, so only the non-refinable guard can stop the loop. Without it this does not return.
    circle = CS.CertificationCircle(0.0 + 0.0im, 1.0; samples = 4)
    arcs = collect(CS._initial_arcs(circle))
    cache = Dict{ComplexF64, Any}()
    res = CS.adaptive_arcs!(arcs, cache, Dict{Int, Any}(), 0.5; evaluator = tiny_eval,
        io = stdout, check_interval = 100_000)
    @test !res.ok
    @test res.reason === :arc_not_refinable
    @test res.z isa ComplexF64
    @test res.processed > 0
end

@testset "_refinable" begin
    @test CS._refinable(0.0 + 0.0im, 1.0 + 0.0im)
    @test CS._refinable(1.0 + 0.0im, 1.0 + 1.0e-300im)
    # adjacent floats: the midpoint cannot separate from both endpoints
    @test !CS._refinable(1.0 + 0.0im, nextfloat(1.0) + 0.0im)
    @test !CS._refinable(1.0 + 0.0im, 1.0 + 0.0im)
end
