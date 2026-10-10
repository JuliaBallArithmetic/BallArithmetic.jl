using LinearAlgebra

@testset "pseudospectra" begin
    A = [1.0 0.0; 0.0 -1.0]
    bA = BallMatrix(A)
    K = svd(A)
    # for diag(1, −1) the resolvent norm at z is 1/min(|z − 1|, |z + 1|)
    truth(z) = 1 / min(abs(z - 1), abs(z + 1))
    # the bound of a chain against the resolvent norm at the centre and on the rim of every disc
    function bound_holds(E)
        b = BallArithmetic.bound_resolvent(E)
        return all(b >= truth(c + s * r * cispi(θ)) * (1 - 1e-12)
        for (c, r) in zip(E.points, E.radiuses) for s in (0.0, 1.0) for θ in 0:0.25:1.75)
    end

    @test BallArithmetic._follow_level_set(0.5 + im * 0, 0.01, K) == (0.5 - 0.01im, 1.0)

    enc = BallArithmetic.compute_enclosure(bA, 0.0, 2.0, 0.01)
    @test length(enc) == 2
    @test enc[1].λ == 1.0 + 0.0 * im
    @test BallArithmetic.bound_resolvent(enc[1]) >= 100
    @test all(abs.(enc[1].points .- 1.0) .<= 0.02)
    for E in enc
        @test E.loop_closure && BallArithmetic.check_enclosure(E)
        @test length(E.points) == length(E.radiuses) == length(E.bounds)
        @test isfinite(BallArithmetic.bound_resolvent(E)) && bound_holds(E)
    end

    enc = BallArithmetic.compute_enclosure(bA, 2.0, 3.0, 0.01)
    @test enc[1].λ == 0.0
    @test BallArithmetic.bound_resolvent(enc[1]) >= 1
    @test all(abs.(enc[1].points) .- 2.0 .<= 0.02)
    @test enc[1].loop_closure && bound_holds(enc[1])

    enc = BallArithmetic.compute_enclosure(bA, 0.0, 0.1, 0.01)
    @test enc[1].λ == 0.0
    @test BallArithmetic.bound_resolvent(enc[1]) >= 1.0
    @test all(abs.((enc[1].points)) .- 0.1 .<= 0.02)
    @test enc[1].loop_closure && bound_holds(enc[1])

    E = BallArithmetic._compute_exclusion_circle_level_set_priori(A, 1.0, 0.01;
        rel_pearl_size = 1 / 64, max_initial_newton = 16)
    @test E.loop_closure && bound_holds(E)
    @test BallArithmetic.bound_resolvent(E) > 100

    E = BallArithmetic._compute_exclusion_circle_level_set_ode(A, 1.0, 0.01;
        max_initial_newton = 16, max_steps = 1000, rel_steps = 16)
    @test E.loop_closure && bound_holds(E)
    @test BallArithmetic.bound_resolvent(E) > 100

    # a chain cut short is not reported as closed, and discs that do not touch are not a chain
    E_open = BallArithmetic._compute_exclusion_set(A, 2.0; max_steps = 5, rel_steps = 16)
    @test !E_open.loop_closure
    E_gap = BallArithmetic.Enclosure(0.0, [1.0 + 0im, -1.0 + 0im, 1im], E.bounds[1:3],
        [0.1, 0.1, 0.1], true)
    @test !BallArithmetic.check_enclosure(E_gap)
    @test !isdefined(BallArithmetic, :bound_enclosure)
end
