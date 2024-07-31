@testset "Pseudospectra" begin
    A = [1.0 0.0; 0.0 -1.0]
    bA = BallMatrix(A)

    using LinearAlgebra
    K = svd(A)

    @test BallArithmetic._follow_level_set(0.5 + im * 0, 0.01, K) == (0.5 - 0.01im, 1.0)

    enc, errF, errT, normZ, normZinv = BallArithmetic.compute_enclosure_circles(
        bA, 0.0, 2.0, 0.01)

    @test enc[1].λ == 1.0 + 0.0 * im
    @test BallArithmetic.bound_resolvent(enc[1], errF, errT, normZ, normZinv) >= 100
    @test all(abs.(enc[1].points .- 1.0) .<= 0.02)

    A = [1.0 0.0; 0.0 -1.0]
    bA = BallMatrix(A)

    enc, errF, errT, normZ, normZinv = BallArithmetic.compute_enclosure_circles(
        bA, 2.0, 3.0, 0.01)
    @test enc[1].λ == 0.0
    @test BallArithmetic.bound_resolvent(enc[1], errF, errT, normZ, normZinv) >= 1
    @test all(abs.(enc[1].points) .- 2.0 .<= 0.02)

    enc, errF, errT, normZ, normZinv = BallArithmetic.compute_enclosure_circles(
        bA, 0.0, 0.1, 0.01)
    @test enc[1].λ == 0.0
    @test BallArithmetic.bound_resolvent(enc[1], errF, errT, normZ, normZinv) >= 1.0
    @test all(abs.((enc[1].points)) .- 0.1 .<= 0.02)

    E = BallArithmetic._compute_exclusion_circle_level_set_priori(A,
        1.0,
        0.01;
        rel_pearl_size = 1 / 64,
        max_initial_newton = 16)
    @test all([abs(E.points[i + 1] - E.points[i]) for i in 1:(length(E.points) - 1)] .<
              2 * E.radiuses[1])
    @test BallArithmetic.bound_resolvent(E, errF, errT, normZ, normZinv) > 100

    E = BallArithmetic._compute_exclusion_circle_level_set_ode(A,
        1.0,
        0.01; max_initial_newton = 16,
        max_steps = 1000,
        rel_steps = 16)
    @test BallArithmetic.bound_resolvent(E, errF, errT, normZ, normZinv) > 100
end
