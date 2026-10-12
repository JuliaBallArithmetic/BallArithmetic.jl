@testset "Eigvals" begin
    using BallArithmetic

    A = BallMatrix(rand(256, 256))

    gev_result = BallArithmetic.rigorous_generalized_eigenvalues(A, A)
    v = BallArithmetic.gevbox(A, A)

    @test gev_result isa BallArithmetic.RigorousGeneralizedEigenvaluesResult
    @test gev_result.eigenvalues == v
    @test gev_result[1] == v[1]
    @test all(abs(gev_result[i].c - 1.0) < gev_result[i].r for i in 1:256)
    @test gev_result.coupling_defect_norm >= zero(radtype(first(v)))
    @test gev_result.projected_residual_norm >= zero(radtype(first(v)))

    bA = BallMatrix([125.0 0.00000001; 0.0 256.0])

    @test BallArithmetic.collatz_upper_bound(bA) >= 256.0

    ev_result = BallArithmetic.rigorous_eigenvalues(bA)
    v = BallArithmetic.evbox(bA)

    @test ev_result isa BallArithmetic.RigorousEigenvaluesResult
    @test ev_result.eigenvalues == v
    @test ev_result[1] == v[1]
    @test abs(ev_result[1].c - 125.0) <= ev_result[1].r
    @test ev_result.inverse_defect_norm >= zero(radtype(first(v)))
    @test ev_result.projected_residual_norm >= zero(radtype(first(v)))
end

@testset "generalised eigenvalues in BigFloat" begin
    setprecision(BigFloat, 256) do
        A = big.([4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]) / 3
        B = big.([2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0]) / 7
        for (bA, bB) in ((BallMatrix(A), BallMatrix(B)),
            (BallMatrix(complex(A)), BallMatrix(complex(B))))
            r = BallArithmetic.rigorous_generalized_eigenvalues(bA, bB)
            @test BallArithmetic.gevbox(bA, bB) == r.eigenvalues
            @test all(rad(b) < big(2.0)^-200 for b in r.eigenvalues)
            # the eigenvalues of the pencil, from the eigenvalues of B⁻¹A at twice the precision
            λ = setprecision(() -> eigvals(B \ A), BigFloat, 512)
            @test all(any(abs(l - mid(b)) <= rad(b) + big(2.0)^-240 for b in r.eigenvalues)
            for l in λ)
        end
    end
end

