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
            @test length(BallArithmetic.gevbox(bA, bB)) == 3
            @test all(rad(b) < big(2.0)^-200 for b in r.eigenvalues)
            # the eigenvalues of the pencil, from the eigenvalues of B⁻¹A at twice the precision
            λ = setprecision(() -> eigvals(B \ A), BigFloat, 512)
            @test all(any(abs(l - mid(b)) <= rad(b) + big(2.0)^-240 for b in r.eigenvalues)
            for l in λ)
        end
    end
end


@testset "the kind of ball follows the type of the matrix" begin
    using LinearAlgebra, Random
    rng = MersenneTwister(12)
    setprecision(BigFloat, 256) do
        # a real matrix whose midpoint has real eigenvalues, while matrices in the ball have
        # complex ones: the balls are discs of the complex plane and contain them
        M = [1.0 1.0; 0.0 1.0 + 1e-8]
        A = BallMatrix(M, fill(1e-6, 2, 2))
        r = BallArithmetic.rigorous_eigenvalues(A)
        @test eltype(r.eigenvalues) == Ball{Float64, ComplexF64}
        X = [1.0 1.0; -1e-6 1.0 + 1e-8]                  # in the ball, eigenvalues 1 ± 1e-3 i
        λ = eigvals(big.(X))
        @test maximum(abs ∘ imag, λ) > 1e-4
        @test all(any(abs(l - mid(b)) <= rad(b) for b in r.eigenvalues) for l in λ)
        @test (Test.@inferred BallArithmetic.evbox(A)) isa Vector{Ball{Float64, ComplexF64}}
        @test (Test.@inferred BallArithmetic.gevbox(A, BallMatrix(Matrix(1.0I, 2, 2)))) isa
              Vector{Ball{Float64, ComplexF64}}

        # a point matrix that is Hermitian or real symmetric by its type: real intervals
        for S in (Float64, ComplexF64), n in (3, 8)
            B = randn(rng, S, n, n)
            Hm = (B + B') / 2
            for H in (S == Float64 ? (Hermitian(Hm), Symmetric(Hm), Hermitian(Hm, :L)) :
                      (Hermitian(Hm), Hermitian(Hm, :L)))
                v = BallArithmetic.evbox(H)
                @test v isa Vector{Ball{Float64, Float64}}
                λ = eigvals(Hermitian(big.(Matrix(H))))
                @test all(any(abs(l - mid(b)) <= rad(b) for b in v) for l in λ)
                @test maximum(rad, v) < 1e-10
            end
            # only the stored triangle is the matrix
            M2 = triu(Hm) + 7 .* tril(ones(S, n, n), -1)
            @test all(any(abs(l - mid(b)) <= rad(b) for b in BallArithmetic.evbox(Hermitian(M2)))
            for l in eigvals(Hermitian(big.(Hm))))
            # a ball matrix is an open set of matrices and is not Hermitian: no such method
            A = BallMatrix(Hm, fill(1e-9, n, n))
            @test_throws MethodError BallArithmetic.evbox(Hermitian(A))
            @test eltype(BallArithmetic.evbox(A)) == Ball{Float64, ComplexF64}
        end
        # a complex symmetric matrix is not Hermitian
        @test_throws MethodError BallArithmetic.evbox(Symmetric(ComplexF64[1 im; im 2]))
    end
end
