# Tests of certify_eigenpair and certify_eigenpairs against 256-bit eigendecompositions.

using Test
using BallArithmetic
using LinearAlgebra
using Random
using GenericSchur

# the eigenpair of the reference nearest to λ, normalised by uᴴ w = 1
function _nk_reference(F, λ, u)
    k = argmin(abs.(F.values .- λ))
    w = F.vectors[:, k]
    return F.values[k], w / dot(Complex{BigFloat}.(u), w)
end

@testset "Newton-Kantorovich eigenpair" begin
    setprecision(BigFloat, 256) do
        @testset "the discriminant is rounded downwards" begin
            d = BallArithmetic._nk_discriminant
            @test d(0.0, 0.25, 1.0) <= 0.0
            @test d(0.5, 1.0, 0.01) <= 0.25 - 0.04
            # a product that rounds: 4 c y above, the difference below
            q, c, y = 0.1, 1 / 3, 0.3
            @test big(d(q, c, y)) <= (1 - big(q))^2 - 4 * big(c) * big(y)
            @test d(2.0, 1.0, 0.0) > 0        # a diagnostic only when q₀ ≥ 1
        end

        @testset "one pair, u given: containment and the normalisation" begin
            rng = MersenneTwister(11)
            for T in (Float64, ComplexF64), n in (3, 8, 15)
                M = randn(rng, T, n, n)
                E = eigen(M)
                U = inv(E.vectors)'
                F = eigen(Complex{BigFloat}.(M))
                for i in 1:n
                    res = BallArithmetic.certify_eigenpair(M, E.values[i], E.vectors[:, i];
                        u_approx = U[:, i])
                    @test res.certified
                    res.certified || continue
                    r = res.enclosure_radius
                    @test r < 1e-9
                    # u as the routine normalises it
                    u = U[:, i] / conj(dot(U[:, i], E.vectors[:, i]))
                    z, w = _nk_reference(F, E.values[i], u)
                    dz = z - mid(res.eigenvalue)
                    dw = w .- mid.(res.eigenvector)
                    # the pair is in the ball of radius r of ℂ × ℂⁿ
                    @test sqrt(abs2(dz) + sum(abs2, dw)) <= r
                    @test all(rad.(res.eigenvector) .== r)
                    @test rad(res.eigenvalue) == r
                end
            end
        end

        @testset "one pair, u computed, complex eigenvalue of a real matrix" begin
            # a rotation block gives the pair ±i up to the coupling
            M = [0.0 -1.0 0.3 0.0; 1.0 0.0 0.0 0.2; 0.1 0.0 2.0 0.5; 0.0 0.4 0.0 -3.0]
            E = eigen(M)
            i = findfirst(λ -> abs(imag(λ)) > 1e-3, E.values)
            @test i !== nothing
            res = BallArithmetic.certify_eigenpair(M, E.values[i], E.vectors[:, i])
            @test res.certified
            F = eigen(Complex{BigFloat}.(M))
            z = F.values[argmin(abs.(F.values .- E.values[i]))]
            @test abs(z - mid(res.eigenvalue)) <= rad(res.eigenvalue)
        end

        @testset "all pairs, and a ball matrix" begin
            rng = MersenneTwister(13)
            M = randn(rng, 7, 7)
            res = certify_eigenpairs(M)
            @test res.n_total == 7
            @test res.n_certified == 7
            @test length(res) == 7
            F = eigen(Complex{BigFloat}.(M))
            for p in res
                z = F.values[argmin(abs.(F.values .- mid(p.eigenvalue)))]
                @test abs(z - mid(p.eigenvalue)) <= rad(p.eigenvalue)
            end
            # a symmetric matrix through the Hermitian route
            S = Matrix(Symmetric(M + M'))
            rs = certify_eigenpairs(S)
            @test rs.n_certified == 7
            Fs = eigen(Hermitian(BigFloat.(S)))
            for p in rs
                @test minimum(abs.(Fs.values .- mid(p.eigenvalue))) <= rad(p.eigenvalue)
            end
            # a ball matrix: the enclosures hold for matrices of the ball
            ρ = 1e-9
            rb = certify_eigenpairs(BallMatrix(M, fill(ρ, 7, 7)))
            @test rb.n_certified == 7
            for _ in 1:5
                Mv = M .+ ρ .* rand(rng, (-1, 1), 7, 7)
                Fv = eigen(Complex{BigFloat}.(Mv))
                for p in rb
                    @test minimum(abs.(Fv.values .- mid(p.eigenvalue))) <= rad(p.eigenvalue)
                end
            end
            @test certify_eigenpairs(M; indices = [2, 5]).n_total == 2
        end

        @testset "what is not certified" begin
            # a double eigenvalue with a Jordan block: the Jacobian is singular at the pair
            J = [1.0 1.0; 0.0 1.0]
            res = BallArithmetic.certify_eigenpair(J, 1.0, [1.0, 0.0]; u_approx = [1.0, 0.0])
            @test !res.certified
            @test isinf(rad(res.eigenvalue))
            # a semisimple double eigenvalue
            res = BallArithmetic.certify_eigenpair([1.0 0.0; 0.0 1.0], 1.0, [1.0, 0.0];
                u_approx = [1.0, 0.0])
            @test !res.certified
            # a poor approximation
            M = [2.0 1.0; 1.0 3.0]
            res = BallArithmetic.certify_eigenpair(M, 10.0, [1.0, 1.0]; u_approx = [1.0, 1.0])
            @test !res.certified
            # u orthogonal to v
            res = BallArithmetic.certify_eigenpair(M, 2.0, [1.0, 0.0]; u_approx = [0.0, 1.0])
            @test !res.certified
            @test_throws ArgumentError BallArithmetic.certify_eigenpair(ones(2, 3), 1.0, [1.0, 0.0])
        end
    end
end

# The tests that came with the routine (commit e24453c).
@testset "Newton-Kantorovich Eigenpair Certifier" begin

    @testset "2×2 symmetric [2 1; 1 2]" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        eig = eigen(Hermitian(mid(A)))

        for i in 1:2
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.certified
            @test eig.values[i] ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-13
            @test result.defect_q < 1.0
            @test result.discriminant > 0.0
        end
    end

    @testset "Diagonal matrix diag(1, 5, 10)" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        eig = eigen(mid(A))

        for i in 1:3
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.certified
            @test eig.values[i] ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-14
        end
    end

    @testset "3×3 non-symmetric with real eigenvalues" begin
        A_mid = [3.0 1.0 0.0;
                 0.0 2.0 1.0;
                 0.0 0.0 1.0]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        for i in 1:3
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.certified
            @test real(eig.values[i]) ∈ result.eigenvalue
            @test result.enclosure_radius < 1e-12
        end
    end

    @testset "Complex eigenvalues (rotation-like matrix)" begin
        θ = π / 4
        A_mid = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        for i in 1:2
            result = certify_eigenpair(A, eig.values[i], eig.vectors[:, i])
            @test result.certified
            @test result.enclosure_radius < 1e-13
        end
    end

    @testset "Batch certify_eigenpairs" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0])
        result = certify_eigenpairs(A)

        @test result.n_total == 2
        @test result.n_certified == 2
        @test length(result) == 2

        for r in result
            @test r.certified
            @test r.enclosure_radius < 1e-13
        end
    end

    @testset "Selective indices certify_eigenpairs" begin
        A = BallMatrix(Diagonal([1.0, 5.0, 10.0]))
        result = certify_eigenpairs(A; indices=[1, 3])

        @test result.n_total == 2
        @test result.n_certified == 2
        @test length(result) == 2

        # First result corresponds to eigenvalue 1.0
        @test 1.0 ∈ result[1].eigenvalue
        # Second result corresponds to eigenvalue 10.0
        @test 10.0 ∈ result[2].eigenvalue
    end

    @testset "Near-defective matrix returns verified=false" begin
        # Jordan-like matrix with repeated eigenvalue
        ε = 1e-15
        A_mid = [1.0 1.0; ε 1.0]
        A = BallMatrix(A_mid)
        eig = eigen(A_mid)

        # At least one eigenpair should fail to certify (near-defective)
        results = [certify_eigenpair(A, eig.values[i], eig.vectors[:, i]) for i in 1:2]
        # The pair is nearly defective so at least one should fail or have large radius
        any_failed = any(r -> !r.certified || r.enclosure_radius > 1e-3, results)
        @test any_failed
    end

    @testset "Result struct fields" begin
        A = BallMatrix([2.0 0.0; 0.0 3.0])
        result = certify_eigenpair(A, 2.0, [1.0, 0.0])

        @test result isa NKEigenpairResult{Float64, Float64}
        @test result.certified == true
        @test result.eigenvalue isa Ball{Float64, Float64}
        @test length(result.eigenvector) == 2
        @test result.enclosure_radius isa Float64
        @test result.defect_q isa Float64
        @test result.C_norm isa Float64
        @test result.residual_y isa Float64
        @test result.discriminant isa Float64
    end

    @testset "NKEigenpairsResult iteration" begin
        A = BallMatrix(Diagonal([1.0, 2.0, 3.0]))
        result = certify_eigenpairs(A)

        @test length(result) == 3
        @test firstindex(result) == 1
        @test lastindex(result) == 3

        # Test iteration
        count = 0
        for r in result
            count += 1
            @test r isa NKEigenpairResult
        end
        @test count == 3
    end

    @testset "Non-hermitian batch" begin
        A_mid = [2.0 1.0; 0.5 3.0]
        A = BallMatrix(A_mid)
        result = certify_eigenpairs(A)

        @test result.n_total == 2
        @test result.n_certified == 2
    end

    @testset "norm_method=:svd option" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0])
        eig = eigen(Hermitian(mid(A)))
        result = certify_eigenpair(A, eig.values[1], eig.vectors[:, 1]; norm_method=:svd)
        @test result.certified
    end
end

@testset "certify_eigenpairs: a complex Hermitian matrix through the Hermitian route" begin
    # the eigenvalues of the midpoint are real numbers and the certified pairs are complex:
    # this threw a MethodError on the element type of the result
    B = ComplexF64[2 1im 0; -1im 3 0.5; 0 0.5 -1]
    rs = certify_eigenpairs(B)
    @test rs.n_certified == 3
    μ = eigvals(Hermitian(B))
    @test all(any(abs(l - mid(p.eigenvalue)) <= rad(p.eigenvalue) + 1e-12 for p in rs) for l in μ)
    # the keyword that chose the eigendecomposition is gone: it is chosen from the midpoint
    @test_throws MethodError certify_eigenpairs(B; hermitian = true)
    # with radii the statement is for every matrix of the ball, Hermitian or not
    rb = certify_eigenpairs(BallMatrix(B, fill(1e-9, 3, 3)))
    @test rb.n_certified == 3
    X = B + 1e-9 .* [0 1 0; 1 0 0; 0 0 0]
    @test all(any(abs(l - mid(p.eigenvalue)) <= rad(p.eigenvalue) for p in rb) for l in eigvals(X))
end
