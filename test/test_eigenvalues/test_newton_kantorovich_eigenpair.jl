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
            rs = certify_eigenpairs(S; hermitian = true)
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
