using Random

@testset "Miyajima 2010 two-residual GEV enclosure" begin
    setprecision(BigFloat, 512)
    Random.seed!(4242)

    _encloses(res, tr) = all(minimum(abs(Complex{BigFloat}(λ) - Complex{BigFloat}(c))
                             for c in res.centers) <= res.radius for λ in tr)

    @testset "beta bound covers the whole ball" begin
        # The old Theorem-10 implementation read only B.c, so it returned the
        # same β however large the radius grew; and on failure it fell back to
        # sqrt(cond(B.c)), which bounds the wrong quantity.
        Bc = [2.0 0.5; 0.5 2.0]

        # exact input: β must still bound √‖B⁻¹‖₂
        β0 = compute_beta_bound(BallMatrix(Bc))
        @test β0 >= sqrt(opnorm(inv(BigFloat.(Bc)), 2))

        # β must be monotone in the radius, and must cover perturbed matrices
        βs = [compute_beta_bound(BallMatrix(Bc, fill(r, 2, 2)))
              for r in (0.0, 1e-8, 1e-3, 0.5)]
        @test issorted(βs)

        B_wide = BallMatrix(Bc, fill(0.5, 2, 2))
        β_wide = compute_beta_bound(B_wide)
        # One aggregated assertion instead of one per sample.
        violations = 0
        for _ in 1:200
            E = (2rand(2, 2) .- 1) .* 0.5
            E = (E + E') / 2
            Bp = Bc + E
            if all(eigvals(Symmetric(Bp)) .> 0)
                β_wide >= sqrt(opnorm(inv(BigFloat.(Bp)), 2)) || (violations += 1)
            end
        end
        @test violations == 0

        # a ball containing singular matrices cannot be certified
        @test compute_beta_bound(BallMatrix(Bc, fill(1.4, 2, 2))) == Inf

        # small-scale B: sqrt(cond) would have undershot by √‖B‖
        for s in (1e-2, 1e-4, 1e-6)
            Bs = Matrix(s * I, 4, 4)
            @test compute_beta_bound(BallMatrix(Bs)) >=
                  sqrt(opnorm(inv(BigFloat.(Bs)), 2))
        end

        # very ill-conditioned B: the old code took the non-rigorous fallback
        for c in (1e15, 1e16, 1e18)
            Bi = Matrix(Diagonal([1.0, 1.0, 1.0, 1 / c]))
            β = compute_beta_bound(BallMatrix(Bi))
            @test β == Inf || β >= sqrt(opnorm(inv(BigFloat.(Bi)), 2))
        end
    end

    @testset "Theorem 1 encloses every eigenvalue" begin
        for trial in 1:20
            n = rand(2:5)
            Ac = randn(n, n)
            Bc = randn(n, n)
            Bc = Matrix(Bc'Bc + n * I)
            trial % 2 == 0 && (Ac = Matrix((Ac + Ac') / 2))

            F = eigen(Bc \ Ac)
            res = miyajima_gev_enclosure(BallMatrix(complex(Ac)), BallMatrix(complex(Bc)),
                complex(Matrix(F.vectors)), collect(F.values))
            @test res.success
            @test res.nrmR2 < 1
            @test isfinite(res.radius)
            @test _encloses(res, eigvals(BigFloat.(Bc) \ BigFloat.(Ac)))
        end
    end

    @testset "no symmetry or definiteness required" begin
        # A not symmetric — outside the hypotheses of the symmetric-definite path
        Ac = [4.0 1.0; -2.0 3.0]
        Bc = [2.0 0.5; 0.5 2.0]
        F = eigen(Bc \ Ac)
        res = miyajima_gev_enclosure(BallMatrix(complex(Ac)), BallMatrix(complex(Bc)),
            complex(Matrix(F.vectors)), collect(F.values))
        @test res.success
        @test _encloses(res, eigvals(BigFloat.(Bc) \ BigFloat.(Ac)))
    end

    @testset "the enclosure widens with the input interval" begin
        Ac = complex([4.0 1.0; 1.0 3.0])
        Bc = complex([2.0 0.5; 0.5 2.0])
        radii = Float64[]
        for r in (0.0, 1e-12, 1e-8, 1e-4)
            A = BallMatrix(Ac, fill(r, 2, 2))
            B = BallMatrix(Bc, fill(r, 2, 2))
            F = eigen(mid(B) \ mid(A))
            res = miyajima_gev_enclosure(A, B, Matrix(F.vectors), collect(F.values))
            @test res.success
            push!(radii, res.radius)
            # the true eigenvalues of the centre pencil stay enclosed
            @test _encloses(res, eigvals(BigFloat.(real(mid(B))) \
                                         BigFloat.(real(mid(A)))))
        end
        @test issorted(radii)
    end

    @testset "Y is free" begin
        # Correctness must not depend on the quality of Y: a deliberately poor
        # Y only inflates ‖R₂‖ (and hence ε), never invalidates the enclosure.
        Ac = complex([4.0 1.0; 1.0 3.0])
        Bc = complex([2.0 0.5; 0.5 2.0])
        F = eigen(Bc \ Ac)
        X̃ = Matrix(F.vectors)
        λ̃ = collect(F.values)
        A = BallMatrix(Ac)
        B = BallMatrix(Bc)

        best = miyajima_gev_enclosure(A, B, X̃, λ̃)
        Ypoor = inv(mid(B * BallMatrix(X̃))) .* (1 + 1e-6)
        poor = miyajima_gev_enclosure(A, B, X̃, λ̃; Y = Ypoor)

        @test best.success && poor.success
        @test poor.radius >= best.radius            # worse Y ⇒ looser, still valid
        tr = eigvals(BigFloat.(real(Bc)) \ BigFloat.(real(Ac)))
        @test _encloses(best, tr)
        @test _encloses(poor, tr)
    end

    @testset "failure is reported, not faked" begin
        Ac = complex([4.0 1.0; 1.0 3.0])
        Bc = complex([2.0 0.5; 0.5 2.0])
        # a hopeless X̃ drives ‖R₂‖ ≥ 1
        res = miyajima_gev_enclosure(BallMatrix(Ac), BallMatrix(Bc),
            complex([1.0 1.0; 1.0 1.0000001]), [1.0, 2.0])
        @test res.success == false || res.nrmR2 < 1

        @test_throws DimensionMismatch miyajima_gev_enclosure(
            BallMatrix(Ac), BallMatrix(Bc), complex(rand(3, 3)), [1.0, 2.0])
    end
end

@testset "Corollary 3.2 per-eigenvalue radii" begin
    setprecision(BigFloat, 512)
    Random.seed!(606)

    @testset "never worse than Theorem 1's uniform ε, and still rigorous" begin
        for _ in 1:12
            n = rand(3:6)
            Ac = randn(n, n)
            Bc = randn(n, n)
            Bc = Matrix(Bc'Bc + n * I)
            F = eigen(Bc \ Ac)
            res = miyajima_gev_enclosure(BallMatrix(complex(Ac)), BallMatrix(complex(Bc)),
                complex(Matrix(F.vectors)), collect(F.values))
            @test res.success
            @test length(res.radii) == n
            @test all(res.radii .>= 0)

            # Corollary 3.2 ≤ Theorem 1
            ε_thm1 = res.nrmR1 / (1 - res.nrmR2)
            @test all(res.radii .<= ε_thm1 * (1 + 1e-12))
            # the reported scalar radius is the tighter of the two
            @test res.radius <= ε_thm1 * (1 + 1e-12)
            @test res.radius >= maximum(res.radii) * (1 - 1e-12)

            # rigor: every eigenvalue inside its own per-row disc
            tr = eigvals(Complex{BigFloat}.(BigFloat.(Bc) \ BigFloat.(Ac)))
            for λ in tr
                @test any(
                    i -> abs(Complex{BigFloat}(λ) -
                             Complex{BigFloat}(res.centers[i])) <= res.radii[i],
                    1:n)
            end
        end
    end

    @testset "failure paths carry an empty radii vector" begin
        Ac = complex([4.0 1.0; 1.0 3.0])
        Bc = complex([2.0 0.5; 0.5 2.0])
        res = miyajima_gev_enclosure(BallMatrix(Ac), BallMatrix(Bc),
            complex([1.0 1.0; 1.0 1.0]), [1.0, 2.0])
        @test res.success == false || res.nrmR2 < 1
        res.success || @test isempty(res.radii)
    end
end
