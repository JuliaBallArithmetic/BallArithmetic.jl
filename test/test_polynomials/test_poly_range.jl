@testset "certified polynomial range (bisection vs critical-point)" begin
    # ascending coeffs ↦ p(x) = Σ cₖ xᵏ ; dense reference range over a fine grid
    pval(c, x) = evalpoly(x, c)
    function ref_range(c, a, b; m = 200_001)
        xs = range(a, b; length = m)
        vs = pval.(Ref(c), xs)
        return minimum(vs), maximum(vs)
    end
    # a certified enclosure must CONTAIN the true range; brackets must be ordered
    function sound(R, tlo, thi)
        @test R.min_lo <= R.min_hi
        @test R.max_lo <= R.max_hi
        @test R.min_lo <= tlo + 1e-7          # encloses the true minimum from below
        @test R.max_hi >= thi - 1e-7          # encloses the true maximum from above
    end

    @testset "smoke / API" begin
        c = [0.0, -1.0, 0.0, 1.0]             # x³ − x
        Rb = range_bisect(c, -2.0, 2.0)
        Rc = range_critical(c, -2.0, 2.0)
        @test Rb isa PolyRange && Rc isa PolyRange
        @test Rb.method === :bisection && Rc.method === :critical
        for R in (Rb, Rc)
            sound(R, -6.0, 6.0)
            @test R.min_lo ≈ -6 atol = 1e-6
            @test R.max_hi ≈ 6 atol = 1e-6
        end
        @test Rc.diag.ncrit == 2              # x = ±1/√3 both interior
    end

    @testset "monotone & extrema at endpoints" begin
        c = [1.0, 2.0]                        # 1 + 2x, linear ⇒ no interior critical pt
        for R in (range_bisect(c, 0.0, 3.0), range_critical(c, 0.0, 3.0))
            sound(R, 1.0, 7.0)
        end
        @test range_critical(c, 0.0, 3.0).diag.ncrit == 0
    end

    @testset "soundness & agreement on random polynomials" begin
        rng = MersenneTwister(7)
        for trial in 1:25
            d = rand(rng, 2:9)
            c = randn(rng, d + 1)
            a = -1.0 - rand(rng);
            b = 1.0 + rand(rng)
            tlo, thi = ref_range(c, a, b)
            Rb = range_bisect(c, a, b; tol = 1e-9)
            Rc = range_critical(c, a, b)
            sound(Rb, tlo, thi)
            sound(Rc, tlo, thi)
            # both pin the extrema to within their reported brackets ⇒ they agree
            @test Rb.min_lo ≈ Rc.min_lo atol = 1e-5
            @test Rb.max_hi ≈ Rc.max_hi atol = 1e-5
        end
    end

    @testset "Chebyshev T_n — known range [-1,1], non-normal companion" begin
        # T_n via recurrence; on [-1,1] the range is exactly [-1,1]
        cheb(n) = (Tm = [1.0]; T = [0.0, 1.0];
            for _ in 2:n
                Tn = zeros(length(T) + 1)
                for i in eachindex(T)
                    Tn[i + 1] += 2T[i]
                end
                for i in eachindex(Tm)
                    Tn[i] -= Tm[i]
                end
                (Tm, T) = (T, Tn)
            end;
            n == 0 ? [1.0] : n == 1 ? [0.0, 1.0] : T)
        for n in (4, 8, 12)
            c = cheb(n)
            Rc = range_critical(c, -1.0, 1.0)
            sound(Rc, -1.0, 1.0)
            @test Rc.diag.ncrit == n - 1      # all n−1 interior extrema found
            @test Rc.max_hi - Rc.min_lo < 2 + 1e-4   # tight about [-1,1]
        end
    end

    @testset "clustered critical points (ill-conditioned companion)" begin
        ε = 1e-4                              # (x²−ε)² : roots of p' are 0, ±√ε (clustered)
        c = [ε^2, 0.0, -2ε, 0.0, 1.0]
        tlo, thi = 0.0, (1 - ε)^2
        for R in (range_bisect(c, -1.0, 1.0; tol = 1e-10), range_critical(c, -1.0, 1.0))
            sound(R, tlo, thi)
        end
    end

    @testset "evaluation never exceeds a finer enclosure (rigor)" begin
        # the critical-point enclosure must contain the bisection enclosure's truth bracket
        rng = MersenneTwister(11)
        for _ in 1:10
            c = randn(rng, 6)
            Rb = range_bisect(c, -1.5, 1.5; tol = 1e-11)
            Rc = range_critical(c, -1.5, 1.5)
            @test Rc.min_lo <= Rb.min_hi + 1e-7   # critical min-bracket below bisection's upper
            @test Rc.max_hi >= Rb.max_lo - 1e-7
        end
    end
end
