"""
Test suite for methods from Horáček's PhD thesis.

Tests implementations from:
- Iterative methods (Gauss-Seidel, Jacobi)
- Direct methods (Gaussian elimination, HBR)
- Shaving and preconditioning
- Regularity testing
- Determinant computation
- Overdetermined systems
"""

using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "Horáček Methods" begin

# NOTE: Interval iterative methods (Gauss-Seidel, Jacobi) have fundamental convergence
# limitations due to the "wrapping effect" in interval arithmetic. Each iteration can
# introduce overestimation, causing interval widths to grow instead of shrink, even for
# diagonally dominant matrices where point arithmetic would converge.
#
# Reference: Horáček, J. (2012), "Interval linear and nonlinear systems", PhD thesis, §3.2
#
# For reliable interval linear system solving, use direct methods like:
# - interval_gaussian_elimination()
# - krawczyk_linear_system()
# The hull of the solution set of a regular real interval system is the hull of the solutions of
# its vertex systems, which are computed here in rational arithmetic.
function _vertex_solutions(Ac, AΔ, bc, bΔ)
    n = size(Ac, 1)
    C, D, c, d = Rational{BigInt}.(Ac), Rational{BigInt}.(AΔ), Rational{BigInt}.(bc),
        Rational{BigInt}.(bΔ)
    sols = Vector{Vector{Rational{BigInt}}}()
    for mask in 0:(2^(n * n + n) - 1)
        V = [C[i, j] + (isodd(mask >> ((j - 1) * n + i - 1)) ? D[i, j] : -D[i, j])
             for i in 1:n, j in 1:n]
        r = [c[i] + (isodd(mask >> (n * n + i - 1)) ? d[i] : -d[i]) for i in 1:n]
        push!(sols, V \ r)
    end
    return sols
end
_contains_all(x, sols) = all(
    all(big(mid(x)[i]) - big(rad(x)[i]) <= s[i] <= big(mid(x)[i]) + big(rad(x)[i])
        for i in eachindex(s)) for s in sols)
_hull_radius(sols) = maximum(
    Float64((maximum(s[i] for s in sols) - minimum(s[i] for s in sols)) / 2)
    for i in eachindex(sols[1]))

@testset "Iterative Methods" begin
    rng = MersenneTwister(20261011)
    @testset "initial enclosure, Jacobi and Gauss-Seidel contain the solution set" begin
        counts = Dict(:initial => 0, :jacobi => 0, :gs => 0, :jacobi_pre => 0, :gs_pre => 0)
        for n in (2, 3), _ in 1:(n == 2 ? 150 : 12)
            Ac = randn(rng, n, n) + 4I
            AΔ = rand(rng, n, n) .* 10.0^rand(rng, -4:-1)
            bc = randn(rng, n)
            bΔ = rand(rng, n) .* 10.0^rand(rng, -4:-1)
            A, b = BallMatrix(Ac, AΔ), BallVector(bc, bΔ)
            is_regular(A).is_regular || continue
            sols = _vertex_solutions(Ac, AΔ, bc, bΔ)
            for method in (:proposition, :norm, :auto)
                x0 = interval_initial_enclosure(A, b; method)
                x0 === nothing && continue
                counts[:initial] += 1
                @test _contains_all(x0, sols)
            end
            C = inv(Ac)
            for (key, f, pre) in ((:jacobi, interval_jacobi, nothing),
                (:gs, interval_gauss_seidel, nothing), (:jacobi_pre, interval_jacobi, C),
                (:gs_pre, interval_gauss_seidel, C))
                r = f(A, b; preconditioner = pre)
                isfinite(r.final_width) || continue
                counts[key] += 1
                @test !r.empty
                @test _contains_all(r.solution, sols)
                @test r.final_width == maximum(rad(r.solution))
            end
        end
        @test all(>(0), values(counts))
    end

    @testset "Gauss-Seidel Method" begin
        A = BallMatrix([5.0 1.0; 1.0 5.0], fill(0.001, 2, 2))
        b = BallVector([6.0, 6.0], fill(0.001, 2))
        result = interval_gauss_seidel(A, b)
        @test result.converged && result.iterations <= 20
        sols = _vertex_solutions(mid(A), rad(A), mid(b), rad(b))
        @test _contains_all(result.solution, sols)
        # an M-matrix: without preconditioning the method returns the hull (Section 5.7.4)
        @test result.final_width <= _hull_radius(sols) * (1 + 1e-3) + 1e-12
    end

    @testset "Jacobi Method" begin
        A = BallMatrix([6.0 1.0; 1.0 6.0], fill(0.001, 2, 2))
        b = BallVector([7.0, 7.0], fill(0.001, 2))
        result = interval_jacobi(A, b)
        @test result.converged && result.iterations <= 20
        @test _contains_all(result.solution, _vertex_solutions(mid(A), rad(A), mid(b), rad(b)))
    end

    @testset "a box that misses the solution set" begin
        A = BallMatrix([5.0 1.0; 1.0 5.0], fill(0.001, 2, 2))
        b = BallVector([6.0, 6.0], fill(0.001, 2))
        far = BallVector([10.0, 10.0], [0.5, 0.5])
        @test interval_gauss_seidel(A, b; x0 = far).empty
        @test interval_jacobi(A, b; x0 = far).empty
        @test_throws ArgumentError interval_jacobi(BallMatrix(randn(ComplexF64, 2, 2)),
            BallVector(randn(ComplexF64, 2)))
    end
end

@testset "Gaussian Elimination" begin
    @testset "Regular 3×3 System" begin
        A = BallMatrix([2.0 1.0 1.0; 4.0 3.0 3.0; 8.0 7.0 9.0],
                      fill(0.01, 3, 3))
        b = BallVector([4.0, 10.0, 24.0], fill(0.01, 3))

        result = interval_gaussian_elimination(A, b)

        @test result.success
        @test !result.singular
        @test length(result.solution) == 3

        # Verify solution
        x_c = mid(result.solution)
        A_c = mid(A)
        b_c = mid(b)
        @test norm(A_c * x_c - b_c) < 0.1
    end

    @testset "Determinant Computation" begin
        A = BallMatrix([2.0 1.0; 1.0 2.0], fill(0.01, 2, 2))

        det_result = interval_gaussian_elimination_det(A)

        # True determinant should be near 3.0
        @test 3.0 ∈ det_result
        @test mid(det_result) ≈ 3.0 atol=0.2
    end

    @testset "Singularity Detection" begin
        # Nearly singular matrix
        A = BallMatrix([1.0 1.0; 1.0 1.0], fill(0.001, 2, 2))

        is_reg = is_regular_gaussian_elimination(A)

        # Should detect singularity
        @test !is_reg
    end
end

@testset "HBR Method" begin
    rng = MersenneTwister(20261012)
    @testset "Theorem 5.12 contains the solution set" begin
        solved = Dict(:preconditioned => 0, :simple => 0)
        for n in (2, 3), _ in 1:(n == 2 ? 150 : 12)
            Ac = randn(rng, n, n) + 4I
            AΔ = rand(rng, n, n) .* 10.0^rand(rng, -4:-1)
            bc = randn(rng, n)
            bΔ = rand(rng, n) .* 10.0^rand(rng, -4:-1)
            A, b = BallMatrix(Ac, AΔ), BallVector(bc, bΔ)
            is_regular(A).is_regular || continue
            sols = _vertex_solutions(Ac, AΔ, bc, bΔ)
            for (key, r) in ((:preconditioned, hbr_method(A, b)), (:simple, hbr_method_simple(A, b)))
                r.success || continue
                solved[key] += 1
                @test _contains_all(r.solution, sols)
            end
        end
        @test all(>(0), values(solved))
    end

    @testset "a diagonal midpoint gives the hull" begin
        A = BallMatrix([4.0 0.0; 0.0 5.0], [0.1 0.2; 0.3 0.1])
        b = BallVector([1.0, 2.0], [0.1, 0.1])
        r = hbr_method_simple(A, b)
        sols = _vertex_solutions(mid(A), rad(A), mid(b), rad(b))
        @test r.success && _contains_all(r.solution, sols)
        @test maximum(rad(r.solution)) <= _hull_radius(sols) * (1 + 1e-10)
    end

    @testset "the radius of b is used, and failure is reported" begin
        r = hbr_method(BallMatrix(fill(1.0, 1, 1)), BallVector([3.0], [0.1]))
        @test r.success && rad(r.solution)[1] >= 0.1
        notH = BallMatrix([1.0 2.0; 2.0 1.0], fill(0.5, 2, 2))
        @test !hbr_method_simple(notH, BallVector([1.0, 1.0])).success
        @test all(isinf, rad(hbr_method_simple(notH, BallVector([1.0, 1.0])).solution))
    end
end

@testset "Preconditioning" begin
    @testset "Midpoint Preconditioner" begin
        A = BallMatrix([3.0 1.0; 1.0 2.0], fill(0.1, 2, 2))

        prec = compute_preconditioner(A, method=:midpoint)

        @test prec.success
        @test prec.method == MidpointInverse
        @test prec.condition_number < 100
    end

    @testset "LU Preconditioner" begin
        A = BallMatrix([4.0 2.0; 2.0 3.0], fill(0.05, 2, 2))

        prec = compute_preconditioner(A, method=:lu)

        @test prec.success
        @test prec.method == LUFactorization
        @test prec.factorization !== nothing
    end

    @testset "LDLT Preconditioner (Symmetric)" begin
        # Symmetric positive definite matrix
        A_sym = [3.0 1.0; 1.0 2.0]
        A = BallMatrix(A_sym, fill(0.05, 2, 2))

        prec = compute_preconditioner(A, method=:ldlt)

        @test prec.success
        # LDLT may fall back to LU if LDLT is not supported for the input type
        @test prec.method ∈ (LDLTFactorization, LUFactorization)
    end

    @testset "Preconditioner Quality Check" begin
        A = BallMatrix([5.0 1.0; 1.0 4.0], fill(0.1, 2, 2))

        prec = compute_preconditioner(A)

        @test is_well_preconditioned(A, prec, threshold=0.5)
    end

    @testset "the quality check is a bound over the whole ball" begin
        rng = MersenneTwister(20261013)
        for _ in 1:200
            Ac = randn(rng, 3, 3) + 3I
            AΔ = rand(rng, 3, 3) .* 0.2
            A = BallMatrix(Ac, AΔ)
            prec = compute_preconditioner(A; check_conditioning = false)
            C = prec.preconditioner
            # the largest ‖I − C Ã‖_∞ over the vertex matrices, in floating point
            worst = maximum(opnorm(I - C * (Ac + AΔ .* [isodd(m >> (k - 1)) ? 1 : -1
                                                        for k in LinearIndices(Ac)]), Inf)
            for m in 0:(2^9 - 1))
            for t in (0.3, 0.5, 0.9)
                is_well_preconditioned(A, prec; threshold = t) && @test worst < t * (1 + 1e-12)
            end
        end
        A = BallMatrix([5.0 1.0; 1.0 4.0], fill(0.1, 2, 2))
        prec = compute_preconditioner(A)
        @test rad(apply_preconditioner(prec, A)) != zeros(2, 2)
        @test apply_preconditioner(prec, BallVector([1.0, 2.0])) isa BallVector
    end
end

@testset "Regularity Testing" begin
    @testset "Sufficient Condition - Regular Matrix" begin
        # Well-separated eigenvalues
        A = BallMatrix([3.0 0.5; 0.5 2.0], fill(0.1, 2, 2))

        result = is_regular_sufficient_condition(A)

        @test result.is_regular
        @test result.certificate > 0
    end

    @testset "Gershgorin Test" begin
        # Diagonally dominant → regular
        A = BallMatrix([5.0 1.0 0.5; 1.0 4.0 0.5; 0.5 0.5 3.0],
                      fill(0.1, 3, 3))

        result = is_regular_gershgorin(A)

        @test result.is_regular
    end

    @testset "Diagonal Dominance Test" begin
        A = BallMatrix([10.0 1.0; 1.0 8.0], fill(0.1, 2, 2))

        result = is_regular_diagonal_dominance(A)

        @test result.is_regular
        @test result.method == :diagonal_dominance
    end

    @testset "Combined Regularity Test" begin
        A = BallMatrix([4.0 1.0; 1.0 3.0], fill(0.1, 2, 2))

        result = is_regular(A, verbose=false)

        @test result.is_regular
    end

    @testset "Singularity Test" begin
        # Matrix with small separation
        A = BallMatrix([1.0 1.0; 1.0 1.0], fill(0.5, 2, 2))

        # Should not be able to prove regularity
        result = is_regular(A)
        @test !result.is_regular
    end
end

@testset "Regularity and singularity against the vertex determinants" begin
    # det is affine in each entry, so over a real interval matrix its extreme values are taken
    # at vertex matrices: the matrix is regular exactly when all vertex determinants have one
    # strict sign. The determinants are computed in rational arithmetic.
    rng = MersenneTwister(20261011)
    function vertex_signs(Ac, AΔ)
        n = size(Ac, 1)
        C, D = Rational{BigInt}.(Ac), Rational{BigInt}.(AΔ)
        signs = Set{Int}()
        for mask in 0:(2^(n * n) - 1)
            V = [C[i, j] + (isodd(mask >> ((j - 1) * n + i - 1)) ? D[i, j] : -D[i, j])
                 for i in 1:n, j in 1:n]
            push!(signs, Int(sign(det(V))))
        end
        return signs
    end
    regular_methods = (A -> is_regular_sufficient_condition(A).is_regular,
        A -> is_regular_sufficient_condition(A; condition = 2).is_regular,
        A -> is_regular_spectral_radius(A).is_regular,
        A -> is_regular_diagonal_dominance(A).is_regular,
        A -> is_regular_gershgorin(A).is_regular,
        A -> is_regular(A).is_regular)
    proved_regular = zeros(Int, length(regular_methods))
    proved_singular = zeros(Int, 4)
    for n in (2, 3), _ in 1:400
        Ac = randn(rng, n, n) + (rand(rng, Bool) ? 2I : 0I)
        AΔ = rand(rng, n, n) .* 10.0^rand(rng, -3:0.5:0.5)
        A = BallMatrix(Ac, AΔ)
        signs = vertex_signs(Ac, AΔ)
        truly_regular = signs == Set([1]) || signs == Set([-1])
        for (k, f) in enumerate(regular_methods)
            if f(A)
                proved_regular[k] += 1
                @test truly_regular
            end
        end
        for c in 1:4
            if is_singular_sufficient_condition(A; conditions = (c,))
                proved_singular[c] += 1
                @test !truly_regular
            end
        end
    end
    # every condition was exercised on the positive side
    @test all(>(0), proved_regular)
    @info "regularity proved by $(proved_regular), singularity by $(proved_singular), of 800"
    # condition 1 needs σ_max(A_c) ≤ σ_min(A_Δ), which these samples do not produce; it is
    # exercised on a matrix built for it below
    @test all(>(0), proved_singular[[2, 3, 4]])

    # Examples 4.36 and 4.39 of the thesis
    A36 = BallMatrix([1.0 1.0; -1.0 1.0], [1.0 0.0; 0.0 1.0])
    @test !is_regular_spectral_radius(A36).is_regular
    A39 = BallMatrix([0.5 1.0; -1.0 0.5], [0.5 0.0; 0.0 0.5])
    r39 = is_regular_spectral_radius(A39)
    @test r39.is_regular && 0.6 <= r39.certificate < 0.6 + 1e-12

    # each condition of Theorem 11.13 on a matrix built for it
    Z = BallMatrix(zeros(3, 3), Matrix(1.0I, 3, 3))
    @test is_singular_sufficient_condition(Z; conditions = (1,))
    # condition 3 asks for an inverse that is nonnegative entrywise; an enclosure can show that
    # only where the entries are positive. Here A_Δ − |A_c| = [2 -1; -1 2]/3, with inverse
    # [2 1; 1 2], and the interval matrix contains [1/3 1/3; 1/3 1/3].
    A3 = BallMatrix([0.0 1/3; 1/3 0.0], [2/3 0.0; 0.0 2/3])
    @test is_singular_sufficient_condition(A3; conditions = (3,))
    @test !is_singular_sufficient_condition(Z; conditions = (3,))    # the inverse is I, with zeros
    @test is_singular_sufficient_condition(Z; conditions = (4,))
    @test is_singular_sufficient_condition(BallMatrix(Matrix(1.0I, 3, 3), Matrix(1.5I, 3, 3));
        conditions = (2,))
    @test !is_singular_sufficient_condition(BallMatrix(Matrix(1.0I, 3, 3), fill(1e-3, 3, 3)))
    @test_throws ArgumentError is_singular_sufficient_condition(BallMatrix(randn(ComplexF64, 2, 2)))
    @test_throws ArgumentError is_regular(Z; methods = [:nonsense])
end

@testset "Determinant Methods" begin
    rng = MersenneTwister(20261011)
    # det is affine in each entry: over a real interval matrix its range is the hull of the
    # vertex determinants, computed here in rational arithmetic
    function vertex_range(Ac, AΔ)
        n = size(Ac, 1)
        C, D = Rational{BigInt}.(Ac), Rational{BigInt}.(AΔ)
        dets = [det([C[i, j] + (isodd(mask >> ((j - 1) * n + i - 1)) ? D[i, j] : -D[i, j])
                     for i in 1:n, j in 1:n]) for mask in 0:(2^(n * n) - 1)]
        return minimum(dets), maximum(dets)
    end
    encloses(d, lo, hi) = isinf(rad(d)) ||
        (big(mid(d)) - big(rad(d)) <= lo && hi <= big(mid(d)) + big(rad(d)))
    methods = (:hadamard, :gershgorin, :cramer, :cofactor, :gaussian_elimination, :auto)
    finite = Dict(m => 0 for m in methods)
    for n in (1, 2, 3), _ in 1:300
        Ac = randn(rng, n, n) + (rand(rng, Bool) ? 3I : 0I)
        AΔ = rand(rng, n, n) .* 10.0^rand(rng, -6:0.5:0)
        A = BallMatrix(Ac, AΔ)
        lo, hi = vertex_range(Ac, AΔ)
        for m in methods
            r = interval_det(A; method = m)
            @test encloses(r.determinant, lo, hi)
            isfinite(rad(r.determinant)) && (finite[m] += 1)
            # a determinant enclosure away from zero is a proof of regularity
            contains_zero(r) || @test lo > 0 || hi < 0
        end
    end
    @test all(finite[m] > 0 for m in methods)

    @testset "the bunches of Figures 8.3 and 8.4" begin
        b = BallArithmetic._bunch_product_bounds
        for (a, bb) in ((0.5, 2.0), (-2.0, -0.5), (-1.0, 3.0), (-3.0, 1.0), (-2.0, 2.0)), k in 1:5
            lo, hi = b(a, bb, k)
            # every product of k numbers taken at the two ends, and of conjugate pairs of
            # modulus between the smallest and the largest modulus of [a, b]
            moduli = a > 0 ? (a, bb) : bb < 0 ? (-bb, -a) : (0.0, max(-a, bb))
            for j in 0:k
                p = big(a)^j * big(bb)^(k - j)
                @test lo <= p <= hi
                for pairs in 1:(k ÷ 2), m in moduli, i in 0:(k - 2pairs)
                    q = big(m)^(2pairs) * big(a)^i * big(bb)^(k - 2pairs - i)
                    @test lo <= q <= hi
                end
            end
        end
        @test b(1.0, 2.0, 1) == (1.0, 2.0) && b(-1.0, 3.0, 1) == (-1.0, 3.0)
    end

    @testset "examples" begin
        # diagonal with separated discs: the product of the diagonal intervals
        D = BallMatrix([2.0 0.0; 0.0 -3.0], [0.5 0.0; 0.0 0.5])
        r = det_gershgorin(D)
        @test encloses(r.determinant, big(-8.75), big(-3.75))
        @test rad(r.determinant) < 2.8        # the hull has radius 2.5; the ball product gives 2.75
        @test !contains_zero(r)
        # Hadamard on the identity gives 1; no sign
        h = det_hadamard(BallMatrix(Matrix(1.0I, 4, 4)))
        @test mid(h.determinant) == 0 && 1 <= rad(h.determinant) < 1 + 1e-12
        @test contains_zero(h)
        # Cramer on a point matrix
        c = det_cramer(BallMatrix([2.0 1.0; 1.0 3.0]))
        @test abs(mid(c.determinant) - 5) < 1e-12 && rad(c.determinant) < 1e-12
        @test interval_det(BallMatrix([2.0 1.0; 1.0 2.0], fill(0.01, 2, 2))).method == :cofactor
        @test interval_det(BallMatrix(Matrix(1.0I, 5, 5), fill(0.01, 5, 5))).method ==
              :gaussian_elimination
        @test_throws ArgumentError det_gershgorin(BallMatrix(randn(ComplexF64, 2, 2)))
        @test_throws ArgumentError interval_det(D; method = :nonsense)
        # complex input for the methods that take it
        Z = randn(rng, ComplexF64, 3, 3)
        exact = det(Complex{BigFloat}.(Z))
        for m in (:hadamard, :cramer, :cofactor)
            d = interval_det(BallMatrix(Z); method = m).determinant
            @test abs(exact - mid(d)) <= rad(d)
        end
    end
end

@testset "Overdetermined Systems" begin
    @testset "Subsquares Method" begin
        # 3×2 system with solution
        A = BallMatrix([2.0 1.0; 1.0 2.0; 3.0 1.0], fill(0.05, 3, 2))
        b = BallVector([3.0, 3.0, 4.0], fill(0.05, 3))

        result = subsquares_method(A, b, max_subsystems=10)

        @test result.solvable || result.subsystems_checked > 0
        if result.solvable
            @test length(result.solution) == 2
            @test result.residual < 1.0
        end
    end

    @testset "Multi-Jacobi Method" begin
        # Overdetermined system
        A = BallMatrix([3.0 1.0; 1.0 2.0; 2.0 3.0], fill(0.05, 3, 2))
        b = BallVector([4.0, 3.0, 5.0], fill(0.05, 3))

        result = multi_jacobi_method(A, b, max_iterations=50)

        # May or may not converge depending on system properties
        if result.solvable
            @test length(result.solution) == 2
        end
    end

    @testset "Interval Least Squares" begin
        # Overdetermined system without exact solution
        # Use smaller radii for tighter bounds
        A = BallMatrix([2.0 1.0; 1.0 2.0; 3.0 1.0], fill(0.01, 3, 2))
        b = BallVector([3.1, 2.9, 4.2], fill(0.01, 3))

        result = interval_least_squares(A, b)

        @test result.solvable
        @test length(result.solution) == 2
        @test result.method == :least_squares
        # Residual should be reasonable (depends on system conditioning)
        @test result.residual < 2.0
    end
end

@testset "Sherman-Morrison Formula" begin
    @testset "Inverse Update" begin
        # Test Sherman-Morrison formula correctness
        A = [3.0 1.0; 1.0 2.0]
        A_inv = inv(A)

        # Rank-1 update
        u = [1.0, 0.0]
        v = [0.0, 1.0]

        # Compute updated inverse
        A_updated_inv = sherman_morrison_inverse_update(A_inv, u, v)

        # Check correctness
        A_updated = A + u * v'
        A_updated_inv_direct = inv(A_updated)

        @test A_updated_inv ≈ A_updated_inv_direct atol=1e-10
    end
end

@testset "Integration Tests" begin
    @testset "Complete Solution Pipeline" begin
        # Test complete workflow: precondition → solve → verify
        A = BallMatrix([4.0 1.0 0.5; 1.0 3.0 0.5; 0.5 0.5 2.0],
                      fill(0.05, 3, 3))
        b = BallVector([5.5, 4.5, 3.0], fill(0.05, 3))

        # 1. Check regularity
        reg_result = is_regular(A)
        @test reg_result.is_regular

        # 2. Compute preconditioner
        prec = compute_preconditioner(A)
        @test prec.success

        # 3. Solve with Gauss-Seidel
        gs_result = interval_gauss_seidel(A, b)
        @test gs_result.converged
        @test _contains_all(gs_result.solution,
            _vertex_solutions(mid(A), rad(A), mid(b), rad(b)))
    end

    @testset "Method Comparison" begin
        # Compare different solution methods on same problem
        # This demonstrates why direct methods are preferred for interval systems
        A = BallMatrix([5.0 1.0; 1.0 5.0], fill(0.01, 2, 2))
        b = BallVector([6.0, 6.0], fill(0.01, 2))

        result_gs = interval_gauss_seidel(A, b)
        result_jacobi = interval_jacobi(A, b)
        result_ge = interval_gaussian_elimination(A, b)
        result_hbr = hbr_method(A, b)
        @test result_ge.success && result_gs.converged && result_jacobi.converged &&
              result_hbr.success
        sols = _vertex_solutions(mid(A), rad(A), mid(b), rad(b))
        for x in (result_gs.solution, result_jacobi.solution, result_ge.solution,
            result_hbr.solution)
            @test _contains_all(x, sols)
        end
    end
end

end  # Horáček Methods testset

println("\n✓ All Horáček method tests completed successfully!")
