# The three kinds of ball matrix that the storage types distinguish (see the docstring of
# BallMatrix): a plain one, one around a Hermitian midpoint, and a HermitianBallMatrix, the
# Hermitian matrices within the tolerances.

using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "HermitianBallMatrix" begin
    rng = MersenneTwister(20261013)
    setprecision(BigFloat, 256) do
        herm(S, n) = (B = randn(rng, S, n, n); (B + B') / 2)
        # a Hermitian matrix within the tolerances of (M, R), near their ends
        function hmember(M, R)
            n = size(M, 1)
            E = zeros(Complex{BigFloat}, n, n)
            for j in 1:n, i in 1:j
                # just inside the tolerance: a phase computed in floating point has modulus 1
                # only up to rounding
                e = big(R[i, j]) * 0.999 * (i == j || eltype(M) <: Real ? big(rand(rng, (-1, 1))) :
                     cis(big(2pi * rand(rng))))
                E[i, j] = e
                E[j, i] = conj(e)
            end
            return Complex{BigFloat}.(M) .+ E
        end
        inball(l, b) = (mid(b) isa Real ? abs(imag(l)) <= big(1e-40) : true) &&
                       abs(l - mid(b)) <= rad(b) + big(1e-40)

        @testset "construction and kind, $S" for S in (Float64, ComplexF64)
            n = 4
            M, R = herm(S, n), Matrix(Symmetric(rand(rng, n, n) .* 1e-6))
            A = HermitianBallMatrix(M, R)
            @test A isa HermitianBallMatrix && A isa BallMatrix
            @test mid(A) isa Hermitian && rad(A) isa Symmetric
            @test HermitianBallMatrix(BallMatrix(M, R)) isa HermitianBallMatrix
            @test HermitianBallMatrix(M) isa HermitianBallMatrix && iszero(rad(HermitianBallMatrix(M)))
            @test HermitianBallMatrix(Hermitian(M)) isa HermitianBallMatrix
            @test HermitianBallMatrix(A) === A
            # only one triangle is read
            M2 = triu(M) + 9 .* tril(ones(S, n, n), -1)
            @test Matrix(mid(HermitianBallMatrix(M2, R))) == Matrix(Hermitian(M))
            @test Matrix(mid(HermitianBallMatrix(M2', R; uplo = :L))) == Matrix(Hermitian(M))
            # the other kinds are not of this one
            @test !(BallMatrix(M, R) isa HermitianBallMatrix)
            @test !(BallMatrix(Hermitian(M), R) isa HermitianBallMatrix)
            @test !(BallMatrix(M, Symmetric(R)) isa HermitianBallMatrix)
            S <: Complex && @test !(BallMatrix(Symmetric(M), Symmetric(R)) isa HermitianBallMatrix)
            @test_throws DimensionMismatch HermitianBallMatrix(randn(2, 3), rand(2, 3))
        end

        @testset "the three kinds and the enclosures of the eigenvalues, $S" for S in (
            Float64, ComplexF64)
            n = 5
            M = herm(S, n)
            R = Matrix(Symmetric(rand(rng, n, n) .* 1e-5))
            plain, around, hball = BallMatrix(M, R), BallMatrix(Hermitian(M), R),
            HermitianBallMatrix(M, R)
            routines = (evbox, A -> rigorous_eigenvalues(A).eigenvalues,
                A -> schur_gershgorin_enclosure(A).cluster_intervals,
                A -> rigorous_block_schur(A).cluster_intervals)
            # a matrix of the entrywise ball that is not Hermitian
            K = [i < j ? 1 : (i > j ? -1 : 0) for i in 1:n, j in 1:n]
            X = Complex{BigFloat}.(M) .+ big.(R) .* K
            for f in routines
                # kinds 1 and 2: discs of the complex plane, for every matrix of the ball
                for A in (plain, around)
                    b = f(A)
                    @test eltype(b) == Ball{Float64, ComplexF64}
                    @test all(any(inball(l, x) for x in b) for l in eigvals(X))
                    @test all(any(inball(l, x) for x in b) for l in eigvals(hmember(M, R)))
                end
                # kind 3: real intervals, for the Hermitian matrices within the tolerances
                b = f(hball)
                @test eltype(b) == Ball{Float64, Float64}
                for _ in 1:5
                    @test all(any(inball(l, x) for x in b) for l in eigvals(Hermitian(hmember(M, R))))
                end
            end
            # the same matrices, the same radii: the kind does not change the size of the sets
            d, i = evbox(plain), evbox(hball)
            @test maximum(rad, i) <= maximum(rad, d) * (1 + 1e-8)
            # Rump and Lange for the Hermitian kind; refused for a plain ball matrix
            r = verifyeigall(hball)
            @test all(r.certified)
            for _ in 1:5
                λ = eigvals(Hermitian(hmember(M, R)))
                @test all(any(abs(l - r.centers[k]) <= r.radii[k] + big(1e-40) for k in eachindex(r.radii)) for l in λ)
            end
            @test verifyeigall(Hermitian(M)).spectrum_covered
            @test_throws ArgumentError verifyeigall(plain; method = :rumplange2023)
            @test verifyeigall(plain) isa VerifyEigAllResult            # the general algorithm
        end

        @testset "which operations keep the kind, $S" for S in (Float64, ComplexF64)
            n = 4
            M, N = herm(S, n), herm(S, n)
            R, Q = Matrix(Symmetric(rand(rng, n, n) .* 1e-4)), Matrix(Symmetric(rand(rng, n, n) .* 1e-4))
            A, B = HermitianBallMatrix(M, R), HermitianBallMatrix(N, Q)
            D = BallMatrix(randn(rng, S, n, n), rand(rng, n, n) .* 1e-4)
            ops = Pair{String, Function}[
                "A + B" => (X, Y) -> X + Y, "A - B" => (X, Y) -> X - Y, "-A" => (X, Y) -> -X,
                "2.5A" => (X, Y) -> 2.5 * X, "A * (-3)" => (X, Y) -> X * (-3),
                "(1 + 2im)A" => (X, Y) -> (1 + 2im) * X, "A * B" => (X, Y) -> X * Y,
                "A * A" => (X, Y) -> X * X, "A'" => (X, Y) -> X', "transpose(A)" => (X, Y) -> transpose(X),
                "A + I" => (X, Y) -> X + I, "A - 2I" => (X, Y) -> X - 2I,
                "A + (1 + 2im)I" => (X, Y) -> X + (1 + 2im) * I,
                "A + N" => (X, Y) -> X + (X isa BallMatrix ? Hermitian(N) : Complex{BigFloat}.(N)),
                "A[1:2, 1:2]" => (X, Y) -> X[1:2, 1:2], "A' * A" => (X, Y) -> X' * X]
            kept = String[]
            for (name, op) in ops
                C = try
                    op(A, B)
                catch
                    nothing
                end
                C === nothing && continue          # the operation is not defined for these types
                C isa BallMatrix || continue
                if C isa HermitianBallMatrix
                    push!(kept, name)
                    for _ in 1:6
                        X, Y = hmember(M, R), hmember(N, Q)
                        Z = op(X, Y)
                        # the exact result is Hermitian and within the tolerances of the result
                        @test Z == Z'
                        @test all(abs.(Z .- Complex{BigFloat}.(Matrix(mid(C)))) .<= big.(Matrix(rad(C))))
                    end
                end
            end
            # sums, differences, the opposite, real multiples and real shifts keep the kind
            @test issubset(["A + B", "A - B", "-A", "2.5A", "A * (-3)", "A'", "A + I", "A - 2I",
                    "A + N"], kept)
            # products and complex multiples do not
            @test isempty(intersect(["A * B", "A * A", "(1 + 2im)A", "A + (1 + 2im)I", "A' * A"], kept))
            # with a plain ball matrix the result is plain
            for C in (A + D, D + A, A - D, A * D, D * A)
                @test !(C isa HermitianBallMatrix)
            end
        end

        @testset "the symmetric pencil takes the symmetric kinds" begin
            n = 4
            A, G = herm(Float64, n), herm(Float64, n)
            B = G * G + n * I
            B = (B + B') / 2
            F = eigen(Symmetric(A), Symmetric(B))
            r = verify_generalized_eigenpairs(Symmetric(A), Symmetric(B), F.vectors, F.values)
            @test r.certified
            Rt = Matrix(Symmetric(fill(1e-9, n, n)))
            rb = verify_generalized_eigenpairs(HermitianBallMatrix(A, Rt), HermitianBallMatrix(B, Rt),
                F.vectors, F.values)
            @test rb.certified
            # symmetric matrices within the tolerances: their eigenvalues are in the intervals
            for _ in 1:5
                Ax, Bx = real.(hmember(A, Rt)), real.(hmember(B, Rt))
                λ = sort(real.(eigvals(Bx \ Ax)))
                @test all(rb.eigenvalue_intervals[i][1] <= λ[i] <= rb.eigenvalue_intervals[i][2] for i in 1:n)
            end
            # a plain matrix or a plain ball matrix does not state the hypothesis
            @test_throws ArgumentError verify_generalized_eigenpairs(A, B, F.vectors, F.values)
            @test_throws ArgumentError verify_generalized_eigenpairs(BallMatrix(A, Rt),
                BallMatrix(B, Rt), F.vectors, F.values)
        end
    end
end
