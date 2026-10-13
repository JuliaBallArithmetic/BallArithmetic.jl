# Operations on ball matrices stored as Diagonal, triangular, Hermitian or plain. A ball matrix is
# the set of the matrices whose stored entries lie in the stored balls (docstring of BallMatrix),
# so the storage of a result is a statement: here every operation is applied to members of the
# operands and the exact result must be in the set that the returned ball matrix stands for.

using Test
using BallArithmetic
using LinearAlgebra
using Random

@testset "operations on structured ball matrices" begin
    rng = MersenneTwister(31)
    n = 4
    setprecision(BigFloat, 256) do
        big_(X) = Complex{BigFloat}.(X)
        # members are taken at 0.999 of the tolerances: a phase computed in floating point has
        # modulus 1 only up to rounding
        signs(S, dims...) = S <: Complex ? cis.(big.(2pi .* rand(rng, dims...))) :
                            big.(rand(rng, (-1, 1), dims...))
        function make(k, S)
            M = randn(rng, S, n, n)
            R = rand(rng, n, n) .* 1e-3
            if k == :plain
                return BallMatrix(M, R), () -> big_(M) .+ big.(R) .* 0.999 .* signs(S, n, n)
            elseif k == :diagonal
                d, r = diag(M), diag(R)
                return BallMatrix(Diagonal(d), Diagonal(r)),
                () -> Matrix(Diagonal(big_(d) .+ big.(r) .* 0.999 .* signs(S, n)))
            elseif k == :upper
                return BallMatrix(UpperTriangular(M), UpperTriangular(R)),
                () -> triu(big_(M) .+ big.(R) .* 0.999 .* signs(S, n, n))
            elseif k == :lower
                return BallMatrix(LowerTriangular(M), LowerTriangular(R)),
                () -> tril(big_(M) .+ big.(R) .* 0.999 .* signs(S, n, n))
            elseif k == :hermitian
                H = (M + M') / 2
                Rs = Matrix(Symmetric(R))
                return HermitianBallMatrix(H, Rs), () -> begin
                    E = zeros(Complex{BigFloat}, n, n)
                    for j in 1:n, i in 1:j
                        e = big(Rs[i, j]) * 0.999 * (i == j ? big(rand(rng, (-1, 1))) : signs(S, 1)[1])
                        E[i, j] = e
                        E[j, i] = conj(e)
                    end
                    big_(H) .+ E
                end
            else    # a Hermitian midpoint with a plain radius: the full ball around it
                H = (M + M') / 2
                return BallMatrix(Hermitian(H), R), () -> big_(H) .+ big.(R) .* 0.999 .* signs(S, n, n)
            end
        end
        # the set that the ball matrix C stands for contains Z
        function contains(C, Z)
            ok = all(abs.(Z .- big_(Matrix(mid(C)))) .<= big.(Matrix(rad(C))))
            C isa HermitianBallMatrix && (ok &= Z == Z')
            return ok
        end
        kinds = (:plain, :diagonal, :upper, :lower, :hermitian, :hermitian_midpoint)
        unary = ("-A" => X -> -X, "2.5A" => X -> 2.5 * X, "(1+2im)A" => X -> (1 + 2im) * X,
            "A'" => X -> X', "transpose(A)" => X -> transpose(X), "A + I" => X -> X + I,
            "A - 2I" => X -> X - 2I, "3I - A" => X -> 3I - X,
            "A + (1+2im)I" => X -> X + (1 + 2im) * I, "A * A" => X -> X * X,
            "A' * A" => X -> X' * X)
        binary = ("A + B" => (X, Y) -> X + Y, "A - B" => (X, Y) -> X - Y, "A * B" => (X, Y) -> X * Y)

        @testset "$S" for S in (Float64, ComplexF64)
            for (name, f) in unary, ka in kinds
                A, sa = make(ka, S)
                C = f(A)
                @test C isa BallMatrix
                @test all(contains(C, f(sa())) for _ in 1:4)
            end
            for (name, f) in binary, ka in kinds, kb in kinds
                A, sa = make(ka, S)
                B, sb = make(kb, S)
                C = f(A, B)
                @test C isa BallMatrix
                @test all(contains(C, f(sa(), sb())) for _ in 1:4)
            end
            # what keeps the storage, on a few cases where it must and where it must not
            D, _ = make(:diagonal, S)
            U, _ = make(:upper, S)
            L, _ = make(:lower, S)
            H, _ = make(:hermitian, S)
            @test mid(2.5 * D) isa Diagonal && rad(2.5 * D) isa Diagonal
            @test mid(2.5 * U) isa UpperTriangular && rad(2.5 * U) isa UpperTriangular
            @test rad(D + U) isa UpperTriangular && rad(U + U) isa UpperTriangular
            @test rad(U * U) isa UpperTriangular && rad(D * D) isa Diagonal
            @test rad(U') isa LowerTriangular && rad(transpose(L)) isa UpperTriangular
            @test rad(U + L) isa Matrix && rad(U * L) isa Matrix
            @test (H + H) isa HermitianBallMatrix && (2.5 * H) isa HermitianBallMatrix
            @test !((H * H) isa HermitianBallMatrix) && !(((1 + 2im) * H) isa HermitianBallMatrix)
            @test !((H + (1 + 2im) * I) isa HermitianBallMatrix) && !((H + U) isa HermitianBallMatrix)
        end
    end
end
