using SparseArrays

@testset "Indexing fallback, validation, imag, vector products" begin
    B = BallMatrix(rand(3, 3), rand(3, 3) * 1e-9)

    @testset "getindex fallback handles scalar results" begin
        # The catch-all `getindex(::BallArray, inds...)` used to wrap its result
        # in a `BallArray` unconditionally. Index patterns mixing scalars with
        # `CartesianIndex{0}` select a single element, so that threw a
        # MethodError — and Base generates exactly such patterns internally,
        # which is why `permutedims` was broken.
        @test B[1, CartesianIndex()] isa Ball
        @test B[1, CartesianIndex(), 1] isa Ball

        P = permutedims(B)
        @test size(P) == (3, 3)
        @test all(P[j, i].c == B[i, j].c && P[j, i].r == B[i, j].r
        for i in 1:3, j in 1:3)

        # Genuine slices must still come back as ball arrays.
        @test B[1:2, 1:2] isa BallMatrix
        @test B[1, :] isa BallVector
        @test B[:, 1] isa BallVector
        @test B[:] isa BallVector
        @test B[1, 1] isa Ball
        @test B[CartesianIndex(1, 1)] isa Ball
    end

    @testset "axes are validated on construction" begin
        # Mismatched shapes used to be accepted silently, failing much later
        # with a confusing BoundsError or DimensionMismatch.
        @test_throws DimensionMismatch BallMatrix(rand(3, 3), rand(2, 2))
        @test_throws DimensionMismatch BallMatrix(rand(3, 3), rand(3, 4))
        @test_throws DimensionMismatch BallVector(rand(4), rand(3))
        @test_throws DimensionMismatch BallArray(rand(2, 2, 2), rand(2, 2, 3))

        # Matching axes are unaffected.
        @test BallMatrix(rand(3, 3), rand(3, 3)) isa BallMatrix
        @test BallVector(rand(4), rand(4)) isa BallVector
    end

    @testset "radius validity is the caller's responsibility" begin
        # Deliberately not checked on construction — the scan would cost more
        # than the arithmetic it guards — but available on demand.
        @test isvalid_enclosure(BallMatrix(rand(3, 3), rand(3, 3)))
        @test isvalid_enclosure(BallMatrix(rand(3, 3)))
        @test !isvalid_enclosure(BallMatrix(rand(3, 3), -ones(3, 3)))
        @test !isvalid_enclosure(BallMatrix(rand(3, 3), fill(NaN, 3, 3)))
        @test isvalid_enclosure(BallVector(rand(4), zeros(4)))
        @test !isvalid_enclosure(BallVector(rand(4), [-1.0, 0.0, 0.0, 0.0]))

        # Structured and sparse radii go through the same predicate.
        @test isvalid_enclosure(BallMatrix(sprand(6, 6, 0.3)))
        @test isvalid_enclosure(BallMatrix(Diagonal(rand(4))))

        # An infinite radius is a legitimate (if useless) enclosure.
        @test isvalid_enclosure(BallMatrix(rand(2, 2), fill(Inf, 2, 2)))

        good = BallMatrix(rand(3, 3), rand(3, 3))
        @test check_enclosure(good) === good
        @test_throws ArgumentError check_enclosure(BallMatrix(rand(3, 3), -ones(3, 3)))
        @test_throws ArgumentError check_enclosure(BallMatrix(rand(3, 3),
            fill(NaN, 3, 3)))
    end

    @testset "imag preserves the floating-point type" begin
        # `zeros(size(A))` returned Float64 storage whatever T was.
        for T in (Float64, Float32, BigFloat)
            J = imag(BallMatrix(T.(rand(3, 3))))
            @test eltype(J.c) == T
            @test eltype(J.r) == T
            @test all(iszero, J.c)
            @test all(iszero, J.r)
        end

        # Complex input keeps its stored radii and real part type.
        A = BallMatrix(rand(ComplexF64, 3, 3), rand(3, 3) * 1e-9)
        @test eltype(imag(A).c) == Float64
        @test imag(A).r == A.r

        # real() was already correct; guard against regressions.
        @test eltype(real(BallMatrix(BigFloat.(rand(3, 3)))).c) == BigFloat
    end

    @testset "matrix times a general vector" begin
        # Typed on Vector before, so views, ranges and sparse vectors fell
        # through to generic element-wise Ball multiplication: slow, and
        # returning Vector{Ball} rather than BallVector.
        M = BallMatrix(rand(4, 4), rand(4, 4) * 1e-9)
        vectors = [rand(4), view(rand(8), 1:4), 1.0:4.0,
            sprand(4, 0.7), vec(rand(1, 4)')]

        for v in vectors
            w = M * v
            @test w isa BallVector
            @test length(w) == 4
            # The enclosure must hold whatever the vector's storage.
            ex = BigFloat.(mid(M)) * BigFloat.(collect(v))
            @test all(abs(ex[i] - w.c[i]) <= w.r[i] for i in eachindex(ex))
        end

        # The BallVector and AbstractMatrix overloads must still resolve.
        @test M * BallVector(rand(4)) isa BallVector
        @test rand(4, 4) * BallVector(rand(4), rand(4) * 1e-9) isa BallVector
    end
end
