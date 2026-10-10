using BallArithmetic
using LinearAlgebra
using Random
using Test

@testset "Block resolvent floor" begin
    Random.seed!(20261012)
    σtrue(A, z) = minimum(svdvals(ComplexF64.(A) - z * I))
    grid(c, h, m) = [c + x + im * y for x in range(-h, h; length = m), y in range(-h, h; length = m)]

    function grcar(n)
        A = zeros(n, n)
        for i in 1:n
            A[i, i] = 1.0
            i < n && (A[i + 1, i] = -1.0)
            for j in (i + 1):min(i + 3, n)
                A[i, j] = 1.0
            end
        end
        return A
    end
    Qn = Matrix(qr(randn(8, 8)).Q)
    gallery = [
        ("normal", Qn * Diagonal([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 6.0]) * Qn'),
        ("two clusters", [1.0 5.0 0.1 0.0 0.05 0.0; 0.0 1.2 0.0 0.1 0.0 0.02;
            0.02 0.0 6.0 4.0 0.1 0.0; 0.0 0.03 0.0 6.3 0.0 0.1;
            0.01 0.0 0.02 0.0 -4.0 0.3; 0.0 0.0 0.0 0.01 0.0 -9.0]),
        ("grcar 12", grcar(12)),
        ("random", randn(7, 7)),
        ("chain", Matrix(Bidiagonal(Float64.(1:9), fill(0.3, 8), :U)))
    ]

    @testset "$name" for (name, A) in gallery
        vbd = miyajima2014a_schurnewton(BallMatrix(A))
        f = block_resolvent_floor(vbd)
        W = Matrix(vbd.basis)
        @test f.kappa ≥ cond(W) * (1 - 1e-10)
        @test length(f.centres) == length(vbd.clusters) == length(f.blocks)
        positive_far = 0
        positive_near = 0
        for z in grid(sum(diag(A)) / size(A, 1) + 0im, 8.0, 13)
            truth = σtrue(A, z)
            far = sigma_min_floor(f, z)
            near = sigma_min_floor(f, z; near = true)
            @test 0 ≤ far ≤ truth * (1 + 1e-12)
            @test far ≤ near ≤ truth * (1 + 1e-12)
            positive_far += far > 0
            positive_near += near > 0
            rb = resolvent_bound(f, z; near = true)
            @test rb ≥ (1 / truth) * (1 - 1e-12)
            @test (near > 0) == isfinite(rb)
        end
        @test positive_near ≥ positive_far
        @test positive_near > 0                      # the bound says something on this box
    end

    @testset "from verifyeigall, $name, $method" for (name, A) in gallery,
        method in (:rump2022a, :rump2022aneumann, :rump2022adiscclusters)

        r = verifyeigall(BallMatrix(A); method)
        f = block_resolvent_floor(r)
        # the frame and the transformed matrix against a 512-bit computation
        S = Complex{BigFloat}.(mid(r.similarity))
        @test f.kappa ≥ cond(ComplexF64.(S)) * (1 - 1e-10)
        if maximum(rad(r.similarity)) == 0
            M = setprecision(() -> S \ (Complex{BigFloat}.(A) * S), BigFloat, 512)
            @test all(abs.(M - mid(r.transformed)) .≤ rad(r.transformed) .* (1 + 1e-12))
        end
        @test length(f.centres) == length(r.clusters) == length(f.blocks)
        positive = 0
        for z in grid(sum(diag(A)) / size(A, 1) + 0im, 8.0, 13)
            truth = σtrue(A, z)
            far = sigma_min_floor(f, z)
            near = sigma_min_floor(f, z; near = true)
            @test 0 ≤ far ≤ near ≤ truth * (1 + 1e-12)
            positive += near > 0
        end
        @test positive > 0
    end

    @testset "from verifyeigall: no transformation, and another method" begin
        # a Jordan block has no invertible eigenvector matrix in floating point; the result then
        # carries the input and the identity, and the bound is the one with singleton blocks
        J = Matrix(Bidiagonal(fill(2.0, 6), fill(1.0, 5), :U))
        r = verifyeigall(BallMatrix(J))
        f = block_resolvent_floor(r)
        if all(iszero, mid(r.similarity) - I)
            @test f.kappa ≤ 1 + 1e-12
            @test mid(r.transformed) == J
        end
        for z in grid(2.0 + 0im, 6.0, 9)
            @test 0 ≤ sigma_min_floor(f, z) ≤ σtrue(J, z) * (1 + 1e-12)
        end
        m = verifyeigall(BallMatrix(randn(5, 5)); method = :miyajima2014a)
        @test_throws ArgumentError block_resolvent_floor(m)
    end

    @testset "one non-normal block: what the block singular values buy" begin
        # with the clustering separation 2 the Grcar matrix is kept as one block in the Schur
        # frame; inside the disc |z − c| ≤ ‖P − cI‖ only the singular value computation gives a
        # positive bound, and it is then the smallest singular value of A up to the frame
        A = grcar(12)
        vbd = miyajima2014a_schurnewton(BallMatrix(A); sep = 2.0)
        @test length(vbd.clusters) == 1
        f = block_resolvent_floor(vbd)
        @test 1 ≤ f.kappa ≤ 1 + 1e-10
        for θ in (0.3, 2.0, 4.1)
            z = f.centres[1] + 0.9 * f.nonnormality[1] * cis(θ)
            truth = σtrue(A, z)
            @test sigma_min_floor(f, z) == 0
            near = sigma_min_floor(f, z; near = true)
            @test near ≤ truth * (1 + 1e-12)
            @test near ≥ truth * (1 - 1e-6)
        end
        # the same matrix split into singletons: valid, with the conditioning in κ
        f1 = block_resolvent_floor(miyajima2014a_schurnewton(BallMatrix(A)))
        @test f1.kappa > 10
        @test sigma_min_floor(f1, 5.0 + 5im) ≤ σtrue(A, 5.0 + 5im) * (1 + 1e-12)
    end

    @testset "a normal matrix: the frame is unitary and the bound is the distance" begin
        A = gallery[1][2]
        f = block_resolvent_floor(miyajima2014a_schurnewton(BallMatrix(A)))
        @test f.kappa ≤ 1 + 1e-10
        z = 2.5 + 1.0im
        @test σtrue(A, z) * (1 - 1e-9) ≤ sigma_min_floor(f, z) ≤ σtrue(A, z) * (1 + 1e-12)
    end

    @testset "a ball of matrices" begin
        A = [1.0 0.4 0.0 0.1; 0.0 3.0 0.5 0.0; 0.1 0.0 -2.0 0.3; 0.0 0.2 0.0 6.0]
        r = 1e-6
        f = block_resolvent_floor(miyajima2014a_schurnewton(BallMatrix(A, fill(r, 4, 4))))
        for z in grid(2.0 + 0im, 6.0, 7), _ in 1:4
            Am = A + r .* rand((-1.0, 1.0), 4, 4)
            @test sigma_min_floor(f, z; near = true) ≤ σtrue(Am, z) * (1 + 1e-12)
        end
    end

    @testset "Hermitian route and BigFloat" begin
        H = [4.0 1.0 0.2; 1.0 3.0 0.1; 0.2 0.1 -1.0]
        f = block_resolvent_floor(schur_gershgorin_enclosure(BallMatrix(H); hermitian = true))
        for z in grid(2.0 + 0im, 5.0, 9)
            s = sigma_min_floor(f, z; near = true)
            @test s ≤ σtrue(H, z) * (1 + 1e-12)
        end
        @test 0 < sigma_min_floor(f, 20.0) ≤ σtrue(H, 20.0) * (1 + 1e-12)
        setprecision(BigFloat, 256) do
            A = BigFloat.([1.0 0.4 0.0; 0.0 3.0 0.5; 0.1 0.0 -2.0])
            fb = block_resolvent_floor(miyajima2014a_schurnewton(BallMatrix(A)))
            s = sigma_min_floor(fb, 10 + 2im; near = true)
            @test s isa BigFloat && 0 < s ≤ σtrue(A, 10 + 2im) * (1 + 1e-12)
        end
    end

    @testset "a pencil is refused" begin
        A = [1.0 0.4; 0.0 3.0]
        B = [2.0 0.1; 0.0 1.0]
        vbd = miyajima2014a_schurnewton(BallMatrix(A), BallMatrix(B))
        @test vbd.pencil
        @test !miyajima2014a_schurnewton(BallMatrix(A)).pencil
        @test_throws ArgumentError block_resolvent_floor(vbd)
    end
end
