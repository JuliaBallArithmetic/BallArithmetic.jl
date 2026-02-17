using BallArithmetic
using LinearAlgebra
using Test

@testset "triangular_eigenvectors" begin

    @testset "Lower triangular Float64" begin
        L = [2.0 0.0 0.0;
             1.0 5.0 0.0;
             0.5 0.3 8.0]
        V, W, λ = triangular_eigenvectors(L)

        # Eigenvalues are the diagonal
        @test λ ≈ [2.0, 5.0, 8.0]

        # V * W ≈ I
        @test V * W ≈ I atol = 1e-12

        # L * V ≈ V * Diagonal(λ)
        @test L * V ≈ V * Diagonal(λ) atol = 1e-12

        # V is unit lower triangular
        @test istril(V)
        for i in 1:3
            @test V[i, i] ≈ 1.0
        end
    end

    @testset "Upper triangular Float64" begin
        U = [1.0 2.0 0.5;
             0.0 3.0 1.0;
             0.0 0.0 7.0]
        V, W, λ = triangular_eigenvectors(U)

        @test λ ≈ [1.0, 3.0, 7.0]
        @test V * W ≈ I atol = 1e-12
        @test U * V ≈ V * Diagonal(λ) atol = 1e-12

        # V is unit upper triangular
        @test istriu(V)
        for i in 1:3
            @test V[i, i] ≈ 1.0
        end
    end

    @testset "Complex triangular" begin
        U = [1.0+1.0im 2.0+0.5im 0.0+0.0im;
             0.0+0.0im 3.0-1.0im 1.0+0.0im;
             0.0+0.0im 0.0+0.0im 5.0+2.0im]
        V, W, λ = triangular_eigenvectors(U)

        @test λ ≈ diag(U)
        @test V * W ≈ I atol = 1e-12
        @test U * V ≈ V * Diagonal(λ) atol = 1e-12
    end

    @testset "BigFloat 256-bit" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            L = BigFloat[3 0 0; 1 7 0; 2 4 11]
            V, W, λ = triangular_eigenvectors(L)

            @test λ ≈ BigFloat[3, 7, 11]
            @test V * W ≈ I atol = BigFloat(10)^(-70)
            @test L * V ≈ V * Diagonal(λ) atol = BigFloat(10)^(-70)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "Fallback on repeated eigenvalues" begin
        # Matrix with repeated eigenvalue
        L = [2.0 0.0 0.0;
             1.0 2.0 0.0;
             0.5 0.3 5.0]
        # Should not error — falls back to eigen()
        V, W, λ = triangular_eigenvectors(L)

        # V * W ≈ I (within eigen() precision; defective matrices have lower accuracy)
        @test V * W ≈ I atol = 1e-2
    end

    @testset "1×1 matrix" begin
        M = fill(3.0, 1, 1)
        V, W, λ = triangular_eigenvectors(M)
        @test λ ≈ [3.0]
        @test V ≈ ones(1, 1)
        @test W ≈ ones(1, 1)
    end

    @testset "Consistency with eigen() — lower triangular" begin
        n = 5
        L = tril(randn(n, n)) .+ Diagonal(collect(1.0:n) .* 3)  # distinct diagonal
        V, W, λ = triangular_eigenvectors(L)
        F = eigen(L)

        # Eigenvalues should match (up to ordering)
        @test sort(real.(λ)) ≈ sort(real.(F.values)) atol = 1e-10

        # Key property: L * V = V * Diagonal(λ)
        @test L * V ≈ V * Diagonal(λ) atol = 1e-10
    end

    @testset "Consistency with eigen() — upper triangular" begin
        n = 5
        U = triu(randn(n, n)) .+ Diagonal(collect(1.0:n) .* 3)
        V, W, λ = triangular_eigenvectors(U)
        F = eigen(U)

        @test sort(real.(λ)) ≈ sort(real.(F.values)) atol = 1e-10
        @test U * V ≈ V * Diagonal(λ) atol = 1e-10
    end

    @testset "Sylvester regression — enclosures unchanged" begin
        # Use an upper-triangular matrix from Schur form and verify
        # that the Sylvester enclosure still works after the fast-path integration
        T_mid = [1.0 0.5 0.2; 0.0 3.0 0.7; 0.0 0.0 5.0]
        k = 1

        Y = triangular_sylvester_miyajima_enclosure(T_mid, k)
        @test all(isfinite, mid(Y))
        @test all(isfinite, rad(Y))

        # Verify the Sylvester equation: T22^H Y - Y T11^H = T12^H
        # i.e. T11 (-Y^H) - (-Y^H) T22 = T12
        T11 = T_mid[1:k, 1:k]
        T22 = T_mid[(k+1):3, (k+1):3]
        T12 = T_mid[1:k, (k+1):3]
        Y_coupling = -adjoint(mid(Y))
        residual = T11 * Y_coupling - Y_coupling * T22 - T12
        @test norm(residual) < 1e-10
    end
end

@testset "Direct triangular Sylvester fallback (Bug 1)" begin

    @testset "Ill-conditioned triangular — direct fallback succeeds" begin
        # Construct a triangular matrix with eigenvalues spanning many orders
        # of magnitude (mimicking the 513×513 GKW case).
        # The Miyajima eigenvector approach fails here because V is too ill-conditioned.
        n = 50
        eigenvalues = [10.0^(-i/2) for i in 0:(n-1)]  # 1.0 down to ~10^(-24)
        T_diag = Diagonal(eigenvalues)
        # Add off-diagonal entries to make it non-trivial
        T_mat = Matrix(T_diag)
        for i in 1:n
            for j in (i+1):n
                T_mat[i, j] = 0.01 * randn() / (j - i)
            end
        end
        @assert istriu(T_mat)

        k = 1
        # This should NOT throw — the direct fallback handles it
        Y = triangular_sylvester_miyajima_enclosure(T_mat, k)
        @test all(isfinite, mid(Y))
        @test all(isfinite, rad(Y))

        # Verify: T22^H Y - Y T11^H ≈ T12^H
        T11 = T_mat[1:k, 1:k]
        T22 = T_mat[(k+1):n, (k+1):n]
        T12 = T_mat[1:k, (k+1):n]
        Y_coupling = -adjoint(mid(Y))
        residual = T11 * Y_coupling - Y_coupling * T22 - T12
        @test norm(residual) < 1e-6
    end

    @testset "Direct solve matches Miyajima for well-conditioned case" begin
        T_mid = [1.0 0.5 0.2; 0.0 3.0 0.7; 0.0 0.0 5.0]
        k = 1

        # Both paths should give the same midpoint
        Y = triangular_sylvester_miyajima_enclosure(T_mid, k)

        # Direct solve via internal function
        A = adjoint(T_mid[(k+1):3, (k+1):3])
        B = -adjoint(T_mid[1:k, 1:k])
        C = adjoint(T_mid[1:k, (k+1):3])
        Y_direct = BallArithmetic._sylvester_triangular_direct_ball(
            Matrix(A), Matrix(B), Matrix(C))

        # Midpoints should agree closely
        @test mid(Y) ≈ mid(Y_direct) atol = 1e-12
    end

    @testset "Direct solve with k > 1" begin
        T_mid = [1.0 0.5 0.2 0.1;
                 0.0 3.0 0.7 0.3;
                 0.0 0.0 7.0 0.4;
                 0.0 0.0 0.0 10.0]
        k = 2

        Y = triangular_sylvester_miyajima_enclosure(T_mid, k)
        @test all(isfinite, mid(Y))
        @test all(isfinite, rad(Y))

        # Verify Sylvester equation
        T11 = T_mid[1:k, 1:k]
        T22 = T_mid[(k+1):4, (k+1):4]
        T12 = T_mid[1:k, (k+1):4]
        Y_coupling = -adjoint(mid(Y))
        residual = T11 * Y_coupling - Y_coupling * T22 - T12
        @test norm(residual) < 1e-10
    end

    @testset "Column-by-column solve replaces Kronecker" begin
        # Verify that the column-by-column approach gives correct results
        # by checking the Sylvester equation directly
        A = [2.0 0.0 0.0; 1.0 5.0 0.0; 0.5 0.3 8.0]  # lower triangular
        B = [-1.0 0.0; -0.5 -3.0]  # lower triangular
        C = randn(3, 2)

        X = BallArithmetic._sylvester_triangular_columns(A, B, C)
        residual = A * X + X * B - C
        @test norm(residual) < 1e-12
    end
end
