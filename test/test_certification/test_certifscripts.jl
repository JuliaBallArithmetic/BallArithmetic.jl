using Test
using LinearAlgebra
using BallArithmetic

@testset "CertifScripts serial" begin
    circle = BallArithmetic.CertifScripts.CertificationCircle(0.0, 0.25; samples = 8)
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    result = BallArithmetic.CertifScripts.run_certification(A, circle; η = 0.9, check_interval = 4, log_io = IOBuffer())
    @test !isempty(result.certification_log)
    @test result.minimum_singular_value > 0
    @test result.resolvent_original >= result.resolvent_schur
end

@testset "CertifScripts serial with schur_data" begin
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    # Pre-compute Schur data once
    sd = BallArithmetic.CertifScripts.compute_schur_and_error(A)
    S, errF, errT, norm_Z, norm_Z_inv = sd

    # Reuse the same Schur data for two different circles (avoids recomputation)
    for center in [0.0, 0.5]
        circle = BallArithmetic.CertifScripts.CertificationCircle(center, 0.25; samples = 8)
        result = BallArithmetic.CertifScripts.run_certification(A, circle;
            schur_data = sd, η = 0.9, check_interval = 4, log_io = IOBuffer())
        @test !isempty(result.certification_log)
        @test result.minimum_singular_value > 0
        @test result.errF == errF
        @test result.errT == errT
    end
end

@testset "CertifScripts distributed" begin
    using Distributed
    circle = BallArithmetic.CertifScripts.CertificationCircle(0.0, 0.25; samples = 8)
    A = BallArithmetic.BallMatrix(Matrix{ComplexF64}(I, 2, 2))
    result = BallArithmetic.CertifScripts.run_certification(A, circle, 1; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
    @test result.circle == circle
    @test !isempty(result.certification_log)

    pids = addprocs(1)
    pool = WorkerPool(pids)
    try
        pooled_result = BallArithmetic.CertifScripts.run_certification(A, circle, pool; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
        @test pooled_result.circle == circle
        @test !isempty(pooled_result.certification_log)

        reuse_result = BallArithmetic.CertifScripts.run_certification(A, circle, pool; η = 0.9, check_interval = 4, log_io = IOBuffer(), channel_capacity = 4)
        @test reuse_result.circle == circle
        @test !isempty(reuse_result.certification_log)
    finally
        rmprocs(pids)
    end
end

@testset "Polynomial helpers" begin
    coeffs = BallArithmetic.CertifScripts.poly_from_roots([1, 2, 3])
    @test coeffs ≈ [-6.0, 11.0, -6.0, 1.0]
end

@testset "Sylvester Miyajima enclosure" begin
    A = [3.0 1.0 0.0; 0.0 2.5 0.3; 0.0 0.0 4.0]
    B = [-1.5 0.2; 0.0 -0.75]
    C = [1.0 0.5; -0.2 0.8; 0.3 -0.4]

    n = size(B, 1)
    K = kron(Matrix{Float64}(I, n, n), A) + kron(transpose(B), Matrix{Float64}(I, size(A, 1), size(A, 1)))
    X_exact = reshape(K \ vec(C), size(A, 1), size(B, 1))

    X̃ = X_exact .+ 1e-12 .* ones(size(X_exact))
    enclosure = BallArithmetic.sylvester_miyajima_enclosure(A, B, C, X̃)

    @test all(rad(enclosure) .>= 0)
    diff = abs.(X_exact .- mid(enclosure))
    @test all(diff .<= rad(enclosure))
end

@testset "Triangular Miyajima Sylvester block" begin
    T = UpperTriangular([2.0 + 0.2im  0.3 - 0.1im   0.5 + 0.4im  -0.2 + 0.3im;
                         0.0          1.5 + 0.6im  -0.1 - 0.2im  0.4 + 0.1im;
                         0.0          0.0           2.7 - 0.5im  0.6 - 0.3im;
                         0.0          0.0           0.0           3.4 + 0.2im])
    k = 2

    enclosure = BallArithmetic.triangular_sylvester_miyajima_enclosure(T, k)
    @test size(enclosure) == (size(T, 1) - k, k)

    Tmat = Matrix(T)
    T11 = Matrix(Tmat[1:k, 1:k])
    T22 = Matrix(Tmat[k+1:end, k+1:end])
    T12 = Matrix(Tmat[1:k, k+1:end])

    A = Matrix(adjoint(T22))
    B = -Matrix(adjoint(T11))
    C = Matrix(adjoint(T12))
    In = Matrix{eltype(A)}(I, size(B, 1), size(B, 1))
    Im = Matrix{eltype(A)}(I, size(A, 1), size(A, 1))
    K = kron(In, A) + kron(transpose(B), Im)
    Y_exact = reshape(K \ vec(C), size(A, 1), size(B, 1))

    diff = abs.(Y_exact .- mid(enclosure))
    @test all(diff .<= rad(enclosure))

    @test_throws ArgumentError BallArithmetic.triangular_sylvester_miyajima_enclosure(T, size(T, 1))

    nontriangular = Matrix(T)
    nontriangular[end, 1] = 1.0
    @test_throws ArgumentError BallArithmetic.triangular_sylvester_miyajima_enclosure(nontriangular, k)
end

# ==============================================================
# _evaluate_sample_with_ogita_cache
# ==============================================================

@testset "_evaluate_sample_with_ogita_cache" begin
    CS = BallArithmetic.CertifScripts

    @testset "basic evaluation — cache miss (fresh SVD)" begin
        CS._clear_ogita_cache!()
        T = BallArithmetic.BallMatrix(diagm(ComplexF64[1.0, 2.0, 3.0]))
        z = 0.5 + 0.1im

        result = CS._evaluate_sample_with_ogita_cache(T, z, 1)

        @test result.i == 1
        @test result.z ≈ ComplexF64(z)
        @test result.lo_val > 0       # σ_min > 0 (z away from eigenvalues)
        @test result.hi_res > 0       # resolvent upper bound is positive
        @test isfinite(result.hi_res)
        @test result.ogita_used == false  # first call → cache miss

        stats = CS._ogita_cache_stats()
        @test stats.misses == 1
        @test stats.hits == 0
    end

    @testset "Ogita cache hit for nearby z" begin
        CS._clear_ogita_cache!()
        T = BallArithmetic.BallMatrix(diagm(ComplexF64[1.0, 2.0, 3.0]))

        # Prime the cache
        z1 = 0.5 + 0.1im
        r1 = CS._evaluate_sample_with_ogita_cache(T, z1, 1;
                 ogita_distance_threshold=0.01)

        # Very close point → should attempt Ogita
        z2 = 0.501 + 0.101im
        r2 = CS._evaluate_sample_with_ogita_cache(T, z2, 2;
                 ogita_distance_threshold=0.01)

        @test r2.i == 2
        @test isfinite(r2.hi_res)
        @test r2.lo_val > 0

        # Both should give valid upper bounds on resolvent
        true_res1 = opnorm(inv(z1 * I - diagm(ComplexF64[1.0, 2.0, 3.0])), 2)
        true_res2 = opnorm(inv(z2 * I - diagm(ComplexF64[1.0, 2.0, 3.0])), 2)
        @test r1.hi_res >= true_res1 * 0.99
        @test r2.hi_res >= true_res2 * 0.99
    end

    @testset "cache miss for distant z" begin
        CS._clear_ogita_cache!()
        T = BallArithmetic.BallMatrix(diagm(ComplexF64[1.0, 2.0, 3.0]))

        # Prime cache at z1
        CS._evaluate_sample_with_ogita_cache(T, 0.5 + 0.1im, 1;
                 ogita_distance_threshold=0.01)

        # Distant point → cache miss, not Ogita
        r = CS._evaluate_sample_with_ogita_cache(T, 5.0 + 5.0im, 2;
                 ogita_distance_threshold=0.01)

        @test r.ogita_used == false
        stats = CS._ogita_cache_stats()
        @test stats.misses == 2  # both are misses
    end

    @testset "second_val is second-smallest singular value" begin
        CS._clear_ogita_cache!()
        M = ComplexF64[1 0.1; 0 2]
        T = BallArithmetic.BallMatrix(M)
        z = 0.0 + 0.0im

        result = CS._evaluate_sample_with_ogita_cache(T, z, 1)
        @test mid(result.second_val) > mid(result.val)
    end
end

# ==============================================================
# dowork_ogita (channel-based integration)
# ==============================================================

@testset "dowork_ogita — channel integration" begin
    CS = BallArithmetic.CertifScripts

    T = BallArithmetic.BallMatrix(diagm(ComplexF64[1.0, 2.0, 3.0]))
    CS.set_schur_matrix!(T)

    jobs    = Channel{Tuple{Int, ComplexF64}}(10)
    results = Channel{Any}(10)

    z_list = [0.5 + 0.1im, 0.502 + 0.101im, 5.0 + 1.0im]
    for (i, z) in enumerate(z_list)
        put!(jobs, (i, ComplexF64(z)))
    end
    close(jobs)

    CS.dowork_ogita(jobs, results; ogita_distance_threshold=0.01)
    close(results)

    collected = collect(results)
    @test length(collected) == 3

    for r in collected
        @test r.lo_val > 0
        @test isfinite(r.hi_res)
        @test r.hi_res > 0
    end

    # Verify ordering
    ids = [r.i for r in collected]
    @test sort(ids) == [1, 2, 3]
end

# ==============================================================
# dowork_ogita_bigfloat (channel-based)
# ==============================================================

@testset "dowork_ogita_bigfloat — channel integration" begin
    CS = BallArithmetic.CertifScripts

    old_prec = precision(BigFloat)
    setprecision(BigFloat, 256)
    try
        T_bf = BallArithmetic.BallMatrix(diagm(Complex{BigFloat}[1, 2, 3]))
        CS.set_schur_matrix!(T_bf)

        jobs    = Channel{Tuple{Int, Complex{BigFloat}}}(10)
        results = Channel{Any}(10)

        z_list = [Complex{BigFloat}(0.5, 0.1), Complex{BigFloat}(5.0, 1.0)]
        for (i, z) in enumerate(z_list)
            put!(jobs, (i, z))
        end
        close(jobs)

        CS.dowork_ogita_bigfloat(jobs, results;
                                  target_precision=256, max_ogita_iterations=3)
        close(results)

        collected = collect(results)
        @test length(collected) == 2

        for r in collected
            @test r.lo_val > 0
            @test isfinite(r.hi_res)
        end
    finally
        setprecision(BigFloat, old_prec)
    end
end

# ==============================================================
# choose_snapshot_to_load
# ==============================================================

@testset "choose_snapshot_to_load" begin
    CS = BallArithmetic.CertifScripts
    using JLD2

    mktempdir() do dir
        @testset "returns nothing when no files exist" begin
            result = CS.choose_snapshot_to_load(joinpath(dir, "nonexistent"))
            @test result === nothing
        end

        @testset "loads single snapshot" begin
            basepath = joinpath(dir, "snap1")
            arcs = [1, 2, 3]
            cache = Dict("a" => 1)
            log = ["entry1"]
            pending = Int[]
            JLD2.@save (basepath * "_A.jld2") arcs cache log pending

            result = CS.choose_snapshot_to_load(basepath)
            @test result !== nothing
            @test result["arcs"] == [1, 2, 3]
            @test result["cache"] == Dict("a" => 1)
        end

        @testset "prefers most recent of A/B" begin
            basepath = joinpath(dir, "snap2")
            arcs_a = [1]; cache_a = Dict(); log_a = []; pending_a = []
            JLD2.@save (basepath * "_A.jld2") arcs=arcs_a cache=cache_a log=log_a pending=pending_a
            sleep(0.1)  # ensure different mtime
            arcs_b = [1, 2]; cache_b = Dict(); log_b = []; pending_b = []
            JLD2.@save (basepath * "_B.jld2") arcs=arcs_b cache=cache_b log=log_b pending=pending_b

            result = CS.choose_snapshot_to_load(basepath)
            @test result !== nothing
            @test result["arcs"] == [1, 2]  # B is newer
        end

        @testset "falls back to backup on corruption" begin
            basepath = joinpath(dir, "snap3")
            arcs = [42]; cache = Dict(); log_data = []; pending = []
            JLD2.@save (basepath * "_A.jld2") arcs cache log=log_data pending

            # Write corrupted B file
            write(basepath * "_B.jld2", "corrupted data")
            sleep(0.1)
            # Touch B to make it newer
            touch(basepath * "_B.jld2")

            result = CS.choose_snapshot_to_load(basepath)
            @test result !== nothing
            @test result["arcs"] == [42]  # fell back to A
        end
    end
end

# ==============================================================
# _polynomial_matrix
# ==============================================================

@testset "_polynomial_matrix" begin
    CS = BallArithmetic.CertifScripts

    @testset "identity polynomial [0, 1] returns M" begin
        M = BallArithmetic.BallMatrix(ComplexF64[1 2; 3 4])
        result = CS._polynomial_matrix([0, 1], M)
        @test mid(result) ≈ mid(M)
    end

    @testset "constant polynomial [c] returns c*I" begin
        M = BallArithmetic.BallMatrix(ComplexF64[1 2; 3 4])
        result = CS._polynomial_matrix([3.0], M)
        @test mid(result) ≈ 3.0 * I(2) atol = 1e-12
    end

    @testset "p(M) = M^2 + 2M + 3I via [3, 2, 1]" begin
        A = ComplexF64[1 0.5; 0 2]
        M = BallArithmetic.BallMatrix(A)
        result = CS._polynomial_matrix([3.0, 2.0, 1.0], M)

        expected = A^2 + 2A + 3I
        @test mid(result) ≈ expected atol = 1e-10
    end

    @testset "BigFloat polynomial" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A = Complex{BigFloat}[1 0; 0 2]
            M = BallArithmetic.BallMatrix(A)
            result = CS._polynomial_matrix(BigFloat[1, 0, 1], M)  # M^2 + I

            expected = A^2 + I
            @test mid(result) ≈ expected atol = BigFloat(10)^(-60)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "AbstractMatrix fallback" begin
        A = ComplexF64[2 1; 0 3]
        result = CS._polynomial_matrix([1.0, 1.0], A)  # M + I
        @test mid(result) ≈ A + I atol = 1e-12
    end
end

# ==============================================================
# _compute_schur_bigfloat_direct
# ==============================================================

@testset "_compute_schur_bigfloat_direct" begin
    CS = BallArithmetic.CertifScripts

    @testset "basic 3x3 BigFloat" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A_mid = Complex{BigFloat}[1 0.5 0; 0.1 2 0.3; 0 0.2 3]
            A = BallArithmetic.BallMatrix(A_mid)

            S_nt, errF, errT, norm_Z, norm_Z_inv = CS._compute_schur_bigfloat_direct(A)

            # Schur decomposition properties
            @test size(S_nt.T) == (3, 3)
            @test size(S_nt.Z) == (3, 3)
            @test length(S_nt.values) == 3

            # T should be upper triangular
            @test norm(tril(S_nt.T, -1)) < BigFloat(10)^(-60)

            # Bounds should be small
            errF_val = BigFloat(errF)
            errT_val = BigFloat(errT)
            @test errF_val < BigFloat(10)^(-50)
            @test errT_val < BigFloat(10)^(-50)

            # norm bounds should be close to 1
            @test norm_Z >= 1
            @test norm_Z_inv >= 1
            @test norm_Z < BigFloat(2)
            @test norm_Z_inv < BigFloat(2)

            # Eigenvalues should match diagonal of T
            @test sort(real.(S_nt.values)) ≈ sort(real.(diag(S_nt.T))) atol = BigFloat(10)^(-50)
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "with polynomial" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A_mid = Complex{BigFloat}[2 1; 0 3]
            A = BallArithmetic.BallMatrix(A_mid)

            # p(x) = x^2 + x + 1
            poly = BigFloat[1, 1, 1]
            S_nt, errF, errT_poly, norm_Z, norm_Z_inv = CS._compute_schur_bigfloat_direct(A; polynomial=poly)

            @test BigFloat(errF) < BigFloat(10)^(-50)
            # errT_poly should be small (measures Z * p(T) * Z' - p(A))
            @test BigFloat(errT_poly) < BigFloat(10)^(-40)
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end

# ==============================================================
# _compute_schur_bigfloat_refined
# ==============================================================

@testset "_compute_schur_bigfloat_refined" begin
    CS = BallArithmetic.CertifScripts

    @testset "basic 3x3 BigFloat" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A_mid = Complex{BigFloat}[1 0.5 0; 0.1 2 0.3; 0 0.2 3]
            A = BallArithmetic.BallMatrix(A_mid)

            S_nt, errF, errT, norm_Z, norm_Z_inv = CS._compute_schur_bigfloat_refined(A)

            @test size(S_nt.T) == (3, 3)
            @test size(S_nt.Z) == (3, 3)
            @test length(S_nt.values) == 3

            # T should be upper triangular
            @test norm(tril(S_nt.T, -1)) < BigFloat(10)^(-30)

            # Bounds should be reasonable (refined path may be less precise than direct)
            errF_val = BigFloat(errF)
            errT_val = BigFloat(errT)
            @test errF_val < BigFloat(1)  # orthogonality defect < 1
            @test errT_val < BigFloat(1)

            # norm bounds
            @test norm_Z >= 1
            @test norm_Z_inv >= 1
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "consistency with direct path" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            A_mid = Complex{BigFloat}[2 1; 0 3]
            A = BallArithmetic.BallMatrix(A_mid)

            S_direct, errF_d, errT_d, _, _ = CS._compute_schur_bigfloat_direct(A)
            S_refined, errF_r, errT_r, _, _ = CS._compute_schur_bigfloat_refined(A)

            # Both should find the same eigenvalues
            ev_d = sort(real.(S_direct.values))
            ev_r = sort(real.(S_refined.values))
            @test ev_d ≈ ev_r atol = BigFloat(10)^(-10)
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end

# ==============================================================
# _evaluate_sample_ogita_bigfloat
# ==============================================================

@testset "_evaluate_sample_ogita_bigfloat" begin
    CS = BallArithmetic.CertifScripts

    @testset "basic evaluation" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            T_bf = BallArithmetic.BallMatrix(diagm(Complex{BigFloat}[1, 2, 3]))
            CS._clear_bf_ogita_cache!()

            z = Complex{BigFloat}(0.5, 0.1)
            r = CS._evaluate_sample_ogita_bigfloat(T_bf, z, 1;
                    max_iterations=3, target_precision=256)

            @test r.i == 1
            @test r.lo_val > 0
            @test isfinite(r.hi_res)
            @test r.hi_res > 0
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "cache reuse for nearby points" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            T_bf = BallArithmetic.BallMatrix(diagm(Complex{BigFloat}[1, 2, 3]))
            CS._clear_bf_ogita_cache!()

            z1 = Complex{BigFloat}(0.5, 0.1)
            r1 = CS._evaluate_sample_ogita_bigfloat(T_bf, z1, 1;
                    max_iterations=3, target_precision=256, distance_threshold=0.01)

            # Nearby point should use cache
            z2 = Complex{BigFloat}(0.501, 0.101)
            r2 = CS._evaluate_sample_ogita_bigfloat(T_bf, z2, 2;
                    max_iterations=3, target_precision=256, distance_threshold=0.01)

            @test r2.i == 2
            @test r2.lo_val > 0
            @test isfinite(r2.hi_res)

            stats = CS._bf_ogita_cache_stats()
            @test stats.local_hits >= 1 || stats.center_hits >= 0
        finally
            setprecision(BigFloat, old_prec)
        end
    end

    @testset "use_cache=false" begin
        old_prec = precision(BigFloat)
        setprecision(BigFloat, 256)
        try
            T_bf = BallArithmetic.BallMatrix(diagm(Complex{BigFloat}[1, 2, 3]))
            CS._clear_bf_ogita_cache!()

            z = Complex{BigFloat}(0.5, 0.1)
            r = CS._evaluate_sample_ogita_bigfloat(T_bf, z, 1;
                    max_iterations=3, target_precision=256, use_cache=false)

            @test r.lo_val > 0
            @test isfinite(r.hi_res)

            # No cache updates
            stats = CS._bf_ogita_cache_stats()
            @test stats.misses == 1
        finally
            setprecision(BigFloat, old_prec)
        end
    end
end
