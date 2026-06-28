@testset "GPU rigorous SVD (CUDAExt)" begin
    # Gated on a functional GPU: the CI runners have none, so this skips cleanly.
    cuda_ok = false
    try
        @eval using CUDA
        cuda_ok = CUDA.functional()
    catch
        cuda_ok = false
    end

    if !cuda_ok
        @info "CUDA not functional — skipping GPU rigorous_svd tests"
    else
        using LinearAlgebra: opnorm, I
        using BallArithmetic: mid, rad

        @test !isnothing(Base.get_extension(BallArithmetic, :CUDAExt))

        for n in (64, 128, 257)        # include a non-power-of-two, non-square-friendly size
            A = BallMatrix(randn(n, n) ./ sqrt(n), fill(1e-14, n, n))

            cpu = rigorous_svd(A; apply_vbd = false)
            for seed in (:cpu, :gpu)
                gpu = rigorous_svd_gpu(A; apply_vbd = false, seed_on = seed)

                @test length(gpu) == n
                # Gate passed (finite radii) and enclosures are sane.
                @test all(isfinite, rad.(gpu.singular_values))
                @test gpu.right_orthogonality_defect < 1
                @test gpu.left_orthogonality_defect < 1

                # GPU and CPU singular-value enclosures must overlap.
                for i in 1:n
                    mc, rc = mid(cpu[i]), rad(cpu[i])
                    mg, rg = mid(gpu[i]), rad(gpu[i])
                    @test abs(mc - mg) <= rc + rg + 1e-10
                end

                # Each enclosure must contain the true singular value of the midpoint.
                σ_true = svdvals(mid(A))
                for i in 1:n
                    @test mid(gpu[i]) - rad(gpu[i]) - 1e-9 <= σ_true[i] <=
                          mid(gpu[i]) + rad(gpu[i]) + 1e-9
                end

                # Residual / defect bounds are real, finite upper bounds.
                @test gpu.residual_norm >= 0 && isfinite(gpu.residual_norm)
            end
        end

        # VBD path also returns a result on the GPU path.
        A = BallMatrix(randn(96, 96) ./ sqrt(96), fill(1e-14, 96, 96))
        gpu_vbd = rigorous_svd_gpu(A; apply_vbd = true, seed_on = :cpu)
        @test !isnothing(gpu_vbd.block_diagonalisation)
    end
end
