# Prototype: fully on-GPU rigorous SVD.
#   seed factorization  -> cuSOLVER svd(CuMatrix)            (GPU, approximate, OK: it's only a seed)
#   certification matmuls-> GPU MMul4 (INT8-Ozaki, rigorous) (GPU)
#   rigorous O(n^2) bounds (opnorm, Neumann) -> CPU          (setrounding/directed rounding only works on CPU)
#
# The big O(n^3) work stays on the card; only the small O(n^2) product matrices
# are copied back for the rigorous norm bounds.  Compared against the stock CPU
# `rigorous_svd(A; apply_vbd=false)` for time AND enclosure agreement.

using BallArithmetic, CUDA, LinearAlgebra, Random, Printf
using BallArithmetic: mid, rad, MiyajimaM1, upper_bound_L2_opnorm,
                      _compute_svd_bounds, _diagonal_ball_matrix
import BallArithmetic

to_cpu(B) = BallMatrix(Array(mid(B)), Array(rad(B)))
gpu_z(c::CuArray) = BallMatrix(c, CUDA.zeros(Float64, size(c)...))
gpu_z(c) = BallMatrix(CuArray(c), CUDA.zeros(Float64, size(c)...))

# fully-on-GPU certified SVD (MiyajimaM1, no VBD).  Returns (singular_values, normF, normG, timings)
function rigorous_svd_gpu(A::BallMatrix{Float64,Float64}; method=MiyajimaM1(),
                          alg=CUDA.CUSOLVER.QRAlgorithm(), seed_on=:gpu)
    T = Float64
    local sv, S_cpu, Ug, Vg, Vtg
    t_seed = if seed_on == :cpu
        @elapsed begin
            sv = svd(A.c)                              # CPU LAPACK seed, then upload factors
            S_cpu = sv.S
            Ug = gpu_z(CuArray(Matrix(sv.U)))
            Vg = gpu_z(CuArray(Matrix(sv.V)))
            Vtg = gpu_z(CuArray(Matrix(sv.Vt)))
        end
    else
        CUDA.@elapsed begin
            Ac = CuArray(mid(A))
            sv = svd(Ac; alg=alg)                      # cuSOLVER seed (stays on device)
            S_cpu = Array(sv.S)
            Ug = gpu_z(sv.U)
            Vg = gpu_z(sv.V)
            Vtg = gpu_z(sv.Vt isa CuArray ? sv.Vt : CuArray(sv.Vt))
            CUDA.synchronize()
        end
    end
    Σg  = gpu_z(CuArray(Matrix(Diagonal(S_cpu))))

    # GPU MMul4 products (rigorous midpoint+radius); adjoint U' now dispatches to GPU
    local P1, P2, P3
    t_prod = CUDA.@elapsed begin
        P1 = Ug * Σg * Vtg          # ≈ A
        P2 = Vtg * Vg               # ≈ I (right orthogonality)
        P3 = Ug' * Ug               # ≈ I (left orthogonality)
        CUDA.synchronize()
    end

    # rigorous O(n^2) finish on CPU (directed rounding)
    t_cpu = @elapsed begin
        A_cpu = BallMatrix(Array(mid(A)), Array(rad(A)))
        E = to_cpu(P1) - A_cpu
        F = to_cpu(P2) - I
        G = to_cpu(P3) - I
        normE = upper_bound_L2_opnorm(E)
        normF = upper_bound_L2_opnorm(F)
        normG = upper_bound_L2_opnorm(G)
        if normF >= 1 || normG >= 1
            error("verification gate failed: normF=$normF normG=$normG")
        end
        down, up = _compute_svd_bounds(method, S_cpu, normE, normF, normG, T)
        mids = (down .+ up) ./ 2
        radii = setrounding(T, RoundUp) do
            [max(up[i]-mids[i], mids[i]-down[i]) for i in eachindex(mids)]
        end
        global _sv_gpu = [Ball(mids[i], radii[i]) for i in eachindex(mids)]
        global _nF = normF; global _nG = normG
    end
    return _sv_gpu, _nF, _nG, (seed=t_seed, prod=t_prod, cpu=t_cpu)
end

ct(f; nr=3) = (f(); minimum(@elapsed(f()) for _ in 1:nr))
gt(f; nr=3) = (f(); CUDA.synchronize(); minimum((CUDA.@elapsed f()) for _ in 1:nr))

function run(n; seed=1)
    rng = MersenneTwister(seed)
    A = BallMatrix(randn(rng, n, n) ./ sqrt(n), fill(1e-15, n, n))

    # CPU baseline
    t_cpu_total = ct(() -> rigorous_svd(A; apply_vbd=false))
    t_cpu_fact  = ct(() -> svd(A.c))
    res_cpu = rigorous_svd(A; apply_vbd=false)

    # seed factorization: CPU LAPACK vs cuSOLVER QR vs cuSOLVER Jacobi
    Ac = CuArray(A.c)
    t_qr  = gt(() -> svd(Ac; alg=CUDA.CUSOLVER.QRAlgorithm()))
    t_jac = gt(() -> svd(Ac; alg=CUDA.CUSOLVER.JacobiAlgorithm()))
    @printf("  seed-only: CPU LAPACK %.3fs | cuSOLVER QR %.3fs | cuSOLVER Jacobi %.3fs\n",
        t_cpu_fact, t_qr, t_jac)

    # full-GPU (cuSOLVER seed) and hybrid (CPU LAPACK seed + GPU products)
    sv_f, nFf, nGf, _ = rigorous_svd_gpu(A; seed_on=:gpu)
    sv_h, nFh, nGh, _ = rigorous_svd_gpu(A; seed_on=:cpu)
    tf = minimum(begin _,_,_,t = rigorous_svd_gpu(A; seed_on=:gpu); t.seed+t.prod+t.cpu end for _ in 1:3)
    th = minimum(begin _,_,_,t = rigorous_svd_gpu(A; seed_on=:cpu); t.seed+t.prod+t.cpu end for _ in 1:3)
    _,_,_,tgf = rigorous_svd_gpu(A; seed_on=:gpu)
    _,_,_,tgh = rigorous_svd_gpu(A; seed_on=:cpu)

    overlap = all(abs(mid(res_cpu.singular_values[i]) - mid(sv_f[i])) <=
                  rad(res_cpu.singular_values[i]) + rad(sv_f[i]) + 1e-12 for i in 1:n)
    maxrad_cpu = maximum(rad.(res_cpu.singular_values))

    @printf("n=%-4d | CPU-all %.3fs | FULL-GPU %.3fs (seed %.3f prod %.3f cpu %.3f) %.1fx | HYBRID %.3fs (seed %.3f prod %.3f cpu %.3f) %.1fx\n",
        n, t_cpu_total,
        tf, tgf.seed, tgf.prod, tgf.cpu, t_cpu_total/tf,
        th, tgh.seed, tgh.prod, tgh.cpu, t_cpu_total/th)
    @printf("        overlap=%s | σ-radius CPU=%.1e full-gpu=%.1e hybrid=%.1e | gate full(F=%.1e) hybrid(F=%.1e)\n",
        overlap, maxrad_cpu, maximum(rad.(sv_f)), maximum(rad.(sv_h)), nFf, nFh)
    flush(stdout)
end

println("Device: ", CUDA.name(CUDA.device()))
@printf("CUDAExt loaded: %s\n\n", !isnothing(Base.get_extension(BallArithmetic, :CUDAExt)))
for n in (512, 1024, 2048); run(n); end
