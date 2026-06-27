# Rigorous Collatz ‖A‖₂ upper bound on the GPU.
#
# collatz bound = sqrt(ρ(|A|ᵀ|A|)), via Collatz–Wielandt: for B = |A|ᵀ|A| ≥ 0 and
# any x > 0,  ρ(B) ≤ maxᵢ (Bx)ᵢ/xᵢ.  We iterate x ← Bx a few times to get a good x,
# then bound the final ratio.  Rigor without directed rounding: B,x ≥ 0, so a
# round-to-nearest cuBLAS GEMV inflated by (1+(n+2)u) is a valid UPPER bound on Bx,
# and the denominator xᵢ is the exact positive vector we chose -> ratio rounds up.

using BallArithmetic, CUDA, LinearAlgebra, Random, Printf
using BallArithmetic: mid, rad, collatz_upper_bound_L2_opnorm

const u = eps(Float64)

# entrywise upper bound on |A| over the ball: |mid|+rad, inflated to cover RN rounding
function absupper(A::BallMatrix)
    M = abs.(mid(A)) .+ rad(A)
    return M .* (1 + 2u)
end

# rigorous GPU Collatz upper bound on ‖A‖₂; absU is a dense nonneg CuArray ≥ |A|
function collatz_gpu(absU::CuArray{Float64}; iterates=5)
    m, k = size(absU)
    infl(n) = 1 + (n + 2) * u                 # per-GEMV upper-bound inflation
    x = CUDA.ones(Float64, k)
    local xprev
    for _ in 1:iterates
        xprev = x
        y = (absU * x) .* infl(k)             # ≥ |A| x        (RN GEMV + inflate)
        x = (absU' * y) .* infl(m)            # ≥ |A|ᵀ|A| x
    end
    # one more matvec as a rigorous upper bound on B*xprev
    y  = (absU * xprev) .* infl(k)
    Bx = (absU' * y)    .* infl(m)            # ≥ B*xprev (elementwise)
    ratio = Array(Bx ./ xprev)               # xprev exact & >0
    lam = maximum(ratio) * (1 + 2u)
    return sqrt(lam) * (1 + u)
end

gt(f; nr=4) = (f(); CUDA.synchronize(); minimum((CUDA.@elapsed f()) for _ in 1:nr))
ct(f; nr=4) = (f(); minimum(@elapsed(f()) for _ in 1:nr))

function run(n; seed=1)
    rng = MersenneTwister(seed)
    A = BallMatrix(randn(rng, n, n) ./ sqrt(n), fill(1e-13, n, n))

    cpu_bound = collatz_upper_bound_L2_opnorm(A)
    absU_d = CuArray(absupper(A))
    gpu_bound = collatz_gpu(absU_d)

    # rigor reference: true ‖A‖₂ (largest singular value of a sampled matrix in ball)
    truenorm = 0.0
    for _ in 1:50
        S = mid(A) .+ (2 .* rand(rng, n, n) .- 1) .* rad(A)
        truenorm = max(truenorm, opnorm(S, 2))
    end

    tg = gt(() -> collatz_gpu(absU_d))
    tc = ct(() -> collatz_upper_bound_L2_opnorm(A))
    @printf("n=%-4d | CPU collatz=%.6f (%.4fs) | GPU collatz=%.6f (%.4fs) | true≈%.6f | both≥true=%s | speedup %.1fx\n",
        n, cpu_bound, tc, gpu_bound, tg, truenorm,
        (cpu_bound ≥ truenorm && gpu_bound ≥ truenorm), tc/tg)
    flush(stdout)
end

println("Device: ", CUDA.name(CUDA.device()), "\n")
for n in (512, 1024, 2048, 4096); run(n); end
