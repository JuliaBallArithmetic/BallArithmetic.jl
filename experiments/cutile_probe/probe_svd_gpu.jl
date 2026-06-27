# Is RigorousSVDResult faster with the GPU (CUDA INT8) MMul4 dispatch?
#
# The rigorous SVD = LAPACK factorization (CPU) + a handful of BallMatrix products
# to certify it (residual U·Σ·Vt-A, orthogonality Vt·V-I and U'·U-I, residual
# refinement U·ΔΣ·Vt).  Only those products are O(n^3); the opnorm bounds are O(n^2).
# We measure:
#   (1) total CPU rigorous_svd time and how much of it is the certification products,
#   (2) the same products with GPU-resident BallMatrices (CUDAExt MMul4),
#   (3) the adjoint gotcha: U' stores an Adjoint{...,CuArray}, NOT a CuArray, so
#       U'·U does NOT hit the GPU dispatch unless we materialize the adjoint.

using BallArithmetic, CUDA, LinearAlgebra, Random, Printf
using BallArithmetic: mid, rad

const u64 = eps(Float64)

# materialize so the stored midpoint/radius are plain CuArrays (dispatch fires)
gpu(A::BallMatrix) = BallMatrix(CuArray(Matrix(mid(A))), CuArray(Matrix(rad(A))))

gt(f; nw=2, nr=5) = (for _ in 1:nw; f(); CUDA.synchronize(); end;
    ts = Float64[]; for _ in 1:nr; t0 = time_ns(); f(); CUDA.synchronize();
    push!(ts, (time_ns()-t0)/1e9); end; minimum(ts))

ct(f; nr=3) = (f(); minimum(@elapsed(f()) for _ in 1:nr))

# identity ball matrix on the same backend as X (avoids `- I` scalar indexing on GPU)
eyelike(X::BallMatrix) = (n = size(X, 2); m = mid(X);
    BallMatrix(convert(typeof(m), Matrix{eltype(m)}(I, n, n)), zero(m)))

# the certification products of _certify_svd_impl (apply_vbd=false), as a closure
# over BallMatrices U, Σ, Vt, V, A — works for both CPU and GPU operands.
# Ut is U' materialized so its storage stays a plain array (GPU dispatch needs that).
function certify_products(U, Ut, Σ, Vt, V, A, Ivt, Iuu)
    E = U * Σ * Vt - A
    F = Vt * V - Ivt
    G = Ut * U - Iuu
    R = E + U * Σ * Vt          # stand-in for U*ΔΣ*Vt (same shape/cost)
    return E, F, G, R
end

function run(n; seed=1)
    rng = MersenneTwister(seed)
    A = BallMatrix(randn(rng, n, n) ./ sqrt(n))
    # --- CPU reference ---
    t_fact = ct(() -> svd(A.c))
    sv = svd(A.c)
    Ucpu = BallMatrix(sv.U); Vcpu = BallMatrix(sv.V); Vtcpu = BallMatrix(sv.Vt)
    Σcpu = BallMatrix(Diagonal(sv.S)); Acpu = A
    Utcpu = BallMatrix(Matrix(sv.U'))     # materialized adjoint (mirror GPU path)
    Ivt_c = eyelike(Vtcpu * Vcpu); Iuu_c = eyelike(Utcpu * Ucpu)
    t_prod_cpu = ct(() -> certify_products(Ucpu, Utcpu, Σcpu, Vtcpu, Vcpu, Acpu, Ivt_c, Iuu_c))
    t_total_cpu = ct(() -> rigorous_svd(A; apply_vbd=false))

    # --- GPU operands (CUDAExt MMul4) ---
    Ug = gpu(Ucpu); Vg = gpu(Vcpu); Vtg = gpu(Vtcpu); Σg = gpu(Σcpu); Ag = gpu(A)
    Ugt = BallMatrix(CuArray(Matrix(sv.U')), CUDA.zeros(Float64, size(sv.U,2), size(sv.U,1)))
    Ivt_g = eyelike(Vtg * Vg); Iuu_g = eyelike(Ugt * Ug)
    prods_gpu() = certify_products(Ug, Ugt, Σg, Vtg, Vg, Ag, Ivt_g, Iuu_g)
    t_prod_gpu = gt(prods_gpu)

    # correctness: GPU residual must overlap CPU residual (same midpoint SVD)
    if n == 512
        Ec, _, _, _ = certify_products(Ucpu, Utcpu, Σcpu, Vtcpu, Vcpu, Acpu, Ivt_c, Iuu_c)
        Eg, _, _, _ = prods_gpu()
        dmid = maximum(abs.(mid(Ec) .- Array(mid(Eg))))
        @printf("  [check n=512] max |E_cpu - E_gpu| midpoint = %.2e\n", dmid)
    end

    @printf("n=%-5d | CPU total %.3fs  (factor %.3fs, certify-prods %.3fs = %.0f%%) | GPU prods %.3fs  -> prods speedup %.1fx\n",
        n, t_total_cpu, t_fact, t_prod_cpu, 100*t_prod_cpu/t_total_cpu, t_prod_gpu, t_prod_cpu/t_prod_gpu)
    flush(stdout)
end

println("Device: ", CUDA.name(CUDA.device()))
@printf("CUDAExt loaded: %s\n\n", !isnothing(Base.get_extension(BallArithmetic, :CUDAExt)))
for n in (512, 1024, 2048, 4096); run(n); end
