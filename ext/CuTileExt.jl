module CuTileExt

using BallArithmetic
using cuTile: cuTile
import cuTile as ct
using CUDACore

# One thread block computes one (tm × tn) output tile of the BallMatrix
# product MMul4(A, B) using cuTile's elementwise tile ops. The tile-level
# `mma` intrinsic (which backs `muladd(tile, tile, tile)`) silently
# ignores @fpmode because tensor-core MMA is hardware-fixed to round-to-
# nearest-even, so this kernel deliberately accumulates as outer-products
# of (tm × 1) × (1 × tn) tiles under @fpmode, which lowers to addf/mulf
# intrinsics that *do* honor the rounding attribute.
function _mmul4_kernel!(mC::ct.TileArray{T,2}, rC::ct.TileArray{T,2},
                       mA::ct.TileArray{T,2}, rA::ct.TileArray{T,2},
                       mB::ct.TileArray{T,2}, rB::ct.TileArray{T,2},
                       tm::Int, tn::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)
    K = size(mA, 2)

    acc_up = zeros(T, tm, tn)
    acc_r  = zeros(T, tm, tn)

    ct.@fpmode rounding_mode=ct.Rounding.PosInf flush_to_zero=false begin
        for k in Int32(1):Int32(K)
            mA_col = ct.load(mA; index=(bid_m, k), shape=(tm, 1),
                             padding_mode=ct.PaddingMode.Zero)
            mB_row = ct.load(mB; index=(k, bid_n), shape=(1, tn),
                             padding_mode=ct.PaddingMode.Zero)
            rA_col = ct.load(rA; index=(bid_m, k), shape=(tm, 1),
                             padding_mode=ct.PaddingMode.Zero)
            rB_row = ct.load(rB; index=(k, bid_n), shape=(1, tn),
                             padding_mode=ct.PaddingMode.Zero)

            absA = cuTile.Intrinsics.absf(mA_col)
            absB = cuTile.Intrinsics.absf(mB_row)

            acc_up = acc_up + mA_col * mB_row
            acc_r  = acc_r  + absA * rB_row + rA_col * (absB + rB_row)
        end
        acc_up = acc_up + acc_r        # C2
    end

    acc_dn = zeros(T, tm, tn)
    ct.@fpmode rounding_mode=ct.Rounding.NegInf flush_to_zero=false begin
        for k in Int32(1):Int32(K)
            mA_col = ct.load(mA; index=(bid_m, k), shape=(tm, 1),
                             padding_mode=ct.PaddingMode.Zero)
            mB_row = ct.load(mB; index=(k, bid_n), shape=(1, tn),
                             padding_mode=ct.PaddingMode.Zero)
            acc_dn = acc_dn + mA_col * mB_row
        end
        acc_dn = acc_dn - acc_r        # C1
    end

    ct.@fpmode rounding_mode=ct.Rounding.PosInf flush_to_zero=false begin
        mid_tile = (acc_dn + acc_up) * T(0.5)
        rad_tile = mid_tile - acc_dn
        ct.store(mC; index=(bid_m, bid_n), tile=mid_tile)
        ct.store(rC; index=(bid_m, bid_n), tile=rad_tile)
    end
    return nothing
end

# NOTE: this cuTile outer-product kernel is kept for reference (it is the only
# directed-rounding GPU path — see ext/RIGOROUS_GPU_GEMM.md, Route 1). The
# production GPU `MMul4` dispatch now lives in `CUDAExt.jl` (INT8 Ozaki-II port,
# much faster). To avoid a method clash with that overload on Float64 `CuArray`
# ball matrices, `CuTileExt` no longer registers a `MMul4` method; call the
# kernel directly via `_mmul4_kernel!` if you want to exercise this path.
function _cutile_mmul4(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: Union{Float32, Float64}}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    M_ = size(mA, 1); N_ = size(mB, 2)
    mC = CUDACore.CuArray{T}(undef, M_, N_)
    rC = CUDACore.CuArray{T}(undef, M_, N_)
    tm, tn = 16, 16
    grid = (cld(M_, tm), cld(N_, tn))
    @cuda backend=cuTile blocks=grid _mmul4_kernel!(mC, rC, mA, rA, mB, rB,
        ct.Constant(tm), ct.Constant(tn))
    return BallMatrix(mC, rC)
end

end # module
