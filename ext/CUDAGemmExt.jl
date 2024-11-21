module CUDAGemmExt

using CUDA, GemmKernels

# Following
# https://github.com/JuliaGPU/GemmKernels.jl/blob/8c894fd6a6739cc6405515f54f8454abcd2787f6/src/operator.jl#L155C1-L246C1

# ----
# WMMARndUp
# ----

# WMMAOp's register types cannot be configured, and CT/AT should be identical to their
# respective shared memory layouts eltypes. this is because WMMA intrinsics are used
# to load/store shared memory, so we cannot perform any conversions on the fly.
# note that there still can be a conversion between global and shared memory.
struct WMMAOpRndUp{M, N, K, CT, AT} end

@inline shape(::Type{WMMAOpRndUp{M, N, K, CT, AT}}) where {M, N, K, CT, AT} = (
    M = M, N = N, K = K)

for (M, N, K) in [
        (16, 16, 16),
        (8, 32, 16),
        (32, 8, 16)
    ],
    (layout_type, wmma_layout_type) in [
        (Layout.ColMajor, WMMA.ColMajor),
        (Layout.UnsafeAlignedColMajor, WMMA.ColMajor),
        (Layout.RowMajor, WMMA.RowMajor),
        (Layout.UnsafeAlignedRowMajor, WMMA.RowMajor)
    ]

    @eval begin
        # TODO: Have accessors in CUDA.jl to get the fragment sizes?
        # FP16 (16, 16, 16), (8, 32, 16), and (32, 8, 16)
        @inline fragtype_a(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{
            $M, $N, $K, 16, CT, $wmma_layout_type, WMMA.MatrixA}
        @inline fragtype_b(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{CT}}) where {CT, AT} = WMMA.Fragment{
            $M, $N, $K, 16, CT, $wmma_layout_type, WMMA.MatrixB}
        @inline fragtype_accum(::Type{WMMAOp{$M, $N, $K, CT, AT}}, ::Type{$layout_type{AT}}) where {CT, AT} = WMMA.Fragment{
            $M, $N, $K, 8, AT, WMMA.Unspecified, WMMA.Accumulator}
    end
end

# convert_index_func: function used to transpose the index in case of a row-major layout
for (layout_type, wmma_layout_type, convert_index_func) in [
    (Layout.ColMajor, WMMA.ColMajor, identity),
    (Layout.UnsafeAlignedColMajor, WMMA.ColMajor, identity),
    (Layout.RowMajor, WMMA.RowMajor, x -> reverse(Tuple(x))),
    (Layout.UnsafeAlignedRowMajor, WMMA.RowMajor, x -> reverse(Tuple(x)))
]
    @eval begin
        @inline function load_a(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}},
                workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(CT)
            return WMMA.load_a(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_b(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{CT}},
                workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(CT)
            return WMMA.load_b(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function load_c(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}},
                workspace, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(AT)
            return WMMA.load_c(ptr, size(workspace, 1), $wmma_layout_type, conf)
        end

        @inline function store_d(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{$layout_type{AT}},
                workspace, frag, tile::Tile) where {M, N, K, CT, AT}
            conf = WMMA.Config{M, N, K, AT}

            linear_base = linearise($convert_index_func(tile.base), size(workspace))
            linear_offset = linearise($convert_index_func(tile.offset), size(workspace))

            ptr = pointer(workspace, linear_base) + (linear_offset - 1) * sizeof(AT)
            WMMA.store_d(ptr, frag, size(workspace, 1), $wmma_layout_type, conf)
        end
    end
end

@inline function load_c(::Type{WMMAOp{M, N, K, CT, AT}}, ::Type{Layout.Zero{AT}},
        workspace, tile::Tile) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    return WMMA.fill_c(zero(AT), conf)
end

function mma(
        ::Type{WMMAOp{M, N, K, CT, AT}}, a_frag, b_frag, c_frag) where {M, N, K, CT, AT}
    conf = WMMA.Config{M, N, K, AT}
    return WMMA.mma(a_frag, b_frag, c_frag, conf)
end

end
