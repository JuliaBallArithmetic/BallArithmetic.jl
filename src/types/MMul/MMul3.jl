# Midpoint-radius interval matrix multiplication in three point matrix products.
#
#   S. M. Rump, Fast interval matrix multiplication, Numer. Algorithms 61 (2012) 1-34,
#   doi 10.1007/s11075-011-9524-z, Algorithm 4.7 (IImul3) for two interval matrices and
#   Algorithm 4.3 (FImul2) for a point matrix and an interval matrix, without the conversions
#   from and to the infimum-supremum representation; Algorithm 4.7 is Algorithm 1 (MMMul3) of
#   N. Revol and P. Théveny, Parallel implementation of interval matrix multiplication,
#   Reliable Computing 19(1) (2013) 91-106, https://hal.science/hal-00801890.
#
# With eps = 2^-53, realmin the smallest positive normalised number and n the inner dimension,
# Algorithm 4.7 is
#
#     mC  = fl(mA mB)                                       rounding to nearest
#     rB' = fl△((n + 2) eps |mB| + rB)
#     rC  = fl△(|mA| rB' + realmin + rA (|mB| + rB))
#
# and is correct, also in the presence of underflow, provided 2(n + 2) eps ≤ 1 (Theorem 2.1 of
# Rump's paper, which assumes that the products are computed in the same order). Below
# `ϵ = machine_epsilon(T)` is 2 eps, so the first term of rB' is twice the paper's, and
# `η_val / ϵ` is realmin. The papers are for real matrices, and so are the methods: a complex
# product goes through `*`, which forms (A + iB)(C + iD) = AC − BD + i(AD + BC) from four real
# products.

# the hypothesis 2(n + 2) eps ≤ 1 of the algorithms
function _mmul3_check(k::Integer, ::Type{T}) where {T}
    2 * (k + 2) * unit_roundoff(T) <= 1 ||
        throw(ArgumentError("MMul3: inner dimension $k too large, 2(k + 2)eps ≤ 1 is required"))
    return nothing
end

function MMul3(A::BallMatrix{T, T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    m, k = size(A)
    _mmul3_check(k, T)
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)
    mC = mA * mB
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    abs_mB = abs_preserving_structure(mB)
    abs_mA = abs_preserving_structure(mA)
    # all upward: r'B = (k + 2)ϵ|mB| + rB, rC = |mA| r'B + rA (|mB| + rB) + η/ϵ
    rprimeB = _add_up(_scale_up((k + 2) * ϵ, abs_mB), rB)
    rC = _add_up(_sum_product_up(_product_up(abs_mA, rprimeB), rA, _add_up(abs_mB, rB)), η_val / ϵ)
    BallMatrix(mC, rC)
end

function MMul3(A::BallMatrix{T, T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    _mmul3_check(k, T)
    mA, rA = mid(A), rad(A)
    mC = mA * B
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    abs_B = abs_preserving_structure(B)
    abs_mA = abs_preserving_structure(mA)
    # all upward: r'B = (k + 2)ϵ|B|, rC = |mA| r'B + rA |B| + η/ϵ
    rprimeB = _scale_up((k + 2) * ϵ, abs_B)
    rC = _add_up(_sum_product_up(_product_up(abs_mA, rprimeB), rA, abs_B), η_val / ϵ)
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::BallMatrix{T, T}) where {T <: AbstractFloat}
    m, k = size(A)
    _mmul3_check(k, T)
    mB, rB = mid(B), rad(B)
    mC = A * mB
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    abs_mB = abs_preserving_structure(mB)
    abs_A = abs_preserving_structure(A)
    # all upward: r'B = (k + 2)ϵ|mB| + rB, rC = |A| r'B + η/ϵ
    rprimeB = _add_up(_scale_up((k + 2) * ϵ, abs_mB), rB)
    rC = _add_up(_product_up(abs_A, rprimeB), η_val / ϵ)
    BallMatrix(mC, rC)
end

function MMul3(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    m, k = size(A)
    _mmul3_check(k, T)
    mC = A * B
    ϵ = machine_epsilon(T)
    η_val = subnormal_min(T)
    abs_B = abs_preserving_structure(B)
    abs_A = abs_preserving_structure(A)
    # all upward: r'B = (k + 2)ϵ|B|, rC = |A| r'B + η/ϵ
    rprimeB = _scale_up((k + 2) * ϵ, abs_B)
    rC = _add_up(_product_up(abs_A, rprimeB), η_val / ϵ)
    BallMatrix(mC, rC)
end
