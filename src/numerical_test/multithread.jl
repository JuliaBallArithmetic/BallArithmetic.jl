module NumericalTest

export rounding_test

function _test_matrix(k)
    A = zeros(Float64, (k, k))
    A[:, end] = fill(2^(-53), k)
    for i in 1:(k - 1)
        A[i, i] = 1.0
    end
    return A
end

using LinearAlgebra
"""
    rounding_test(n, k)

Let `u=fill(2^(-53), k-1)` and let A be the matrix
[I u;
0 2^(-53)]

This test checks the result of A*A' in different rounding modes,
running BLAS on `n` threads: the diagonal entries but the last are `1 + u²` exactly, so they
must be `nextfloat(1.0)` under `RoundUp` and `1.0` under `RoundDown`.

This is the example of Section 5.1.3 of the thesis below (Listing 5.2), which reports that
OpenBLAS 0.1.0 with more than one thread returned `1` under `RoundUp` in the last three
quarters of the diagonal for `n ≥ 16`.

# Reference

P. Théveny, *Numerical Quality and High Performance in Interval Linear Algebra on Multi-Core
Processors*, doctoral thesis, École normale supérieure de Lyon, 2014, NNT 2014ENSL0941,
https://theses.hal.science/tel-01126973.
"""
function rounding_test(n, k)
    BLAS.set_num_threads(n)
    A = _test_matrix(k)

    test_up = false
    B = setrounding(Float64, RoundUp) do
        BLAS.gemm('N', 'T', 1.0, A, A)
    end
    test_up = all([B[i, i] == nextfloat(1.0) for i in 1:(k - 1)])

    test_down = false
    B = setrounding(Float64, RoundDown) do
        BLAS.gemm('N', 'T', 1.0, A, A)
    end
    test_down = all([B[i, i] == 1.0 for i in 1:(k - 1)])

    return test_up && test_down
end

using RoundingEmulator: add_up, sub_up, sub_down, mul_up

"""
    ARRAY_ROUNDING_OK

Whether [`array_rounding_test`](@ref) passed when the package was loaded. The sums of matrices
that the package forms under a rounding mode fall back to entrywise directed operations when it
is `false`.
"""
const ARRAY_ROUNDING_OK = Ref(true)

# An n × n matrix of type T with full significands and exponents from -3 to 3, the same on every
# machine: sums of two such entries are rounded.
function _array_test_matrix(::Type{T}, n::Integer, seed::Integer) where {T}
    A = Matrix{T}(undef, n, n)
    s = UInt64(seed)
    for i in eachindex(A)
        s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f
        m = Float64(s >> 11) / 2.0^53 + 0.5
        e = Int((s >> 3) % 7) - 3
        A[i] = T(m * 2.0^e) * (iseven(s >> 7) ? 1 : -1)
    end
    return A
end

"""
    array_rounding_test(T = Float64; n = 48) -> Vector{String}

The array operations that the package performs under a rounding mode, each compared entry by
entry with the same operation done with the directed operations of RoundingEmulator. Returns the
names of the forms that differ; an empty vector when all agree.

A product of floating-point matrices is an OpenBLAS call, and the build loaded by the package
makes every BLAS thread take the rounding mode of the caller ([`rounding_test`](@ref) checks
that). A sum of matrices, a multiple, a difference are Julia's own loops, and the OpenBLAS build
has no part in them: that they round as the mode says is a property of the code the Julia
compiler emits for this machine, which Julia does not document as guaranteed. This test is that
check, and it is run when the package is loaded.

The forms: the radius of a sum, `ϵ|C| + r_A + r_B` upward; `P + R` upward and `P − R` downward,
with matrix operations and with broadcasting; the centre `(L + U)/2` and the radius `M − L` of
two bounds, upward; each on dense matrices and on `Symmetric` ones.
"""
function array_rounding_test(::Type{T} = Float64; n::Integer = 48) where {T <: Union{Float32, Float64}}
    C, P = _array_test_matrix(T, n, 1), _array_test_matrix(T, n, 2)
    rA, rB = abs.(_array_test_matrix(T, n, 3)), abs.(_array_test_matrix(T, n, 4))
    ϵ = eps(T)
    failed = String[]
    for (tag, wrap) in (("dense", identity), ("Symmetric", Symmetric))
        c, p, a, b = wrap(abs.(C)), wrap(P), wrap(rA), wrap(rB)
        cm, pm, am, bm = Matrix(c), Matrix(p), Matrix(a), Matrix(b)
        R = setrounding(T, RoundUp) do
            (ϵ * c + a) + b
        end
        Matrix(R) == add_up.(add_up.(mul_up.(ϵ, cm), am), bm) ||
            push!(failed, "$T $tag: ϵ|C| + rA + rB upward")
        U = setrounding(T, RoundUp) do
            p + a
        end
        Matrix(U) == add_up.(pm, am) || push!(failed, "$T $tag: P + R upward")
        L = setrounding(T, RoundDown) do
            p - a
        end
        Matrix(L) == sub_down.(pm, am) || push!(failed, "$T $tag: P − R downward")
        Ub = setrounding(T, RoundUp) do
            p .+ a .+ b
        end
        Matrix(Ub) == add_up.(add_up.(pm, am), bm) || push!(failed, "$T $tag: P .+ R .+ S upward")
        M, Rm = setrounding(T, RoundUp) do
            m = (L + U) / 2
            m, m - L
        end
        Lm, Um = Matrix(L), Matrix(U)
        Mref = add_up.(Lm, Um) ./ 2
        Matrix(M) == Mref || push!(failed, "$T $tag: (L + U)/2 upward")
        Matrix(Rm) == sub_up.(Mref, Lm) || push!(failed, "$T $tag: M − L upward")
    end
    return failed
end

# every form on both types; sets the flag and returns the failed forms
function check_array_rounding()
    failed = vcat(array_rounding_test(Float64), array_rounding_test(Float32))
    ARRAY_ROUNDING_OK[] = isempty(failed)
    return failed
end

end
