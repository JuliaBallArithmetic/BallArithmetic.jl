# struct BallMatrix{T <: AbstractFloat, NT <: Union{T, Complex{T}}, BT <: Ball{T, NT},
#     CM <: AbstractMatrix{NT}, RM <: AbstractMatrix{T}} <: AbstractMatrix{BT}
#     c::CM
#     r::RM
#     function BallMatrix(c::AbstractMatrix{T},
#             r::AbstractMatrix{T}) where {T <: AbstractFloat}
#         new{T, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
#     end
#     function BallMatrix(c::AbstractMatrix{Complex{T}},
#             r::AbstractMatrix{T}) where {T <: AbstractFloat}
#         new{T, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
#     end
# end

"""
    BallMatrix{T, NT, BT, CM, RM}

Alias for the two-dimensional [`BallArray`](@ref) used to represent
matrices with rigorous error control. The type parameters mirror those of
`BallArray` and describe the base floating-point type `T`, the element
type `NT`, the [`Ball`](@ref) container `BT`, and the concrete matrix
types used for the midpoints (`CM`) and radii (`RM`).

Users typically construct instances through the `BallMatrix`
constructors below rather than specifying these parameters explicitly.
"""
const BallMatrix{T, NT, BT, CM, RM} = BallArray{T, 2, NT, BT, CM, RM}
# NOTE: we keep a short comment block that mirrors the five parameters of
# the alias.  This keeps the mapping between the abstract type variables
# and the physical storage (midpoints/radii) fresh when reading the code
# below.
# * `T`  – the real floating-point type used for the radius entries.
# * `NT` – the numeric type stored in the midpoint matrix.
# * `BT` – the concrete `Ball` element type corresponding to the matrix.
# * `CM` – the concrete container used for the midpoint matrix.
# * `RM` – the concrete container used for the radius matrix.

"""
    BallMatrix(M::AbstractMatrix)

Create a `BallMatrix` from an existing matrix of midpoint values. The
radius of every entry defaults to zero, corresponding to exact entries.
"""
function BallMatrix(M::AbstractMatrix)
    # Construction delegates to the `BallArray` helper so that we reuse the
    # validation of midpoint/radius consistency implemented there.
    #
    # For plain matrices the midpoint is the matrix itself (`mid(M) == M`),
    # while `rad(M)` returns a zero matrix of the appropriate floating-point
    # type.  The combination therefore produces the exact enclosure `M ± 0`.
    return BallArray(mid(M), rad(M))
end

"""
    BallMatrix(c::AbstractMatrix, r::AbstractMatrix)

Create a `BallMatrix` from a matrix of midpoints `c` and a matrix of
non-negative radii `r`. Each entry of the resulting `BallMatrix`
contains the ball `c[i, j] ± r[i, j]`.
"""
function BallMatrix(c::AbstractMatrix, r::AbstractMatrix)
    # The two-argument form exposes the storage order explicitly.  Lower
    # level constructors validate sizes, element types, and the
    # non-negativity of `r`, so higher-level methods can assume consistent
    # data once construction succeeds.
    return BallArray(c, r)
end

"""
    mid(A::AbstractMatrix)

Return the midpoint matrix associated with `A`. For plain matrices the
midpoint equals the matrix itself; for `BallMatrix` values this method is
extended elsewhere to extract the stored midpoint data.
"""
mid(A::AbstractMatrix) = A

"""
    rad(A::AbstractMatrix{T})

Return a matrix of radii matching the size of `A`. For non-ball matrices
this defaults to a zero matrix, while for `BallMatrix` values the method
is overloaded to provide the stored uncertainty information.
"""
rad(A::AbstractMatrix{T}) where {T <: AbstractFloat} = zeros(T, size(A))
# Complex matrices still report a real-valued radius since the uncertainty
# is measured in the underlying real field.
"""
    rad(A::AbstractMatrix{Complex{T}})

Return a matrix of real radii matching the size of the complex matrix `A`.
Even for complex entries the radius is measured over the underlying real
field, hence the resulting matrix has element type `T`.
"""
rad(A::AbstractMatrix{Complex{T}}) where {T <: AbstractFloat} = zeros(T, size(A))

# LinearAlgebra functions
"""
    adjoint(M::BallMatrix)

Return the conjugate transpose of `M`, preserving rigorous enclosures by
transposing both the midpoint and the radius matrices.
"""
function LinearAlgebra.adjoint(M::BallMatrix)
    # The midpoint and radius matrices are transposed separately.  The
    # radii remain real-valued, so taking the conjugate transpose simply
    # coincides with the transpose of the stored radii.
    return BallMatrix(mid(M)', rad(M)')
end

# Operations
"""
    Base.:+(A::BallMatrix, B::BallMatrix)

Combine two `BallMatrix` values elementwise using addition while tracking
floating-point and enclosure errors. The midpoint matrices are added
directly and the radii are enlarged using outward rounding to maintain a
rigorous enclosure.
"""
function Base.:+(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C = mA + mB
    R = setrounding(T, RoundUp) do
        (ϵp * abs.(C) + rA) + rB
    end
    BallMatrix(C, R)
end

"""
    Base.:-(A::BallMatrix, B::BallMatrix)

Combine two `BallMatrix` values elementwise using subtraction while
tracking floating-point and enclosure errors. The midpoint matrices are
subtracted directly and the radii are enlarged using outward rounding to
maintain a rigorous enclosure.
"""
function Base.:-(A::BallMatrix{T}, B::BallMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)
    mB, rB = mid(B), rad(B)

    C = mA - mB
    R = setrounding(T, RoundUp) do
        (ϵp * abs.(C) + rA) + rB
    end
    BallMatrix(C, R)
end

"""
    *(λ::Number, A::BallMatrix)

Scale the `BallMatrix` `A` by the scalar `λ`. Both the midpoint and the
radius are scaled, and an outward-rounded padding proportional to
floating-point error is added so the result remains a rigorous enclosure.
"""
function Base.:*(lam::Number, A::BallMatrix{T}) where {T}
    # Prepare a mutable midpoint buffer with the correct promoted element
    # type before performing the scaling.
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c), typeof(lam)}))

    # Scale the midpoint matrix using the standard matrix-scalar product.
    B = lam * A.c

    R = setrounding(T, RoundUp) do
        # The resulting radius is composed of three pieces:
        #   • `η` to account for gradual underflow;
        #   • a proportional floating-point error term;
        #   • the original radii scaled by `|λ|`.
        return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
    end

    return BallMatrix(B, R)
end

"""
    *(λ::Ball, A::BallMatrix)

Scale `A` by a scalar `Ball`. The midpoint is scaled by the midpoint of
`λ` and the radius is enlarged to account for both the uncertainty in
`λ` and floating-point rounding.
"""
function Base.:*(lam::Ball{T, NT}, A::BallMatrix{T}) where {T, NT <: Union{T, Complex{T}}}
    # As above, allocate mutable storage that matches the promoted midpoint
    # type (here determined by the midpoint of `λ`).
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c),
                typeof(mid(lam))}))

    # Only the midpoint of the scalar contributes to the midpoint of the
    # result; the radius enters the enclosure bookkeeping below.
    B = mid(lam) * A.c

    R = setrounding(T, RoundUp) do
        # The uncertainty now has to capture the radius of `λ` as well.  Each
        # midpoint entry is magnified by `rad(λ)` and combined with the input
        # radii, in addition to the floating-point padding described above.
        return (η .+ ϵp * abs.(B)) + ((abs.(A.c) + A.r) * rad(lam) + A.r * abs(mid(lam)))
    end

    return BallMatrix(B, R)
end

# function Base.:*(lam::NT, A::BallMatrix{T}) where {T, NT<:Union{T,Complex{T}}}
#     B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c),typeof(mid(lam))}))

#     B = lam * A.c

#     R = setrounding(T, RoundUp) do
#         return (η .+ ϵp * abs.(B)) + (A.r * abs(mid(lam)))
#     end

#     return BallMatrix(B, R)
# end

"""
    Base.:+(A::BallMatrix, B::AbstractMatrix)

Add a plain matrix to a `BallMatrix` by combining the midpoints directly
and inflating the radii to account for the existing uncertainty in `A`
and the floating-point error introduced by the addition.
"""
function Base.:+(A::BallMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)

    # Apply the operation directly to the midpoint data.
    C = mA + B

    R = setrounding(T, RoundUp) do
        # Only the `BallMatrix` contributes an existing radius, but
        # we still need to compensate for floating-point error in the
        # combined midpoint.
        return (ϵp * abs.(C) + rA)
    end
    BallMatrix(C, R)
end

"""
    Base.:+(B::AbstractMatrix, A::BallMatrix)

Commutative counterpart of [`+(::BallMatrix, ::AbstractMatrix)`](@ref),
allowing the plain matrix to appear on the left-hand side.
"""
function Base.:+(B::AbstractMatrix{T}, A::BallMatrix{T}) where {T <: AbstractFloat}
    # Swap the arguments so both orders share the same code path.
    return A + B
end

"""
    Base.:-(A::BallMatrix, B::AbstractMatrix)

Subtract a plain matrix from a `BallMatrix`. The midpoint subtraction is
performed elementwise while the radius is enlarged to remain enclosure
safe.
"""
function Base.:-(A::BallMatrix{T}, B::AbstractMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)

    # Subtract the midpoint data directly.
    C = mA - B

    R = setrounding(T, RoundUp) do
        # Only `A` carries an existing radius; nevertheless we must still
        # compensate for floating-point roundoff in the midpoint result.
        return (ϵp * abs.(C) + rA)
    end
    BallMatrix(C, R)
end

"""
    Base.:-(B::AbstractMatrix, A::BallMatrix)

Subtract a `BallMatrix` from a plain matrix, reusing the implementation
of [`-(::BallMatrix, ::AbstractMatrix)`](@ref) by swapping the argument
order.
"""
function Base.:-(B::AbstractMatrix{T}, A::BallMatrix{T}) where {T <: AbstractFloat}
    mA, rA = mid(A), rad(A)

    # Compute the midpoint part of `B - A` directly.
    C = B - mA

    R = setrounding(T, RoundUp) do
        # The stored radii of `A` still drive the enclosure, with additional
        # padding for floating-point roundoff in the midpoint subtraction.
        return (ϵp * abs.(C) + rA)
    end
    BallMatrix(C, R)
end

"""
    +(A::BallMatrix, J::UniformScaling)

Add a `UniformScaling` operator (such as `I`) to a square `BallMatrix`.
The diagonal of the midpoint is shifted by `J` and the corresponding
radii are enlarged to maintain a valid enclosure.
"""
function Base.:+(A::BallMatrix{T}, J::UniformScaling) where {T}
    # Uniform scalings only apply to square matrices.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(+,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # Shift the diagonal by the uniform scaling value.
        B[i, i] += J
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

"""
    +(A::BallMatrix, J::UniformScaling{Ball})

Add a ball-valued `UniformScaling` to a `BallMatrix`, incorporating both
the midpoint and the radius of the scaling into the resulting enclosure.
"""
function Base.:+(A::BallMatrix{T},
        J::UniformScaling{Ball{T, NT}}) where {T, NT <: Union{T, Complex{T}}}
    # Uniform scalings require square matrices regardless of scalar type.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # The midpoint of the scaling updates the diagonal entries.
        B[i, i] += J.λ.c
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

"""
    +(J::UniformScaling, A::BallMatrix)

Symmetric counterpart to [`+(::BallMatrix, ::UniformScaling)`](@ref),
allowing the uniform scaling to appear on the left-hand side.
"""
function Base.:+(J::UniformScaling, A::BallMatrix)
    # Delegate to the right-addition overload so both orders share the
    # same implementation.
    return A + J
end

"""
    -(A::BallMatrix, J::UniformScaling)

Subtract a `UniformScaling` operator from a `BallMatrix` while updating
the stored radii so that the result remains enclosure-safe.
"""
function Base.:-(A::BallMatrix{T}, J::UniformScaling) where {T}
    # Subtracting a uniform scaling also requires a square input.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(-,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # Subtract the scaling value from the diagonal entries.
        B[i, i] -= J
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

"""
    -(A::BallMatrix, J::UniformScaling{Ball})

Subtract a ball-valued `UniformScaling` from a `BallMatrix`, accounting
for both the midpoint and radius of the scaling in the resulting
enclosure.
"""
function Base.:-(A::BallMatrix{T},
        J::UniformScaling{Ball{T, NT}}) where {T, NT <: Union{T, Complex{T}}}
    # Same square check for the subtractive variant.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # Remove the midpoint contribution of the scaling from the diagonal.
        B[i, i] -= J.λ.c
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

"""
    -(J::UniformScaling, A::BallMatrix)

Subtract a `BallMatrix` from a uniform scaling operator, useful for
forming expressions such as `λ * I - A` with rigorous error bounds.
"""
function Base.:-(J::UniformScaling, A::BallMatrix{T}) where {T}
    # Enforce the square requirement before forming `J - A`.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c,
        Base._return_type(-,
            Tuple{eltype(A.c), typeof(J)}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # Form the diagonal entries for `J - A` without aliasing the input.
        B[i, i] = J - B[i, i]
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i])
        end
        return R
    end
    return BallMatrix(B, R)
end

"""
    -(J::UniformScaling{Ball}, A::BallMatrix)

Subtract a `BallMatrix` from a ball-valued uniform scaling operator,
expanding the resulting radii by both the uncertainty in `J` and the
rounding performed during the subtraction.
"""
function Base.:-(J::UniformScaling{Ball{T, NT}},
        A::BallMatrix{T}
) where {T, NT <: Union{T, Complex{T}}}
    # Validate the shape before computing the difference.
    LinearAlgebra.checksquare(A)
    B = LinearAlgebra.copymutable_oftype(A.c, Base._return_type(+, Tuple{eltype(A.c), NT}))
    R = copy(A.r)
    @inbounds for i in axes(A, 1)
        # Start from the midpoint of the scaling and subtract the midpoint of
        # the matrix to obtain the new diagonal entry.
        B[i, i] = J.λ.c - B[i, i]
    end

    R = setrounding(T, RoundUp) do
        @inbounds for i in axes(A, 1)
            R[i, i] += ϵp * abs(B[i, i]) + J.λ.r
        end
        return R
    end
    return BallMatrix(B, R)
end

include("MMul/abs_preserving_structure.jl")
include("MMul/MMul2.jl")
include("MMul/MMul3.jl")
include("MMul/MMul4.jl")
include("MMul/MMul5.jl")
# The specialised kernels above expose rigorous, rounded matrix
# multiplication routines of increasing blocking order.  The fourth-order
# variant is currently used by default in the high-level `*` methods, while
# the others remain available for experimentation and benchmarking.

"""
    *(A::BallMatrix, B::BallMatrix)

Multiply two compatible `BallMatrix` values using the rigorously rounded
matrix multiplication kernels provided in the `MMul` submodule.
"""
function Base.:*(A::BallMatrix{T, S}, B::BallMatrix{T, S}) where {S, T <: AbstractFloat}
    return MMul4(A, B)
end

"""
    *(A::BallMatrix, B::AbstractMatrix)

Multiply a `BallMatrix` by an arbitrary dense or structured matrix,
propagating the midpoint product and rigorously bounding the resulting
radii.
"""
function Base.:*(A::BallMatrix{T, S}, B::AbstractMatrix{S}) where {S, T <: AbstractFloat}
    return MMul4(A, B)
end

"""
    *(A::AbstractMatrix, B::BallMatrix)

Multiply an arbitrary dense or structured matrix with a `BallMatrix`.
The enclosure of the result is obtained using the same rigorous matrix
multiplication kernel as in the purely ball-valued case.
"""
function Base.:*(A::AbstractMatrix{S}, B::BallMatrix{T, S}) where {S, T <: AbstractFloat}
    # Symmetric case of the mixed overload above.
    return MMul4(A, B)
end

"""
    *(A::BallMatrix{T, Complex{T}}, B::BallMatrix{T, T})

Multiply a complex `BallMatrix` by a real one by dispatching to the real
and imaginary parts separately.
"""
function Base.:*(
        A::BallMatrix{T, Complex{T}},
        B::BallMatrix{T, T}) where {T <: AbstractFloat}
    # Evaluate the complex product componentwise to reuse the rigorous
    # kernels for real-valued multiplications.
    return real(A) * B + im * (imag(A) * B)
end

"""
    *(A::BallMatrix{T, T}, B::BallMatrix{T, Complex{T}})

Multiply a real `BallMatrix` by a complex one, computing the product of
midpoint matrices componentwise and rigorously propagating the radii.
"""
function Base.:*(
        A::BallMatrix{T, T},
        B::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    # Symmetric case: operate on the real and imaginary parts separately to
    # keep the propagation logic identical.
    return A * real(B) + im * (A * imag(B))
end

"""
    *(A::BallMatrix{T, Complex{T}}, B::BallMatrix{T, Complex{T}})

Multiply two complex `BallMatrix` values. The product is formed by
combining the real and imaginary parts via the previously defined real
matrix multiplications.
"""
function Base.:*(
        A::BallMatrix{T, Complex{T}},
        B::BallMatrix{T, Complex{T}}) where {T <: AbstractFloat}
    # Fall back to the mixed overloads so each component uses the same
    # rigorous kernels as the purely real case.
    return A * real(B) + im * (A * imag(B))
end
