struct BallArray{T<:AbstractFloat,N<:Integer,NT<:Union{T,Complex{T}},BT<:Ball{T,NT},CA<:AbstractArray{NT,N},RA<:AbstractArray{T,N}} <: AbstractArray{BT,N}
    c::CA
    r::RA
    function BallArray(c::AbstractArray{T,N}, r::AbstractArray{T,N}) where {T<:AbstractFloat, N<:Integer}
        new{T,N,T,Ball{T,T},typeof(c),typeof(r)}(c, r)
    end
    function BallArray(c::AbstractArray{Complex{T},N}, r::AbstractArray{T,N}) where {T<:AbstractFloat, N<:Integer}
        new{T,N,Complex{T},Ball{T,Complex{T}},typeof(c),typeof(r)}(c, r)
    end
end