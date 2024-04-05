struct BallVector{T <: AbstractFloat, NT <: Union{T, Complex{T}}, BT <: Ball{T, NT},
    CV <: AbstractVector{NT}, RV <: AbstractVector{T}} <: AbstractVector{BT}
    c::CV
    r::RV
    function BallVector(c::AbstractVector{T},
            r::AbstractVector{T}) where {T <: AbstractFloat}
        new{T, T, Ball{T, T}, typeof(c), typeof(r)}(c, r)
    end
    function BallVector(c::AbstractVector{Complex{T}},
            r::AbstractVector{T}) where {T <: AbstractFloat}
        new{T, Complex{T}, Ball{T, Complex{T}}, typeof(c), typeof(r)}(c, r)
    end
end
