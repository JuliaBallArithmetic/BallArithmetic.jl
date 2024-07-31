"""
A newton step to reach the level set
ϵ of the smallest singular value
"""
function _newton_step(z, K::SVD, ϵ)
    u = K.U[:, end]
    v = K.V[:, end]
    σ = K.S[end]

    z = z + (σ - ϵ) / (u' * v)
    return z
end

"""
Return the next point in a level set
of the smallest singular value
"""
function _follow_level_set(z::ComplexF64, τ::Float64, K::SVD)
    u = K.U[:, end]
    v = K.V[:, end]

    # follow the level set
    grad = v' * u
    ort = im * grad

    #@info "|ort| $(abs(ort))"
    z = z + τ * ort / abs(ort)

    return z
end
