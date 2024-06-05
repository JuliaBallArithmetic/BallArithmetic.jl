function _refine_enclosure_newton_flow(T, enc, ϵ; rel_err = 1 / 256, τ = 1.0)
    out_z = []
    out_bound = []
    out_radiuses = []
    out_gradient = []
    for i in 1:length(enc.points)
        σ = enc.bounds[i].c
        if abs((σ - ϵ) / ϵ) < rel_err
            push!(out_z, enc.points[i])
            push!(out_bound, enc.bounds[i])
            push!(out_radiuses, enc.radiuses[i])
            push!(out_gradient, enc.gradient[i])
        else
            z = enc.points[i]
            grad = enc.gradient[i]
            z_new = z + τ * (σ - ϵ) / grad
            if i > 1
                # Remark that the radius is built so that
                # the new sequence is a pearl necklace
                z_old = out_z[i - 1]
                r_old = out_radiuses[i - 1]
                r_new = max(
                    65 * (abs(z_old - z_new) - r_old) / 128,
                    abs(z_old - z_new) / 2)
            else
                r_new = enc.radiuses[i]
            end
            push!(out_z, z_new)
            push!(out_radiuses, r_new)

            K = svd(T - z_new * I)
            z_ball = Ball(z_new, r_new)

            u = K.U[:, end]
            v = K.V[:, end]

            bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
            push!(out_bound, bound)
            push!(out_gradient, (u' * v))
        end
    end
    return Enclosure(enc.λ, out_z, out_bound, out_radiuses, out_gradient, true)
end

function _refine_enclosure_guarantee(T, enc, ϵ)
    out_z = []
    out_bound = []
    out_radiuses = []
    out_gradient = []
    for i in 1:length(enc.points)
        σ = @down enc.bounds[i].c - enc.bounds[i].r
        if σ > ϵ
            push!(out_z, enc.points[i])
            push!(out_bound, enc.bounds[i])
            push!(out_radiuses, enc.radiuses[i])
            push!(out_gradient, enc.gradient[i])
        else
            if i != 1
                z_new = enc.points[i] +
                        (2 / 3) * enc.radiuses[i] * (enc.points[i - 1] - enc.points[i]) /
                        abs(enc.points[i - 1] - enc.points[i])
                r_new = 43 * enc.radiuses[i] / 128
                push!(out_z, z_new)
                push!(out_radiuses, r_new)

                K = svd(T - z_new * I)
                z_ball = Ball(z_new, r_new)

                u = K.U[:, end]
                v = K.V[:, end]

                bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
                push!(out_bound, bound)
                push!(out_gradient, (u' * v))
            end
            z_new = enc.points[i] / 2
            r_new = 43 * enc.radiuses[i] / 128
            push!(out_z, z_new)
            push!(out_radiuses, r_new)

            K = svd(T - z_new * I)
            z_ball = Ball(z_new, r_new)

            u = K.U[:, end]
            v = K.V[:, end]

            bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
            push!(out_bound, bound)
            push!(out_gradient, (u' * v))
            if i != length(enc.points)
                z_new = enc.points[i] +
                        (2 / 3) * enc.radiuses[i] * (enc.points[i + 1] - enc.points[i]) /
                        abs(enc.points[i + 1] - enc.points[i])
                r_new = 43 * enc.radiuses[i] / 128
                push!(out_z, z_new)
                push!(out_radiuses, r_new)

                K = svd(T - z_new * I)
                z_ball = Ball(z_new, r_new)

                u = K.U[:, end]
                v = K.V[:, end]

                bound = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]
                push!(out_bound, bound)
                push!(out_gradient, (u' * v))
            end
        end
    end
    return Enclosure(enc.λ, out_z, out_bound, out_radiuses, out_gradient, true)
end

function refine_enclosure_svd(enc)
    out_z = []
    out_bound = []
    out_radiuses = []

    for (i, z) in enumerate(enc.points)

        #TODO add the special cases for i=1 and i=end

        if 0 ∈ enc.bounds[i]
            z_minus_one = enc.points[i - 1]
            r_minus_one = enc.radiuses[i - 1]

            z_r = enc.radiuses[i]

            z_mid_minus = z_minus_one + r_minus_one * (z - z_minus_one)

            # this guarantees that the balls cover the old ball centered in z
            r_mid_minus = max(
                17 / 32 * abs(z - z_mid_minus), 129 / 128 * (z_r - abs(z - z_mid_minus)))
            K = svd(T - z_mid_minus * I)
            z_ball = Ball(z, r_mid_minus)
            bound_mid_minus = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]

            z_plus_one = enc.points[i + 1]
            r_plus_one = enc.radiuses[i + 1]

            z_mid_plus = z_plus_one + r_plus_one * (z - z_plus_one)

            # this guarantees that the balls cover the old ball centered in z
            r_mid_plus = max(
                17 / 32 * abs(z - z_mid_plus), 129 / 128 * (z_r - abs(z - z_mid_plus)))
            K = svd(T - z_mid_plus * I)
            z_ball = Ball(z, r_mid_plus)
            bound_mid_plus = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]

            # remark that the radiuses of the balls centered in z_mid_plus
            # and z_mid_minus are at least 17/32* abs(z - z_mid_plus), etc...
            new_r_z = max(17 / 32 * abs(z - z_mid_plus), 17 / 32 * abs(z - z_mid_minus))
            K = svd(T - z * I)
            z_ball = Ball(z, new_r_z)
            new_bound_z = _certify_svd(BallMatrix(T) - z_ball * I, K)[end]

            push!(out_z, z_mid_minus)
            push!(out_radiuses, r_mid_minus)
            push!(out_bound, bound_mid_minus)

            push!(out_z, z)
            push!(out_radiuses, new_r_z)
            push!(out_bound, new_bound_z)

            push!(out_z, z_mid_plus)
            push!(out_radiuses, r_mid_plus)
            push!(out_bound, bound_mid_plus)

        else
            push!(out_z, enc.points[i])
            push!(out_radiuses, enc.radiuses[i])
            push!(out_bound, enc.bounds[i])
        end
    end
    return Enclosure(enc.λ, out_z, out_bound, out_radiuses, loop_closure)
end
