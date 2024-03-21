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
