# The linear programme of the shaving method (Section 5.9.1 of Horáček's thesis), solved by HiGHS
# through JuMP. The solution is a guess for `interval_shaving`, which verifies what it builds
# from it; nothing here is rigorous.
module LinearProgrammingExt

using BallArithmetic
using JuMP
using HiGHS

function BallArithmetic._shaving_lp(c::AbstractVector, Aeq::AbstractMatrix, beq::AbstractVector,
        lb::AbstractVector, ub::AbstractVector)
    N = length(c)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, x[1:N])
    for i in 1:N
        isfinite(lb[i]) && JuMP.set_lower_bound(x[i], lb[i])
        isfinite(ub[i]) && JuMP.set_upper_bound(x[i], ub[i])
    end
    JuMP.@constraint(model, Aeq * x .== beq)
    JuMP.@objective(model, Min, sum(c[i] * x[i] for i in 1:N))
    JuMP.optimize!(model)
    JuMP.termination_status(model) == JuMP.MOI.OPTIMAL || return nothing
    return JuMP.value.(x)
end

end
