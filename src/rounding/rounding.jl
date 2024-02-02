const ϵp = 2.0^-52
const η = 2.0^-1074

const op_up = Dict(:+ => :add_up, :- => :sub_up, :* => :mul_up, :/ => :div_up)
macro up(ex)
    esc(MacroTools.postwalk(x -> get(op_up, x, x), ex))
end

const op_down = Dict(:+ => :add_down, :- => :sub_down, :* => :mul_down, :/ => :div_down)
macro down(ex)
    esc(MacroTools.postwalk(x -> get(op_up, x, x), ex))
end
