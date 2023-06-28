using BallArithmetic
using Documenter

DocMeta.setdocmeta!(BallArithmetic, :DocTestSetup, :(using BallArithmetic);
                    recursive = true)

makedocs(;
         modules = [BallArithmetic],
         authors = "Luca Ferranti",
         repo = "https://github.com/lucaferranti/BallArithmetic.jl/blob/{commit}{path}#{line}",
         sitename = "BallArithmetic.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://lucaferranti.github.io/BallArithmetic.jl",
                                  edit_link = "main",
                                  assets = String[]),
         pages = [
             "Home" => "index.md",
         ])

deploydocs(;
           repo = "github.com/lucaferranti/BallArithmetic.jl",
           devbranch = "main")
