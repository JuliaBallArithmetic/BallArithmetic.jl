using BallArithmetic
using Documenter
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style = :numeric
)

DocMeta.setdocmeta!(BallArithmetic, :DocTestSetup, :(using BallArithmetic);
    recursive = true)

makedocs(;
    plugins = [bib],
    modules = [BallArithmetic],
    authors = "Luca Ferranti",
    repo = "https://github.com/lucaferranti/BallArithmetic.jl/blob/{commit}{path}#{line}",
    sitename = "BallArithmetic.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://lucaferranti.github.io/BallArithmetic.jl",
        edit_link = "main",
        assets = String["assets/citations.css"]),
    pages = [
        "Home" => "index.md"
    ])

deploydocs(;
    repo = "github.com/lucaferranti/BallArithmetic.jl",
    devbranch = "main")
