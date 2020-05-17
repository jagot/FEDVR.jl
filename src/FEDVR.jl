__precompile__()

module FEDVR

function __init__()
    @warn "The FEDVR.jl package has been deprecated in favour of JuliaApproximation/CompactBases.jl"
    nothing
end

using SparseArrays
using RecipesBase

include("grid.jl")
include("lagrange.jl")
include("basis.jl")
include("derivatives.jl")
include("projections.jl")

end # module
