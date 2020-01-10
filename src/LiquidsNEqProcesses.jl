__precompile__(false)

module LiquidsNEqProcesses


### Imports
using Reexport
using Parameters
using LinearAlgebra: I

@reexport using LiquidsDynamics


### Exports
export Interpolator, find_arrest, interpolate!


### Implementation
include("interpolation.jl")
include("arrest.jl")


end # module
