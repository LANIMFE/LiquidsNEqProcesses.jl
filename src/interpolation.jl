# Struct for allocating the vectors necessary for interpolation.
struct Interpolator{U, V}
    A ::Vector{U} # coefficient for `u`
    Sf::Vector{V} # pseudo-structure of the final state
    ΔS::Vector{V} # difference between initial structure and `Sf`
end

function Interpolator(S₀::T, S₁::T) where {T}
    @assert S₀.grid == S₁.grid

    D₀ = LiquidsDynamics.diffusion_coeff(S₀.f.liquid)
    K  = nodes(S₀.grid)
    Sf = project.(S₁.f.(K))
    ΔS = project.(S₀.f.(K)) .- Sf
    A  = LiquidsDynamics.bsfactors(-2D₀, K, Sf) .* inv.(Sf)

    U = eltype(A)
    V = eltype(Sf)

    return Interpolator{U, V}(A, Sf, ΔS)
end

function interpolate!(S, interpolator, u)
    @unpack A, Sf, ΔS = interpolator
    @. S = muladd(ΔS, exp(reduce_dof(A * u)), Sf)
    return S
end
