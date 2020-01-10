function find_arrest(S₀, S₁; rtol = eps(), artol = sqrt(eps()))
    grid = S₁.grid
    Λᵤ = Interpolator(S₀, S₁)
    ws = ApproximationGrids.weights(grid)
    avars, kvars, D₀ = initialize_asymptotics(S₁)

    K = nodes(grid)
    S = kvars.svars.S
    υ = kvars.svars.υ
    w = kvars.svars.w
    d = dimensionality(S₁.f.liquid)
    Z = similar(w)
    T = eltype(K)

    i  = argmax(Λᵤ.A)
    u₀ = zero(T)
    u₁ = log(floatmax(T)) / abs(Λᵤ.A[i])
    uₘ = u₁ / 2

    g = LiquidsDynamics.FixedPoint(Z, kvars, D₀)

    while true
        interpolate!(S, Λᵤ, uₘ)
        LiquidsDynamics.asymptotics_weights!(w, ws, K, S, d, grid)
        ζ = υ * sum(w)

        asymptotics!(g, avars, ζ; rtol = artol)

        if iszero(avars.ζ∞[])
            u₀ = uₘ
        else
            u₁ = uₘ
        end

        if isapprox(u₀, u₁, rtol = rtol, nans = true)
            if iszero(avars.ζ∞[])
                interpolate!(S, Λᵤ, u₁)
                LiquidsDynamics.asymptotics_weights!(w, ws, K, S, d, grid)
                asymptotics!(g, avars, υ * sum(w); rtol = artol)
            end
            break
        end

        uₘ = (u₀ + u₁) / 2
    end

    return u₀, u₁, avars.ζ∞[]
end
