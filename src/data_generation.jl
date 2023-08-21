#=
This module provides functions that generate the datasets
that are then used to calculate fractal dimensions.
The module is structured around functions which accept only keywords,
and always initialize and return a `StateSpaceSet`.
To add new datasets, simply create a new function here.
All functions use the `N` keyword, which is the length of the dataset,
and necessarily use keyword propagation (`kwargs...`).

Functions of this module are used like so:
```
data = :henonmap # This must be a `Symbol`
X = produce_data(; data, parameters...)
```
and `produce_data` always uses `standardize`.
=#

function produce_data(; data = :name, kwargs...)
    data_producing_function = getfield(Data, data)
    X = data_producing_function(; kwargs...)
    return standardize(X)
end

module Data

using DrWatson
using Random, Statistics, LinearAlgebra
using DynamicalSystemsBase, OrdinaryDiffEq

# default constants used if no other is given
const default_N = 10_000
const diffeq = (alg = Vern9(), reltol = 1e-12, abstol = 1e-12, maxiters = typemax(Int))
const diffeq_lowacc = (alg = Tsit5(), reltol = 1e-9, abstol = 1e-9, maxiters = typemax(Int))

# Kaplan Yorke
function kaplanyorkemap(; N = default_N, λ = 0.2, kwargs...)
    rng = Random.Xoshiro(1234)
    f(u, λ, t) = SVector((2u[1]) % 1.0 + 1e-15rand(rng), λ*u[2] + cospi(4u[1]))
    ds = DeterministicIteratedMap(f, SVector(0.15, 0.2), λ)
    tr, = trajectory(ds, N; Ttr = 1000)
    return tr
end

# Geometric sets
function torus2(; N = default_N, Δt = 0.1, ω = sqrt(3), R = 2.0, r = 1.0, kwargs...)
    function torus(u)
        θ, φ = u
        x = (R + r*cos(θ))*cos(φ)
        y = (R + r*cos(θ))*sin(φ)
        z = r*sin(θ)
        return SVector(x, y, z)
    end
    θs = range(0; step = 0.1, length = N)
    φs = ω .* θs
    return StateSpaceSet([torus(u) for u in zip(θs, φs)])
end
function koch(; maxk = 7, kwargs...)
    flakepoints = SVector{2}.([[0.0; 0.0], [0.5; sqrt(3)/2], [1; 0.0], [0.0; 0.0]])
    function innerkoch(points, maxk, α = sqrt(3)/2)
        Q = SMatrix{2,2}(0, 1, -1, 0)
        for k = 1:maxk
            n = length(points)
            new_points = eltype(points)[]
            for i = 1:n-1
                p1, p2 = points[i], points[i+1]
                v = (p2 - p1) / 3
                q1 = p1 + v
                q2 = p1 + 1.5v + α * Q * v
                q3 = q1 + v
                push!(new_points, p1, q1, q2, q3)
            end
            push!(new_points, points[end])
            points = new_points
        end
        return points
    end
    kochpoints = innerkoch(flakepoints, maxk)
    kochdata = StateSpaceSet(unique!(kochpoints))
end
function uniform_sphere(; N = default_N, kwargs...)
    A = SVector{3, Float64}[]
    i = 0
    while i < N
        x = rand(SVector{3, Float64}) .- 0.5
        if norm(x) ≤ 1/2√3
            push!(A, x)
            i += 1
        end
    end
    return StateSpaceSet(A)
end
function brownian_motion(; N = default_N, D = 3, seed = 4532, kwargs...)
    rng = Random.MersenneTwister(seed)
    return StateSpaceSet(cumsum(randn(rng, N, D); dims = 1))
end

# Standard map
function standardmap(; N = default_N, k = 64.0, kwargs...)
    @inbounds function standardmap_rule(x, par, n)
        theta = x[1]; p = x[2]
        p += par[1]*sin(theta)
        theta += p
        while theta >= 2π; theta -= 2π; end
        while theta < 0; theta += 2π; end
        while p >= 2π; p -= 2π; end
        while p < 0; p += 2π; end
        return SVector(theta, p)
    end
    ds = DeterministicIteratedMap(standardmap_rule, [0.08152, 0.122717], [k])
    tr, = trajectory(ds, N; Ttr = 100)
    return tr
end

# Henon
function henonmap(; N = default_N, a = 1.4, b = 0.3, kwargs...)
    henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    ds = DeterministicIteratedMap(henon_rule, [0.08152, 0.122717], [a, b])
    tr, = trajectory(ds, N; Ttr = 100)
    return tr
end

# Towel
function towelmap(; N = default_N, η = 0, correlated = false, kwargs...)
    function towel_rule(x, p, n)
        @inbounds x1, x2, x3 = x[1], x[2], x[3]
        SVector( 3.8*x1*(1-x1) - 0.05*(x2+0.35)*(1-2*x3),
        0.1*( (x2+0.35)*(1-2*x3) - 1 )*(1 - 1.9*x1),
        3.78*x3*(1-x3)+0.2*x2 )
    end
    ds = DeterministicIteratedMap(towel_rule, [0.085, -0.121, 0.075])
    tr, = trajectory(ds, N; Ttr = 100)
    return tr
end


end
