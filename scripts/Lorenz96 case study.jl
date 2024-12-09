"""
This file computes and plots quantities related with the regular variation property for the Lorenz 96
system. The first section of the code iterates the Lorenz 63 system and computes the quantities described in
the src/regularly_varying.jl file for a number of points "npoints" randomly selected in the attractor.
The next section interpolates the data obtained in the first section to a common grid of radii, and the
rest just save, read and plot the data.
"""
using DrWatson
@quickactivate "FractalDimensionsQ"

using CairoMakie
using ChaosTools, OrdinaryDiffEq
using Distances
using DynamicalSystemsBase
using DynamicalSystems
using ProgressMeter
using Interpolations
using FractalDimensions
using CSV
using DataFrames

include(srcdir("regularly_varying.jl"))
include(srcdir("theme.jl"))

##################### Iterating the system #############################


diffeq = (alg = Vern9(), abstol = 1e-12, reltol = 1e-9)

function lorenz96_rule!(du, u, p, t)
    F = p[1]; N = length(u)
    du[1] = (u[2] - u[N - 1]) * u[N] - u[1] + F
    du[2] = (u[3] - u[N]) * u[1] - u[2] + F
    du[N] = (u[1] - u[N - 2]) * u[N - 1] - u[N] + F
    for n in 3:(N - 1)
        du[n] = (u[n + 1] - u[n - 2]) * u[n - 1] - u[n] + F
    end
    return nothing
end
p = [32.0]



npoints = 1000
maxtime = 50_000#_000
nexceedences = 5000
interpolatedratio = zeros(100,npoints)
interpolateddim = zeros(100,npoints)
interpolatedextremalindex = zeros(100,npoints)


points = []
for i in 1:npoints+1
    lorenzds = CoupledODEs(lorenz96_rule!,rand(4)*0.1,p)

    step!(lorenzds,5000)

    push!(points,current_state(lorenzds) + rand(4)*10^(-9))
end
lorenzds  = CoupledODEs(lorenz96_rule!,points[npoints+1],p)



updatetimes, radii, volumeratio, Δloc, points, corr_dim = regularlyvarying_parallel(lorenzds,points,npoints,maxtime)

endpoints = [length(Δloc[i]) for i in 1:npoints]
startradii = minimum([radii[i][1] for i in 1:npoints])
endradii = maximum([radii[i][endpoints[i]] for i in 1:npoints])
logmin = log(startradii) - 10^(-10)
logmax = log(endradii) + 10^(-10)
grid = exp.(range(logmin,logmax,100))


for i in 1:npoints
    nodes = (reverse(radii[i]),)
    itpratio = interpolate(nodes, reverse(volumeratio[i]),Gridded(Linear()))
    itpdim = interpolate(nodes, reverse(Δloc[i]),Gridded(Linear()))
    itpCorr = interpolate(nodes,reverse(corr_dim[i]),Gridded(Linear()))
    interpolatedratio[:,i] = itpratio(grid)
    interpolateddim[:,i] = itpdim(grid)
    interpolatedCorr[:,i] = itpGP(grid)
end



#### Saving data #########


dataratio = DataFrame(hcat(grid,interpolatedratio),:auto)
CSV.write("Lorenz96ManyPointsRatioTrial.csv", dataratio)

datadim = DataFrame(hcat(grid,interpolateddim),:auto)
CSV.write("Lorenz96ManyPointsDimTrial.csv", datadim)

dataCorr = DataFrame(hcat(grid,interpolatedCorr),:auto)
CSV.write("HenonHeilesManyPointsGPTrial.csv", dataCorr)



########### Read and plot ################


df1 = CSV.read("Lorenz96ManyPointsRatio.csv",DataFrame)
df2 = CSV.read("Lorenz96ManyPointsDim.csv",DataFrame)
df3 = CSV.read("Lorenz96ManyPointsCorr.csv",DataFrame)
grid = df1[:,1]
ratios = Array(df1[:,2:1001])
dims = Array(df2[:,2:1001])
corrs = Array(df3[:,2:1001])

plot_regular_variation(grid, ratios, dims, corrs, (0,5), "Lorenz 96")

save("Lorenz96ManyPoints.png", current_figure())
