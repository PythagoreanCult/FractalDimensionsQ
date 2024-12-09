"""
This file computes and plots quantities related with the regular variation property for the Lorenz 63
system. The first section of the code iterates the Lorenz 63 system and computes the quantities described in
the src/regularly_varying.jl file for a number of points "npoints" randomly selected in the attractor.
The next section interpolates the data obtained in the first section to a common grid of radii, and the
rest just save, read and plot the data.
"""

using DrWatson
@quickactivate "FractalDimensionsQ"

using ChaosTools
using OrdinaryDiffEq
using Distances
using DynamicalSystemsBase
using ProgressMeter
using Interpolations
using DataFrames
using Statistics
using CSV
using FractalDimensions

include(srcdir("regularly_varying.jl"))
include(srcdir("theme.jl"))


############## Iterating the system #####################



function lorenzsystem!(du,u, p, n) 
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end


p0 = [10.0, 28.0, 8/3]



npoints = 100
maxtime = 400_000#_000
nexceedences = 5000#0
interpolatedratio = zeros(100,npoints)
interpolateddim = zeros(100,npoints)
interpolatedextremalindex = zeros(100,npoints)


points = []
for i in 1:npoints+1
    lorenzds = CoupledODEs(lorenzsystem!,rand(3)*0.1,p0)

    step!(lorenzds,5000)

    push!(points,current_state(lorenzds) + rand(3)*10^(-9))
end
lorenzds  = CoupledODEs(lorenzsystem!,points[npoints+1],p0)




updatetimes, radii, volumeratio, Δloc, points, corr_dim = regularlyvarying_parallel(lorenzds,points[1:npoints],npoints,maxtime)
    

endpoints = [length(radii[i]) for i in 1:npoints]
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
    interpolatedCorr[:,i] = itpCorr(grid)
end


#### Saving data #########


dataratio = DataFrame(hcat(grid,interpolatedratio),:auto)
CSV.write("LorenzManyPointsRatioTrial.csv", dataratio)

datadim = DataFrame(hcat(grid,interpolateddim),:auto)
CSV.write("LorenzManyPointsDimTrial.csv", datadim)

dataCorr = DataFrame(hcat(grid,interpolatedGP),:auto)
CSV.write("LorenzManyPointsCorr.csv", dataGP)



################## Read and plot ###############


df1 = CSV.read("LorenzManyPointsRatio.csv",DataFrame)
df2 = CSV.read("LorenzManyPointsDim.csv",DataFrame)
df3 = CSV.read("LorenzManyPointsCorr.csv",DataFrame)
grid = df1[:,1]
ratios = Array(df1[:,2:1001])
dims = Array(df2[:,2:1001])
corrs = Array(df3[:,2:1001])

plot_regular_variation(grid, ratios, dims, corrs, (0.5,2), "Lorenz 63")

wsave("Lorenz 63.png",current_figure())