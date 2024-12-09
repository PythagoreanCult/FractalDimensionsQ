"""
This file computes and plots quantities related with the regular variation property for the Hénon map.
The first section of the code iterates the Hénon map and computes the quantities described in
the src/regularly_varying.jl file for a number of points "npoints" randomly selected in the attractor.
The next section interpolates the data obtained in the first section to a common grid of radii, and the
rest just save, read and plot the data.
"""

using DrWatson
@quickactivate "FractalDimensionsQ"

using ChaosTools, OrdinaryDiffEq
using Distances
using DynamicalSystemsBase
using ProgressMeter
using Interpolations
using DataFrames
using CSV
using FractalDimensions
using CairoMakie
using Tables
using MakieForProjects
using DelimitedFiles
using Statistics

include(srcdir("regularly_varying.jl"))
include(srcdir("theme.jl"))

################ Iterating the System #####################

henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])

u0 = [0.5726544706537069, 0.1372527622103796]
p0 = [1.4, 0.3]

henonds = DeterministicIteratedMap(henon_rule,u0,p0)
henonds2 = DeterministicIteratedMap(henon_rule,[0.09703255378598172, 0.16174999501050782],p0)



npoints = 1000
maxtime = 1_000_000_000
nextremes = 5000


X,t = trajectory(henonds2, npoints - 1)
points = zeros(2,npoints)
for i in 1:npoints
    henonds2 = DeterministicIteratedMap(henon_rule,rand(2)*0.1,p0)

    for i in range(1,100000)
        step!(henonds)
    end
    points[:,i] = current_state(henonds)
end


updatetimes =  Array{Any}[]
radii = Array{Any}[]
volumeratio = Array{Any}[]
Δloc = Array{Any}[]
points2 = Array{Any}[]
corr_dim = Array{Any}[]
interpolatedratio = zeros(100,npoints)
interpolateddim = zeros(100,npoints)
interpolatedCorr = zeros(100,npoints)

for i in 1:npoints
    henonds2 = DeterministicIteratedMap(henon_rule,rand(2)*0.1,p0)
    step!(henonds2,100000)
    point = current_state(henonds2)
    henonds = DeterministicIteratedMap(henon_rule,current_state(step!(henonds2,10000)),p0)

    timesL, radiiL, volumeratioL, ΔlocL, pointsL, corr_dimL, = regularlyvarying(henonds, point, nextremes,maxtime,false)
    push!(updatetimes,timesL)
    push!(radii,radiiL)
    push!(volumeratio,volumeratioL)
    push!(Δloc,ΔlocL)
    push!(points2,pointsL)
    push!(corr_dim,corr_dimL)
    println(i*100/npoints,"%")
end

############### Interpolating to a common grid ###################

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
    interpolatedCorr[:,i] = itpCorr(grid)
end

meanratio = vec(mean(interpolatedratio,dims = 2))
stdratio = vec(std(interpolatedratio,dims =2))

meandim = vec(mean(interpolateddim,dims = 2))
stddim = vec(std(interpolateddim,dims =2))

meanCorr = vec(mean(interpolatedCorr,dims = 2))
stdCorr = vec(std(interpolatedCorr,dims =2))


################# Writting the data files ####################

dataratio = DataFrame(hcat(grid,interpolatedratio),:auto)
CSV.write("HenonMapManyPointsRatio.csv", dataratio)

datadim = DataFrame(hcat(grid,interpolateddim),:auto)
CSV.write("HenonMapManyPointsDim.csv", datadim)

dataCorr = DataFrame(hcat(grid,interpolatedCorr),:auto)
CSV.write("HenonMapManyPointsCorr.csv", dataCorr)


#################### Plotting #######################

dataratio = DelimitedFiles.readdlm(projectdir("HenonMapManyPointsRatio.csv"), ','; skipstart = 1)
datadim = DelimitedFiles.readdlm(projectdir("HenonMapManyPointsDim.csv"), ','; skipstart = 1)
datacorr = DelimitedFiles.readdlm(projectdir("HenonMapManyPointsCorr.csv"), ','; skipstart = 1)
grid = dataratio[:, 1]
ratios = dataratio[:, 2:end]
dims = datadim[:, 2:end]
corrs = datacorr[:, 2:end]
titlename = "Hénon map"
dimylims = (0.8,2.5)

fig =plot_regular_variation(grid, ratios, dims, corrs,dimylims, titlename)


wsave(plotsdir("manypoints", "$(titlename).png"), current_figure())
