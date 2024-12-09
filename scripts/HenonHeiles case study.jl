"""
This file computes and plots quantities related with the regular variation property for the Hénon-Heiles
system. The first section of the code iterates the Hénon-Heiles system and computes the quantities 
described in the src/regularly_varying.jl file for a number of points "npoints" randomly selected 
in the energy level in which the dynamics takes place. The next section interpolates the data obtained
in the first section to a common grid of radii, and the rest just save, read and plot the data.
"""


using DrWatson
@quickactivate "FractalDimensionsQ"

using CairoMakie
using ChaosTools, OrdinaryDiffEq
using Distances
using DynamicalSystemsBase
using ProgressMeter
using Interpolations
using FractalDimensions
using CSV
using DataFrames


include(srcdir("regularly_varying.jl"))


##################### Iterating the system #############################


diffeq = (alg = Vern9(), abstol = 1e-12, reltol = 1e-9)
function Henon_Heiles(du, u, p, t)
    x = u[1]
    y = u[2]
    dx = u[3]
    dy = u[4]
    du[1] = dx
    du[2] = dy
    du[3] = -x - 2x * y
    du[4] = y^2 - y - x^2
    return SVector(du[1],du[2],du[3],du[4])
end
p = []


npoints = 1000
maxtime = 10_000_000
nexceedences = 5000
interpolatedratio = zeros(100,npoints)
interpolateddim = zeros(100,npoints)
interpolatedCorr = zeros(100,npoints)


ref_points = []
point = [ -0.15560787798471826,-0.42517947002444845, 0.060690670151482946,-0.09990738249343888]
u0 = [0.0, -0.25, 0.42, 0.0]
hhds = CoupledODEs(Henon_Heiles,point,p)
for i in 1:npoints
    step!(hhds,1000)
    push!(ref_points,current_state(hhds) + rand(4)*10^(-10))
end
hhds  = CoupledODEs(Henon_Heiles,u0,p)

updatetimes, radii, volumeratio, Δloc, points, Corr_dim = regularlyvarying_parallel(hhds,ref_points,nexceedences,maxtime)
   
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
    itpCorr = interpolate(nodes,reverse(Corr_dim[i]),Gridded(Linear()))
    interpolatedratio[:,i] = itpratio(grid)
    interpolateddim[:,i] = itpdim(grid)
    interpolatedCorr[:,i] = itpCorr(grid)
end



#### Saving data #########


dataratio = DataFrame(hcat(grid,interpolatedratio),:auto)
CSV.write("HenonHeilesPointsRatio.csv", dataratio)

datadim = DataFrame(hcat(grid,interpolateddim),:auto)
CSV.write("HenonHeilesManyPointsDim.csv", datadim)

dataCorr = DataFrame(hcat(grid,interpolatedCorr),:auto)
CSV.write("HenonHeilesManyPointsCorr.csv", dataCorr)



################ Read and plot ################



df1 = CSV.read("HenonHeilesPointsRatio.csv",DataFrame)
df2 = CSV.read("HenonHeilesManyPointsDim.csv",DataFrame)
df3 = CSV.read("HenonHeilesManyPointsCorr.csv",DataFrame)
grid = df1[:,1]
interpolatedratio = Array(df1[:,2:1001])
interpolateddim = Array(df2[:,2:1001])
interpolatedCorr = Array(df3[:,2:1001])
plot_regular_variation(grid[1:85],interpolatedratio[1:85,:],interpolateddim[1:85,:],interpolatedCorr[1:85,:], (0.5,4),"Hénon-Heiles System")

wsave("HenonHeiles system.png",current_figure())

