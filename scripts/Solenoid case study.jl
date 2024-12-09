"""
This file computes and plots quantities related with the regular variation property for the Solenoid 
map. The first section of the code iterates the Solenoid map and computes the quantities described in
the src/regularly_varying.jl file for a number of points "npoints" randomly selected in the attractor.
The next section interpolates the data obtained in the first section to a common grid of radii, and the
rest just save, read and plot the data. The last section is an approximation of the invariant measure of
the system.
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
using Combinatorics

include(srcdir("regularly_varying.jl"))
include(srcdir("theme.jl"))


function Solenoid_map(u, p, n) 
    ϕ, α, β = u 
    a = p[1] 
    ϕn = 2*ϕ % (2*π)
    αn = a*α + cos(ϕ)/2
    βn = a*β + sin(ϕ)/2
    return SVector(ϕn,αn,βn)
end

function parametrizationR3(state)
    ϕ, α, β = state
    x = (sqrt(α^2 + β^2)*cos(atan(β/α)) + 1)*cos(ϕ)
    y = (sqrt(α^2 + β^2)*cos(atan(β/α)) + 1)*sin(ϕ)
    z = β
    return SVector(x,y,z)
end


u0 = [2π*rand(), 0.3*rand(),0.3*rand()]


solds = DiscreteDynamicalSystem(Solenoid_map,u0,p0)
solenoid = ProjectedDynamicalSystem(solds,parametrizationR3,parametrizationR3)



npoints = 1000
maxtime = 600_000_000
nextremes = 5000

updatetimes =  Array{Any}[]
radii = Array{Any}[]
volumeratio = Array{Any}[]
Δloc = Array{Any}[]
points = Array{Any}[]
corr_dim = Array{Any}[]
p0 = [0.076]
for i in 1:npoints 
    solds1 = DeterministicIteratedMap(Solenoid_map,[2π*rand(), 0.3*rand(),0.3*rand()],p0)
    step!(solds1,10000)
    point = parametrizationR3(current_state(solds1))
    solds2 = DeterministicIteratedMap(Solenoid_map,current_state(step!(solds1,10000)),p0)
    solenoid2 = ProjectedDynamicalSystem(solds2,parametrizationR3,parametrizationR3)
    timesL, radiiL, volumeratioL, ΔlocL, pointsL,corr_dimL = regularlyvarying(solenoid2, point, nextremes,maxtime,false)
    push!(updatetimes,timesL)
    push!(radii,radiiL)
    push!(volumeratio,volumeratioL)
    push!(Δloc,ΔlocL)
    push!(points,pointsL)
    push!(corr_dim,corr_dimL)
    println(i/npoints*100,"%") 
end

############## Interpolation to a common grid #############

resolution = 100
interpolatedratio = zeros(resolution,npoints)
interpolateddim = zeros(resolution,npoints)
interpolatedCorr = zeros(resolution,npoints)

endpoints = [length(Δloc[i]) for i in 1:npoints]
startradii = minimum([radii[i][1] for i in 1:npoints])
endradii = maximum([radii[i][endpoints[i]] for i in 1:npoints])
logmin = log(startradii) - 10^(-10)
logmax = log(endradii) + 10^(-10)
grid = exp.(range(logmin,logmax,resolution))

for i in 1:npoints
    nodes = (reverse(radii[i]),)
    itpratio = interpolate(nodes, reverse(volumeratio[i]),Gridded(Linear()))
    itpdim = interpolate(nodes, reverse(Δloc[i]),Gridded(Linear()))
    itpCorr = interpolate(nodes,reverse(corr_dim[i]),Gridded(Linear()))
    interpolatedratio[:,i] = itpratio(grid)
    interpolateddim[:,i] = itpdim(grid)
    interpolatedCorr[:,i] = itpCorr(grid)
end


dataratio = DataFrame(hcat(grid,interpolatedratio),:auto)
CSV.write("SolenoidManyPointsRatio.csv", dataratio)

datadim = DataFrame(hcat(grid,interpolateddim),:auto)
CSV.write("SolenoidManyPointsDim.csv", datadim)

dataCorr = DataFrame(hcat(grid,interpolatedCorr),:auto)
CSV.write("SolenoidManyPointsCorr.csv", dataCorr)



############ Read and plot ######################


df1 = CSV.read("SolenoidManyPointsRatio.csv",DataFrame)
df2 = CSV.read("SolenoidManyPointsDim.csv",DataFrame)
df3 = CSV.read("SolenoidManyPointsCorr.csv",DataFrame)
grid = df1[:,1]
interpolatedratio = Array(df1[:,2:1001])
interpolateddim = Array(df2[:,2:1001])
interpolatedCorr = Array(df3[:,2:1001])
plot_regular_variation(grid,interpolatedratio,interpolateddim,interpolatedCorr, (0.8,1.8),"Solenoid Map")
wsave("Solenoid map.png",current_figure())





################ Measure estimation ##############

parameter = 0.076
depth = 17


vectors = Array{Any}(undef,depth)
angles = zeros(2^depth)
for h in 1:depth
    pool = vcat(zeros(h),ones(h))
    col = collect(multiset_permutations(pool,h))
    angles = zeros(2^h)
    for (i,j) in enumerate(col)
        angles[i] = sum([(π * j[k]/2^(h-k)) % (2*π) for k in 1:h])
    end
    vectors[h] = [[cos(angles[l]), sin(angles[l])] for l in 1:2^h]
end


centers = zeros(2^depth,2)
for i in 1:depth
    auxvec = vectors[i]
    grouplength = convert(Int64,2^depth/2^i)
    for k in 1:2^i
        for j in 1:grouplength
            centers[(k-1)*grouplength + j ,:] += auxvec[k] * (parameter^(i-1)/2) 
        end
    end
end


center = centers[rand(1:2^depth),:]

resolution = 100
radiusexp = range(1,16,resolution)
radius = 10 .^ (-radiusexp)
radius2 = radius/2
totlenght = zeros(resolution)
totlenght2 = zeros(resolution)
for i in 1:2^depth
    dist = euclidean(center,centers[i,:])
    for (h,r) in enumerate(radius)
        if dist < r
            totlenght[h] += sqrt(r^2 - dist^2) /(π*2^(depth))
            if dist < r/2
                totlenght2[h] += sqrt((r/2)^2 - dist^2) /(π*2^(depth))
            end
        end
    end
end

lines(radius,totlenght2./totlenght, axis = (; xscale = log10))
lines(log.(radius),log.(totlenght))
