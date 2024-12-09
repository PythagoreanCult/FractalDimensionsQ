"""
This file computes and plots quantities related with the regular variation property for a randomly
ordered sequence of points laying in a fat Cantor set. The set is defined by interating a map with 
a gap, the points that remain after many iterations approximate a invariant set. The first section
of the code constructs the invariant set to be studied and turns it into a data series. The second
iterates the data series to compute the quantities described in the src/regularly_varying.jl file 
for a point in the fat Cantor Set. The rest just save, read and plot the data.
"""

using DrWatson
@quickactivate "FractalDimensionsQ"

using CairoMakie
using ChaosTools, OrdinaryDiffEq
using Distances
using DynamicalSystemsBase
using ProgressMeter
using Interpolations
using Random
using FractalDimensions
using Distributions

include(srcdir("regularly_varying.jl"))
include(srcdir("theme.jl"))

########## Generating the set  ################


function  fat_fractal(u,p,n)
    r = 2*(1+1/2^(n+1)) 
    x = u[1]
    if x<1/2
        dx = x*r
    else
        dx = (-x+1)*r
    end
    SVector(dx)    
end

function inverseff1(u,p,n)
    r = 2*(1+1/2^(n+1)) 
    x = u[1]
    dx = x/r
    SVector(dx)        
end

function inverseff2(u,p,n)
    r = 2*(1+1/2^(n+1)) 
    x = u[1]
    dx = 1-x/r
    SVector(dx)       
end

p = [sqrt(2)]



statespace = rand(5)

maxiterations = 25

for i in 1:maxiterations
    statespace = fat_fractal.(statespace,p,i)
    statespace = filter(x -> 1>x[1]>0 ,statespace)
end


totlength = length(statespace)

statespace2 = []
for j in 1:maxiterations
    for i in 1:totlength
        push!(statespace2,inverseff1(statespace[i],p,maxiterations-j+1))
        push!(statespace2,inverseff2(statespace[i],p,maxiterations-j+1))
    end
    statespace = statespace2
    statespace2 = []
    totlength = length(statespace)
end

statespace = [i[1] for i in statespace]

totlength = length(statespace)
shuffledssset = shuffle(statespace)

############### Testing regular variation ########

function vectorrun(u,p,n)
    dx = p[n]
    SVector(dx)        
end
u0 =[0.1]
fat_fractalds = DeterministicIteratedMap(vectorrun,u0,shuffledssset, t0 = 1)

point = [0.1262477477965864]
maxtime = totlength
nevents = 5000
updatetimes, radii, volumeratio, Δloc, points, corr_dim = regularlyvarying(fat_fractalds,point,nevents,maxtime)

maxupdates = length(updatetimes)


############### Plotting ############


plot_regular_variation_one_line(radii, volumeratio, Δloc, corr_dim, (0,1.5), "Fat Cantor set")


save("Fat_fractal.png",current_figure())


result = DataFrame(Tables.table(hcat(updatetimes, radii, volumeratio, Δloc, vcat(points,zeros(maxupdates - nevents,1)),corr_dim)))
rename!(result,:Column1 => :Update_times, :Column2 => :Radii, :Column3 => :volumeratio, :Column4 => :Δloc, :Column5 => :Pointsx, :Column6 => :Corr_dim)
CSV.write(savename("Fat Cantor Set data",nothing,"csv"),result)