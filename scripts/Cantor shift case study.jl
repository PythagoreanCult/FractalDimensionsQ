"""
This file computes and plots quantities related with the regular variation property for the one 
sided full shift of the Cantor set. The first section of the code iterates the Cantor shift map 
and computes the quantities described in the src/regularly_varying.jl file for a point in the 
Cantor Set. The rest just save, read and plot the data.
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

################### Iterating the process ################

sequence_lenght = 40
function Shift_map!(un,u, p, n) 
    deleteat!(u,1)
    push!(u,rand((0,2)))
    un = u
    return nothing
end

function parametrizationR(state)
    x = sum(state .* [1/3^i for i in 1:sequence_lenght])
    return SVector(x)
end
function fakeinverse(state)    
    return vec([rand((0,2)) for i in 1:sequence_lenght])
end

u0 = [rand((0,2)) for i in 1:sequence_lenght]
p0 = [0.1]
cantor_shift = DeterministicIteratedMap(Shift_map!, u0, p0)
cshift = ProjectedDynamicalSystem(cantor_shift,parametrizationR,fakeinverse)

point = [0.10804982490226529]
maxtime = 500_000_000
nevents = 5000


updatetimes, radii, volumeratio, Δloc, points, corr_dim = regularlyvarying(cshift,point,nevents,maxtime)

maxupdates = length(updatetimes)
endimension = round(Δloc[length(Δloc)],digits=3)
endCorr = round(corr_dim[length(corr_dim)],digits=3)

############ Saving the data

result = DataFrame(Tables.table(hcat(updatetimes, radii, volumeratio, Δloc, vcat(points,zeros(maxupdates - nevents,1)),corr_dim)))
rename!(result,:Column1 => :Update_times, :Column2 => :Radii, :Column3 => :volumeratio, :Column4 => :Δloc, :Column5 => :Pointsx, :Column6 => :Corr_dim)
CSV.write(savename("Cantor Shift data",nothing,"csv"),result)

####### Reading the data #########

cantordata = CSV.read("Cantor Shift data.csv",DataFrame)

######### Plotting #############

plot_regular_variation_one_line(cantordata[:,2],cantordata[:,3],cantordata[:,4], cantordata[:,6], (0.4,1), "Cantor Shift")
wsave("Cantor Shift.png",current_figure())