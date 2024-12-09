"""
This script tests the intervals estimator for the extremal index (Maria Süveges, 2007) in a continuous
time setting. Since the estimator is intended for discrete systems, there is an ambiguity in how to 
correctly discretise the process to apply the algoritm. The code simulates an obsevable computed along
an orbit in the Lorenz 63 system and computes its extremal index for different values of the time step
and the trajectory lenght and saves the data. The last sections reads the data and estimates the 
continuous time that the process spends on average in a cluster through the extremal index.
"""

using CairoMakie
using DynamicalSystems
using StaticArrays
using Statistics
using ChaosTools
using Distances
using ProgressMeter
using  DataFrames, CSV, Tables, DrWatson

################### Same lenght, different time step, changing quantile ################

lorenzds = Systems.lorenz(rand(3);ρ=32.0)

total_time = 1000
samplings = 50
θ1 = Array{Any}(undef,samplings)
for (i,j) in enumerate(range(1,5,samplings))
    timestep = 0.1 * 2^(-j)
    X, time = trajectory(lorenzds, total_time, Δt = timestep)
    p = 1-1/√(length(X))
    ex = Exceedances(p,:mm)
    d, θ1[i] = extremevaltheory_dims_persistences(X,ex)
    println(i/samplings*100,"%")
end


result = DataFrame(Tables.table(θ1))
rename!(result,:Column1 => :θ)
CSV.write(savename("ExtremalIndexVSTimestepChangingQuantile",nothing,"csv"),result,bufsize = 9000000)


################### Same time step,different length, changing quantile ################


samplings = 50
timestep = 0.1 * 2^(-range(1,5,10)[4])
θ2 = Array{Any}(undef,samplings)
for (i,j) in enumerate(range(1,5,samplings))
    total_time = 100 * 2^(j)
    X, time = trajectory(lorenzds, total_time, Δt = timestep)
    p = 1-1/√(length(X))
    ex = Exceedances(p,:mm)
    d, θ2[i] = extremevaltheory_dims_persistences(X,ex)
    println(i/samplings*100,"%")
end



result = DataFrame(Tables.table(θ2))
rename!(result,:Column1 => :θ)
CSV.write(savename("ExtremalIndexVSTrajectoryLengthChangingQuantile",nothing,"csv"),result,bufsize = 9000000)

################### Same lenght, different time step, fixed quantile ################


total_time = 1000
samplings = 50
θ3 = Array{Any}(undef,samplings)
p = 0.99
for (i,j) in enumerate(range(1,5,samplings))
    timestep = 0.1 * 2^(-j)
    X, time = trajectory(lorenzds, total_time, Δt = timestep)
    ex = Exceedances(p,:mm)
    d, θ3[i] = extremevaltheory_dims_persistences(X,ex)
    println(i/samplings*100,"%")
end


result = DataFrame(Tables.table(θ3))
rename!(result,:Column1 => :θ)
CSV.write(savename("ExtremalIndexVSTimestepFixedQuantile",nothing,"csv"),result,bufsize = 9000000)



################### Same time step,different length, fixed quantile ################


samplings = 50
timestep = 0.1 * 2^(-range(1,5,10)[4])
θ4 = Array{Any}(undef,samplings)
p = 0.99
for (i,j) in enumerate(range(1,5,samplings))
    total_time = 100 * 2^(j)
    X, time = trajectory(lorenzds, total_time, Δt = timestep)
    ex = Exceedances(p,:mm)
    d, θ4[i] = extremevaltheory_dims_persistences(X,ex)
    println(i/samplings*100,"%")
end


result = DataFrame(Tables.table(θ4))
rename!(result,:Column1 => :θ)
CSV.write(savename("ExtremalIndexVSTrajectoryLengthFixedQuantile",nothing,"csv"),result,bufsize = 9000000)



############## Plotting #####################


fig = Figure(resolution = (600,1000))
lines(fig[1,1],0.1*2 .^ (-range(1,5,samplings)),[mean(θ1[i]) for i in 1:samplings],axis = (;xlabel = "Δt",ylabel = "Average θ", title= "θ vs. time step",
 xticklabelsize = 25, yticklabelsize = 25, xlabelsize = 25, ylabelsize = 25,titlesize = 25, yticks =  [0,0.25,0.50,0.75,1],limits = (nothing,(0,1))), label = "Changing quantile")

lines!(0.1*2 .^ (-range(1,5,samplings)),[mean(θ3[i]) for i in 1:samplings], label = "Fixed quantile")
axislegend(position = :rb,labelsize = 25)

lines(fig[2,1],100*2 .^ (range(1,5,samplings)),[mean(θ2[i]) for i in 1:samplings],axis = (;  xlabel = "tₗ",ylabel = "Average θ", title= "θ vs. trajectory length",
 xticklabelsize = 25, yticklabelsize = 25, xlabelsize = 25, ylabelsize = 25,titlesize = 25, yticks =  [0,0.25,0.50,0.75,1],limits = (nothing,(0,1))), label = "Changing quantile")

lines!(100*2 .^ (range(1,5,samplings)),[mean(θ4[i]) for i in 1:samplings], label = "Fixed quantile")
axislegend(position = :rb,labelsize = 25)

current_figure()

save("ExtremalIndexSensitivity.png",current_figure())


############ Reading the data ###########


df1 = CSV.read("ExtremalIndexVSTimestepChangingQuantile.csv",DataFrame)
df2 = CSV.read("ExtremalIndexVSTrajectoryLengthChangingQuantile.csv",DataFrame)
df3 = CSV.read("ExtremalIndexVSTimestepFixedQuantile.csv",DataFrame)
df4 = CSV.read("ExtremalIndexVSTrajectoryLengthFixedQuantile.csv",DataFrame)

samplings =50
θ1 = Array{Any}(undef,samplings)
θ2 = Array{Any}(undef,samplings)
θ3 = Array{Any}(undef,samplings)
θ4 = Array{Any}(undef,samplings)
for i in 1:50
    θ1[i] = filter(x->x<1,parse.(Float64, filter(x->!(x=="")&& !(x==" "),split(filter(∉(['[',']']),df1[i,:][1]), ','))))
    θ2[i] = filter(x->x<1,parse.(Float64, filter(x->!(x=="")&& !(x==" "),split(filter(∉(['[',']']),df2[i,:][1]), ','))))
    θ3[i] = filter(x->x<1,parse.(Float64, filter(x->!(x=="")&& !(x==" "),split(filter(∉(['[',']']),df3[i,:][1]), ','))))
    θ4[i] = filter(x->x<1,parse.(Float64, filter(x->!(x=="")&& !(x==" "),split(filter(∉(['[',']']),df4[i,:][1]), ','))))
end


######## Plotting the continuous time spent in cluster #############

fig = Figure(resolution = (600,1000))
lines(fig[1,1], 0.1*2 .^ (-range(1,5,samplings)), [mean(1 ./θ1[i])*( 0.1 * 2^(-range(1,5,samplings)[i])) for i in 1:samplings],axis = (;  xlabel = "Δt",ylabel = "Average t in cluster", title= "Time in cluster vs. time step",
 xticklabelsize = 25, yticklabelsize = 25, xlabelsize = 25, ylabelsize = 25,titlesize = 25 #=,yticks =  [0,0.25,0.50,0.75,1]=#,limits = (nothing,nothing)), label = "Changing quantile")

lines!(0.1*2 .^ (-range(1,5,samplings)),[mean(1 ./θ3[i])*( 0.1 * 2^(-range(1,5,samplings)[i])) for i in 1:samplings], label = "Fixed quantile")
axislegend(position = :rb,labelsize = 25)

lines(fig[2,1],100*2 .^ (range(1,5,samplings)),[mean(1 ./θ2[i])*( 0.1 * 2^(-range(1,5,10)[4])) for i in 1:samplings],axis = (;  xlabel = "tₗ",ylabel = "Average t in cluster", title= "Time in cluster vs. trajectory length",
 xticklabelsize = 25, yticklabelsize = 25, xlabelsize = 25, ylabelsize = 25,titlesize = 25#=, yticks =  [0,0.25,0.50,0.75,1]=#,limits = (nothing,nothing)), label = "Changing quantile")

lines!(100*2 .^ (range(1,5,samplings)),[mean(1 ./θ4[i])*( 0.1 * 2^(-range(1,5,10)[4])) for i in 1:samplings], label = "Fixed quantile")
axislegend(position = :rc,labelsize = 25)

current_figure()


save("ExtremalIndexSensitivityinTime.png",current_figure())
