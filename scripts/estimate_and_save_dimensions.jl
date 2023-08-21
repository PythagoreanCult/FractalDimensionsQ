using DrWatson
@quickactivate "FractalDimensionsQ"
using FractalDimensions, Statistics
include(srcdir("data_generation.jl"))

N = 10^4 # number of points
p = 0.99 # quantile

data = :henonmap
X = produce_data(; data, N)

εs = estimate_boxsizes(X; k = 32)

H0 = generalized_dim(X, εs; q = 0)
H1 = generalized_dim(X, εs; q = 1)
H2 = generalized_dim(X, εs; q = 2)
C = grassberger_proccacia_dim(X, εs)
Ei, θi = extremevaltheory_dims_persistences(X, 0.99)
E = mean(Ei)
θ = mean(θi)

out = @strdict data N H0 H1 H2 C E θ
wsave(datadir("mean_dimensions", "$(data).jld2"), out)
