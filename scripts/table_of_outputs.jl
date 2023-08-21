using DrWatson
@quickactivate "FractalDimensionsQ"
using DataFrames

res = collect_results(datadir("mean_dimensions")) # load all data
res = res[!, Not(:path)] # ignore `path` column