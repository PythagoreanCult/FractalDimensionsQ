"""
This file contains plotting functions for the plots regarding the regular variation properties
of the dynamical systems studied.
"""


using MakieForProjects
using Statistics
using CairoMakie



function plot_regular_variation(grid, ratios, dims, corrs, dimylims, titlename)
    fig, axs = axesgrid(3, 1;
        xlabels = "radius", sharex = true,
        ylabels = [L"R(r)", L"\Delta_\mathrm{GPD}", L"\Delta_\mathrm{Cor}"],
        size = (800, 600), backgroundcolor = :white,
    )

    for (i, quantity) in enumerate((ratios, dims, corrs))
        ax = axs[i]
        m = vec(mean(quantity; dims = 2))
        s = vec(std(quantity; dims = 2))
        for j in 1:max(size(quantity, 2), 100)
            lines!(ax, grid, quantity[:,j]; linewidth = 1, color = (:black, 0.03))
        end
        band!(ax, grid, m .- s, m .+ s; color = (:blue, 0.25))
        lines!(ax, grid, m .- s; color = :blue, linewidth = 1)
        lines!(ax, grid, m .+ s; color = :blue, linewidth = 1)
        lines!(ax, grid, m; color = :red, linewidth = 2)
        ax.xscale = log10
        if i > 1
            estimate = round(m[end]; sigdigits = 4)
            textbox!(ax,  "estimate = $(estimate)", halign = :left)
            ylims!(ax, dimylims)
            xlims!(ax, extrema(grid))
        end
    end
    figuretitle!(fig, titlename)
    return display(fig)
end


function plot_regular_variation_one_line(grid, ratios, dims, corrs, dimylims, titlename)
    fig, axs = axesgrid(3, 1;
        xlabels = "radius", sharex = true,
        ylabels = [L"R(r)", L"\Delta_\mathrm{GPD}", L"\Delta_\mathrm{Cor}"],
        size = (800, 600), backgroundcolor = :white,
    )
    for (i, quantity) in enumerate((ratios, dims, corrs))
        ax = axs[i]
        lines!(ax, grid, quantity; color = (Makie.wong_colors()[1],1))
        ax.xscale = log10
        if i > 1
            estimate = round(quantity[end]; sigdigits = 4)
            textbox!(ax,  "estimate = $(estimate)", halign = :left)
            ylims!(ax, dimylims)
            xlims!(ax, extrema(grid))
        end
    end
    figuretitle!(fig, titlename)
    return display(fig)
end