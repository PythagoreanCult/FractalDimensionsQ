using Statistics, DynamicalSystemsBase, Optim, Roots, FractalDimensions


"""
    regularlyvarying_parallel(ds::DynamicalSystem, ref_points,nextremes,maxtime,enable = false)

Parellised version of the regularlyvarying function described below. This function allows to 
shorten the computation time when applying the function to many different points. 

"""

function regularlyvarying_parallel(ds::DynamicalSystem, ref_points,nextremes,maxtime,enable = false)
    dss = [deepcopy(ds) for _ in 1:Threads.nthreads()]

    updatetimes =  Array{Any}[]
    radii = Array{Any}[]
    volumeratio = Array{Any}[]
    Δloc = Array{Any}[]
    points = Array{Any}[]
    corr_dim = Array{Any}[]

    count = 0
    Threads.@threads for i in eachindex(ref_points)
        point = ref_points[i]
        ds = dss[Threads.threadid()]
        timesL, radiiL, volumeratioL, ΔlocL, pointsL,corr_dimL = regularlyvarying(ds, point, nextremes,maxtime,enable)
        push!(updatetimes,timesL)
        push!(radii,radiiL)
        push!(volumeratio,volumeratioL)
        push!(Δloc,ΔlocL)
        push!(points,pointsL)
        push!(corr_dim,corr_dimL)
        count +=1
        println(count*100/length(ref_points),"%")
    end

    return updatetimes, radii, volumeratio, Δloc, points, corr_dim
end



"""
    regularlyvarying(ds::DynamicalSystem, reference_point, nextremes::Int, max_time::Real, enabled = true)

Inputs:
ds: the dynamical system to be iterated
reference_point: the reference point for the recurrences
nextremes = points you want inside the ball.
maxtime = maximum time the system is allowed to evolve for when looking for recurrences of the
process and adding a closer extreme value to the vector of extremes.
enabled = optional boolean value to enable the progress visualization.

This function reinitiates the dynamical system and iterates it forward, recording the nextremes 
points which are the closest recurrences so far. Every time a point closer to reference is encountered,
the recurrence vector updates and the furthest away point is removed. When such an update happens,
the calculation of the local dimension through the Generalised Pareto Distribution (GPD) method and 
the correlation method is updated as well for the new exceedances vector.

Outputs:
times = time at which the recurrence took place
radii = all the distances from all the balls.
volumeratio = how many points lie in a ball of half the radious of the radius of the ball
Δloc = the local dimension value estimated through the Generalised Pareto Distribution method
points = the final points within the final radius (radii[end]).
corr_dim = an estimation of the correlation dimension
"""
function regularlyvarying(ds::DynamicalSystem, reference_point, nextremes::Int, max_time::Real, enabled = true)
    avgdist = average_step_distance(ds, 1000)
    extremedistances, maxdist, maxindex, indexes = initialize(ds, nextremes, reference_point)
    #@show extremedistances
    target = HSphere(maxdist,reference_point)
    progress = ProgressMeter.Progress(
        round(Int,max_time); desc = "Computing: ", enabled
    )
    updates = 0
    pointsize = size(reference_point)
    radii = Float64[]
    volumeratio = Float64[]
    times = Float64[]
    Δloc = Float64[]
    corr_dim = Float64[]
    points = zeros(nextremes, pointsize[1])
    nboxes = 16

    while current_time(ds) < max_time

        mindist, mintime, minpoint = step_until_recurrence!(ds,target,max_time, avgdist)
        if mintime > max_time
            break
        end
        updates += 1
        extremedistances[maxindex] = mindist
        points[maxindex, :] .= minpoint
        maxdist, maxindex = findmax(extremedistances)
        halfthreshold = maxdist/2
        smallball = count(x -> x < halfthreshold, extremedistances)
        push!(times, mintime)
        push!(radii, maxdist)
        push!(volumeratio, smallball/nextremes)
        push!(Δloc, localdimension(extremedistances, maxdist))
        push!(corr_dim, local_correlation_dimension(StateSpaceSet(points),reference_point, nboxes, fit = LargestLinearRegion()))
        target = HSphere(maxdist,reference_point)
        update!(progress,round(Int,current_time(ds)))
    end
    ProgressMeter.finish!(progress)
    return times, radii, volumeratio, Δloc, points, corr_dim
end




"""
    step_until_recurrence!(ds, args...)

This function takes the dynamical system in an arbitrary state and returns
the next closest recurrence to target.

Inputs:
ds: The dynamical system.
target: A HyperSphere centered in the reference point whose radius is the current maximum distance
of an extreme event, i.e. the effective threshold
max_time: maximum time that the algorithm is allowed to iterate
iteration: Keeps track of the indexes of the process, to compute the extremal index

Outputs:
dist: the next minimum distance between the orbit and the reference point observed
ds.t: the time at which that happens
iteration: the index of the process at which this happens (coincides with ds.t in this case)

"""
function step_until_recurrence!(ds, args...)
    if isdiscretetime(ds)
        step_until_recurrence_discrete!(ds, args...)
    else
        step_until_recurrence_continuous!(ds, args...)
    end
end

function step_until_recurrence_discrete!(ds, target, max_time, avgdist)
    # this just steps until the distance is less, if maxtime is less than the current time
    dist = Inf
    while current_time(ds) < max_time
        step!(ds)
        dist = euclidean(current_state(ds), target.center)
        #iteration += 1
        current_time(ds) ≥ max_time && return (Inf, Inf, current_state(ds))
        if target.radius > dist
            break
        end
    end
    return dist, current_time(ds), current_state(ds)
end

function step_until_recurrence_continuous!(ds, target, max_time, avgdist)
    truetime = current_time(ds)
    truedist = typeof(truetime)(Inf)
    truepoint = current_state(ds)

    while truedist > target.radius # iterate until closer than radius
        step!(ds)
        #iteration += 1
        original_dist = euclidean(current_state(ds), target.center)
        original_time = current_time(ds)
        original_point = copy(current_state(ds))

        # dist2 = euclidean(ds.integ(ds.integ.t), target.center)

        # @show dist, dist2
        # First check: if we are close enough to turn on interpolation.
        # The larger the multiplicative factor, the more "guaranteed" accurate we are.

        if original_dist < max(10avgdist, 10target.radius)

            # there are two things we need to:
            # the first is to get the interpolated/accurate minimum distance to target
            testtime, testdist, testpoint = closest_trajectory_point(ds.integ, target.center)
            # the second thing we need to do is to ensure that this is the "true"
            # minimum distance, that is, if I step one more time, I don't get smaller distance.
            truedist = testdist
            truetime = testtime
            truepoint = testpoint
            while testdist ≤ truedist
                step!(ds)
                current_time(ds) ≥ max_time && return (Inf, Inf, current_state(ds))

                testtime, testdist, testpoint = closest_trajectory_point(ds.integ, target.center)
                if testdist < truedist
                    truedist = testdist
                    truetime = testtime
                    truepoint = testpoint
                end
            end

            # @show truetime, truedist

            if original_dist < truedist
                truedist = original_dist
                truetime = original_time
                truepoint = original_point
            end


            # so, so far we have assigned to `truedist` the true minimum distance
            # from target. However, it is not guaranteed that this is also
            # less than our radius; that is why we have 2 nested while loops,
            # so that once both of these are done we have a guarantee.

        end # end of `if` close enough
    end # end of `while` max time
    return truedist, truetime, truepoint
end


function average_step_distance(ds, n::Int)
    u0 = copy(current_state(ds))
    d = 0.0
    for _ in 1:n
        step!(ds)
        d += euclidean(u0, current_state(ds))
        u0 = copy(current_state(ds))
    end
    return d/n
end


"""
    out_of_target!(ds,target,max_time)

This function takes the dynamical system in an arbitrary state and returns the next iteration in
which the orbit is away from the target ball

Inputs:
ds: The continuous time system
target: A HyperSphere centered in the reference point whose radius is the current maximum distance
of an extreme event, i.e. the effective threshold
max_time: maximum time that the algorithm is allowed to iterate
iteration: Keeps track of the indexes of the process, to compute the extremal index
"""
function out_of_target!(ds,target,max_time)
    dist = euclidean(current_state(ds),target.center)
    while target.radius > dist && ds.integ.t < max_time
        step!(ds)
        dist = euclidean(current_state(ds),target.center)
    end
end


"""
    closest_trajectory_point(integ, u0)

This function is a modified copy from the closest_trajectory_point function of ChaosTools,
in its accurate interpolation version, with the abs_tol and the rel_tol parameters hardcoded.
"""
function closest_trajectory_point(integ, u0)
    # use Optim.jl to find minimum of the function
    f = (t) -> euclidean(integ(t), u0)
    # Then find minimum of `f` in limits `(tprev, t)`
    optim = Optim.optimize(
        f, integ.tprev, integ.t, Optim.Brent();
        store_trace=false, abs_tol = 1e-12, rel_tol = 1e-12,
    )
    # You can show `Optim.iterations(optim)` to get an idea of convergence.
    #@show Optim.iterations(optim)
    tmin, dmin = Optim.minimizer(optim), Optim.minimum(optim)
    pointmin = copy(integ(tmin)) # unreliable, but why? Here is where the extrapolation takes place
    return tmin, dmin, pointmin
end

"""
    localdimension(extremedistances, maxdist)

This function estimates the local dimension through the Generalised Pareto Distribution 
method, assuming that the extreme distances are Exponentially distributed and computing
its average to obtain an unbiased estimate.

"""

function localdimension(extremedistances, maxdist)
    # Here we change this function to not unecessarily allocate
    # two new vectors (by doing `log.(x) .+ a`). We do a simple
    # iteration inside the `mean` function. Old code:
    # dim = 1/mean(-log.(extremedistances) .+ log(maxdist))
    # new code
    dim = 1/mean(-log(ed) + log(maxdist) for ed in extremedistances)
    if dim == 0.0
        #@show extremedistances
        error("Infinitely close recurrence, change reference_point")
    end
    return dim
end


"""
This function initializes the dynamical system to its initial point, and iterates it nextremes times,
    to obtain the first "extreme" distances that will be updated later in the run of the algorithm.
    Returns the vector with the extreme distances, the maximum distance, the index at which it is located
    within the vector, and the indexes vector describing the iteration at which the extreme event happens,
    for computing the extremal index.
"""
function initialize(ds::DynamicalSystem, nextremes::Int, reference_point)
    reinit!(ds)
    u0 = current_state(ds)
    maxdist = 0
    extremedistances = zeros(nextremes)
    indexes = zeros(nextremes)
    extremedistances[1] = euclidean(u0,reference_point)
    indexes[1] = 1
    for i in range(2, nextremes)
        step!(ds)
        extremedistances[i] = euclidean(current_state(ds), reference_point)
        indexes[i] = i
    end
    maxdist, maxindex = findmax(extremedistances)

    return extremedistances, maxdist, maxindex, indexes
end