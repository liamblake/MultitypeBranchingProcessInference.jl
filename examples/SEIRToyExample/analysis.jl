using StatsPlots
using YAML
using Distributions
using LaTeXStrings
using MultitypeBranchingProcessInference

default(; fontfamily="Bookman")

function readparticles(fn)
    particles = open(fn, "r") do io 
        nsteps = read(io, Int64)
        particles = Dict{Int64, Matrix{Int64}}()
        for step in 1:nsteps
            tstamp = read(io, Int64)
            n_particles = read(io, Int64)
            particle_length = read(io, Int64)
            step_particles = Matrix{Int64}(undef, n_particles, particle_length)
            for particleidx in 1:n_particles
                for eltidx in 1:particle_length
                    elt = read(io, Int64)
                    step_particles[particleidx, eltidx] = elt
                end
            end
            particles[tstamp] = step_particles
        end
        particles
    end
    return particles
end

params = YAML.load_file(joinpath(pwd(), ARGS[1]))

mtbp = StateSpaceModels.SEIR(
    1, 1,
    params["infection_rate"], params["exposed_stage_chage_rate"], params["infectious_stage_chage_rate"], 
    params["observation_probability"], 
    [params["E_immigration_rate"], params["I_immigration_rate"]],
    [params["initial_E"], params["initial_I"], params["initial_O"], 1],
)

particles = readparticles(joinpath(pwd(), ARGS[2]))

function makeplot(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)

    series = zeros(eltype(particles[0]), length(particles), mtbp.ntypes+1)
    sorted_keys = sort(collect(keys(particles)))
    plot()
    plot!([NaN], [NaN]; color=:grey, alpha=0.5, label = "Sample paths")
    for sampleidx in 1:size(particles[0], 1)
        count = 0
        for t in sorted_keys
            count += 1
            series[count, 1] = t
            series[count, 2:end] = particles[t][sampleidx,:]
        end
        plot!(series[:,2], series[:,3], alpha=0.1, color=:grey, label=false)
    end
    plot!(xlabel="Adolescents", ylabel="Adults")
    
    scattertimes = [20; 60; 100]
    colours = [:red, :blue, :green]
    i = 1
    for t in scattertimes
        scatter!(particles[t][:,1], particles[t][:,2], color=colours[i], label="Simulation t=$t", markersize=2)
        i += 1
    end
    for t in scattertimes
        moments!(moments, mtbp, t)
        mu = mean(moments, mtbp.state)
        sigma = variance_covariance(moments, mtbp.state)
        covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 1, color=:lightblue, label=nothing, alpha=0.4)
        covellipse!(mu[1:2], sigma[1:2,1:2]; n_std = 2, color=:lightblue, label=nothing, alpha=0.4)
    end

    mu_series = zeros(length(particles), mtbp.ntypes+1)
    count = 0
    for t in sorted_keys
        count += 1
        moments!(moments, mtbp, t)
        mu = zeros(paramtype(mtbp), getntypes(mtbp))
        mean!(mu, moments, mtbp.state)
        vcov = zeros(paramtype(mtbp), getntypes(mtbp), getntypes(mtbp))
        variance_covariance!(vcov, moments, mtbp.state)
        mu_series[count,1] = t
        mu_series[count, 2:end] = mu
        if t in scattertimes
            scatter!([mu_series[count, 2]], [mu_series[count, 3]], markersize=10, markershape=:+, color=:lightblue, label=nothing)
            annotate!([mu_series[count, 2]], [mu_series[count, 3]+sqrt(t)*6], "t=$t")
        end
    end
    plot!(mu_series[:,2], mu_series[:,3], color=:lightblue, label=L"E[z(t)]", linewidth=2)
    
    return plot!(grid=nothing)
end
samplepathplot = makeplot(particles, mtbp)
savefig(samplepathplot, joinpath("figs", "SEIRToyModelSamplePathPlot.pdf"))

function makeqq(particles, mtbp)
    moments = MTBPMomentsOperator(mtbp)
    init!(mtbp)
    plots = Plots.Plot[]
    for t in [20, 60, 100]
        StateSpaceModels.moments!(moments, mtbp, t)
        mu = zeros(paramtype(mtbp), getntypes(mtbp))
        mean!(mu, moments, mtbp.state)
        vcov = zeros(paramtype(mtbp), getntypes(mtbp), getntypes(mtbp))
        variance_covariance!(vcov, moments, mtbp.state)
        for i in 1:2
            p = qqplot(Normal(mu[i], sqrt(vcov[i,i])), particles[t][:,i], 
                ylabel=i==1 ? "t=$t\nSample Quantiles" : "", 
                xlabel=t==100 ? "Theoretical Quantiles" : "",
                title=t==20 ? (i==1 ? "Adolescents" : "Adults") : "",
                grid=nothing)
            push!(plots, p)
        end
    end
    plot(plots...; layout = (3, 2), size=(500,600))
end
qq = makeqq(particles, mtbp)
savefig(qq, joinpath("figs", "SEIRToyModelQQPlot.pdf"))

tmp = joinpath(pwd(), "gr-temp")
println("Press any key to delete temporary directory $(tmp) and contents.")
readline(stdin)
rm(tmp; force=true, recursive=true)