using StatsPlots
using MCMCChains
using YAML
using KernelDensity
using Distributions
using LaTeXStrings
using MultitypeBranchingProcessInference

include("./utils/config.jl")
include("./utils/figs.jl")
include("./utils/io.jl")

function read_datasets(filenames, datasetnames, applyrounding, nburnin, paramnames)
    datasets = Dict{String,Array{Float64,2}}()
    fileid = 1
    chains = Dict{String,Chains}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        if applyrounding
            samples[end, :] .= round.(Int, samples[end, :])
        end
        dataset = samples[:, (nburnin+1):end]
        chain = Chains(dataset', paramnames)
        open(joinpath(pwd(), "$(filename).summary.txt"), "w") do io
            display(TextDisplay(io), chain)
        end
        chains[datasetname] = chain
        fileid += 1
    end
    return chains
end

function maketraceplots(chains)
    plots = Dict{String,Any}()
    p = nothing
    chainid = 1
    for (datasetname, chain) in chains
        if chainid == 1
            p = plot(chain; color=cmap(chainid), linestyle=smap(1))
        else
            plot!(p, chain; color=cmap(chainid), linestyle=smap(1))
        end
        q = plot(chain; color=cmap(chainid), linestyle=smap(1))
        plots[datasetname] = q
        chainid += 1
    end
    if "all" in keys(chains)
        error("Key clash. A samples file cannot have the name all")
    end
    plots["all"] = p
    return plots
end

function make1dposteriorpdf(chains, paramname, prior=nothing)
    p = plot(xlabel=L"%$(string(paramname))", ylabel="Density")
    chainid = 1
    for (datasetname, chain) in chains
        density!(p, chain[paramname]; label=datasetname, linestyle=smap(1), color=cmap(chainid), linewidth=2)
        chainid += 1
    end
    yl, yh = ylims(p)
    chainid = 1
    for (datasetname, chain) in chains
        x = fill(mean(chain[paramname]), 2)
        y = [yl, yh]
        plot!(p, x, y; label=false, linestyle=smap(2), color=cmap(chainid), linewidth=2)
        chainid += 1
    end
    if prior !== nothing
        plot!(p, x -> Distributions.pdf(prior, x); label="Prior", color=cmap(chainid + 1), linestyle=smap(3), linewidth=2)
    end
    return p
end

function main(argv)
    if length(argv) < 2
        error("analysis.jl program expects 2 or more arguments \
         \n    1. config file name.\
         \n    2... one or more strings of the form datasetname=filename where\
         datasetname is a name to be used in plottinr and filename is a\
         the name of a file containing samples.")
    end
    config = YAML.load_file(joinpath(pwd(), argv[1]))

    dataset_metainfo = split.(argv[2:end], '=')
    dataset_names = [info[1] for info in dataset_metainfo]
    dataset_filenames = [info[2] for info in dataset_metainfo]

    nburnin = config["inference"]["mh_config"]["nadapt"]

    # Grab the timestamps and R_0 path given in model config
    # This is the "truth" to compare R_t to if data: simulated
    model_timestamps = config["model"]["stateprocess"]["params"]["timestamps"]
    model_Rt = config["model"]["stateprocess"]["params"]["R_0"]

    R0idx = map(x -> Symbol("R_0_$x"), 1:length(model_timestamps))
    paramnames = [R0idx; :LL] #[Symbol(param) for param in config["inference"]["parameters"]]
    chains = read_datasets(dataset_filenames, dataset_names, true, nburnin, paramnames)

    traceplots = maketraceplots(chains)

    caseidentifier = "config_$(argv[1])_$(join(keys(chains), "-"))"
    caseidentifier = replace(caseidentifier,
        "." => "_", " " => "_", "/" => "_", "\\" => "_")
    for (name, plt) in traceplots
        figfilename = joinpath(pwd(), "figs", "traceplot_$(name)_$(caseidentifier).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    # Direct Rt comparison

    model, _ = makemodel(config)

    if config["inference"]["data"] == "simulated"
        # Get observations
        path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
        # only need daily cases
        observations = pathtodailycases(path, obs_state_idx(model.stateprocess))
        observations = vcat(observations...)
    else
        # Otherwise, assume "filename" and "first_observation_time" keys
        raw_observations = read_observations(joinpath(pwd(), config["inference"]["data"]["filename"]))
        t = config["inference"]["data"]["first_observation_time"] .+ (0:(length(raw_observations)-1))
        observations = Observations(t, raw_observations)
    end

    pR0 = plot(xlabel=L"Days $t$", ylabel=L"R_0")

    # Plot the observations
    # bar!(pR0, t, observations, label="Data", color=:black)

    for (datasetname, chain) in chains
        # errorline!(pR0, model_timestamps, Array(chain[R0idx]), errorstyle=:ribbon, label=datasetname, seriestype=:steppost)

        kdes = []
        max_kde = -Inf

        # Fit a KDE to each set of samples
        for i in eachindex(R0idx)
            kdeR0 = kde(chain[R0idx[i]][:])
            push!(kdes, [kdeR0.x, kdeR0.density])
            max_kde = max(max_kde, maximum(kdeR0.density))
        end

        # Plot KDEs
        for i in eachindex(R0idx)
            plot!(pR0, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1]; ylims=(0, length(R0idx) / 2 - 0.5), color=:black, label=false)
            plot!(pR0, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1];
                ylims=(0, length(R0idx) / 2 - 0.5), fill=true, color=:red, alpha=0.5, ylabel=L"R_0", side=:right,
                label=i == firstindex(R0idx) ? "Density" : false, legend=:topleft)
        end

    end

    if config["inference"]["data"] == "simulated"
        # Plot the simulated R_0
        plot!(pR0, model_timestamps, model_Rt, label="Simulated", color=:black, seriestype=:steppost)
    end

    savefig(pR0, joinpath(pwd(), "figs", "R0_comparison.$(FIGURE_FILE_EXT)"))

    # param inference plot
    # for (datafilename, chain) in chains
    #     p = plot(size=(700, 400))

    #     kdes = []
    #     max_kde = -Inf

    #     # Fit a KDE to each set of samples
    #     for i in eachindex(R0idx)
    #         kdeR0 = kde(chain[R0idx[i]][:])
    #         push!(kdes, [kdeR0.x, kdeR0.density])
    #         max_kde = max(max_kde, maximum(kdeR0.density))
    #     end

    #     # Plot KDEs
    #     for i in eachindex(R0idx)
    #         plot!(p, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1]; ylims=(0, length(R0idx) / 2 - 0.5), color=:black, label=false)
    #         plot!(p, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1];
    #             ylims=(0, length(R0idx) / 2 - 0.5), fill=true, color=:red, alpha=0.5, ylabel=L"R_0", side=:right,
    #             label=i == firstindex(R0idx) ? "Density" : false, legend=:topleft)
    #     end

    #     # Add cases for reference
    #     x = range(0, length(R0idx) / 2 + 1 / 7; length=length(t))
    #     x = repeat(x, inner=2)
    #     y = repeat(observations, inner=2)

    #     plot!(twinx(), x, y;
    #         color=:black, label="Daily cases", ylabel="Daily confirmed cases", legend=:topright)
    #     ylims_ = ylims(p)
    #     for x in 0.5:0.5:(length(R0idx)/2-0.5)
    #         plot!(p, [x; x], [ylims_[1]; ylims_[2]]; color=:grey, linestyle=:dash, label=false)
    #     end

    #     # plot!(p, xlims=(-0.5 / 7, 7 + 1.5 / 7))
    #     plot!(p, xticks=(0:0.5:7, ["$(7*(i-1))" for i in 1:length(R0idx)+1]), xlabel="Days")
    #     plot!(p, grid=:off)

    #     figname = "R_0_densities_and_cases-config_file_$(argv[1])-samples_file_$(argv[2])-$(datafilename)"
    #     figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    #     figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")
    #     savefig(p, figname)
    # end

    return
end

main(ARGS)
