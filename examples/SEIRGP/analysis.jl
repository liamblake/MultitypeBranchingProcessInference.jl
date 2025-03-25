using MultitypeBranchingProcessInference
using MCMCChains
using StatsPlots
using YAML
using LaTeXStrings
using KernelDensity
using PDMats
using Distributions
using Random
import Random.rand!

include("./utils/config.jl")
include("./utils/io.jl")
include("./utils/figs.jl")

function read_datasets(filenames, datasetnames, nburnin)
    datasets = Dict{String,Array{Float64,2}}()
    fileid = 1
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        datasets[datasetname] = samples[:, (nburnin+1):end]
        fileid += 1
    end
    return datasets
end

function datasetstochains(datasets, paramnames)
    chains = Dict{String,Chains}()
    for (datasetname, dataset) in datasets
        chain = Chains(dataset', paramnames)
        chains[datasetname] = chain
    end
    return chains
end

function main(argv)
    if length(argv) != 2
        error("analysis.jl program expects 2 or more arguments \
               \n    1. config file name.\
               \n    2. a string of the form datasetname=filename where\
                        datasetname is a name to be used in plottinr and filename is a\
                        the name of a file containing samples.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    datafilearg = argv[2]
    datafilename, datafilepath = split(datafilearg, "=")

    samples = read_datasets([datafilepath], [datafilename], config["inference"]["mh_config"]["nadapt"])

    model, _ = makemodel(config)

    # Direct Rt comparison
    # Grab the timestamps and R_0 path given in model config
    # This is the "truth" to compare R_t to
    model_timestamps = config["model"]["stateprocess"]["params"]["timestamps"]
    model_Rt = config["model"]["stateprocess"]["params"]["R_0"]

    # Get observations
    path, _ = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
    # only need daily cases
    observations = pathtodailycases(path, obs_state_idx(model.stateprocess))
    observations = vcat(observations...)
    println(observations)

    # Automatically find the R_0 indices from the config
    R0idx = map(x -> Symbol("R_0_$x"), 1:length(model_timestamps))
    println(R0idx)
    #[:R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10, :R_0_11, :R_0_12, :R_0_13, :R_0_14]
    paramnames = [:T_E; :T_I; R0idx; :LL]
    println(paramnames)

    chains_ = datasetstochains(samples, paramnames)
    summaryfilename = "$datafilepath-samples_file_$(argv[2])"
    summaryfilename = replace(summaryfilename, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    chainsummaryfilename = joinpath(pwd(), "data", summaryfilename)
    chainsummaryfilename = "$chainsummaryfilename.summary.txt"
    open(chainsummaryfilename, "w") do io
        display(TextDisplay(io), chains_[datafilename])
    end



    Rtplot = plot()
    plot!(Rtplot, model_timestamps, model_Rt, label="Simulation", color=:black)

    for datasetname in keys(chains_)
        errorline!(Rtplot, model_timestamps, Array(chains_[datasetname][R0idx]), errorstyle=:ribbon, label=datasetname, color=cmap(1))
    end
    plot!(Rtplot, xlabel=L"Day $t$", ylabel=L"$R_t$", legend=:topleft)
    savefig(Rtplot, joinpath(pwd(), "figs", "Rt_comparison.$FIGURE_FILE_EXT"))

    # diagnostic trace plot
    trace_plt = plot(chains_[datafilename])

    trace_plt_figname = "traceplot-config_file_$(argv[1])-samples_file_$(argv[2])" #-dataset_$(config["inference"]["data"]["filename"])"
    trace_plt_figname = replace(trace_plt_figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    trace_plt_figname = joinpath(pwd(), "figs", "$trace_plt_figname.$FIGURE_FILE_EXT")
    savefig(trace_plt, trace_plt_figname)

    # param inference plot
    # p = plot(size=(700, 400))
    # kdes = []
    # max_kde = -Inf
    # for i in eachindex(R0idx)
    #     kdeR0 = kde(chains_[datafilename][R0idx[i]][:])
    #     push!(kdes, [kdeR0.x, kdeR0.density])
    #     max_kde = max(max_kde, maximum(kdeR0.density))
    # end
    # for i in eachindex(R0idx)
    #     plot!(p, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1];
    #         ylims=(0, length(R0idx) / 2 - 0.5), color=:black, label=false)
    #     plot!(p, (i - 1) * 0.5 .+ kdes[i][2] ./ max_kde * 0.5, kdes[i][1];
    #         ylims=(0, length(R0idx) / 2 - 0.5), fill=true, color=:red, alpha=0.5, ylabel=L"R_0", side=:right,
    #         label=i == firstindex(R0idx) ? "Density" : false, legend=:topleft)
    # end
    # x = range(0, length(R0idx) / 2 + 1 / 7; length=101)
    # x = repeat(x, inner=2)
    # x = x[2:end-1]
    # y = repeat(observations, inner=2)
    # plot!(twinx(), x, y;
    #     color=:black, label="Daily cases", ylabel="Daily confirmed cases", legend=:topright)
    # ylims_ = ylims(p)
    # for x in 0.5:0.5:(length(R0idx)/2-0.5)
    #     plot!(p, [x; x], [ylims_[1]; ylims_[2]]; color=:grey, linestyle=:dash, label=false)
    # end
    # plot!(p, xlims=(-0.5 / 7, 7 + 1.5 / 7))
    # plot!(p, xticks=(0:0.5:7, ["$(7*(i-1))" for i in 1:length(R0idx)+1]), xlabel="Days")
    # plot!(p, grid=:off)

    # figname = "R_0_densities_and_cases-config_file_$(argv[1])-samples_file_$(argv[2])" #-dataset_$(config["inference"]["data"]["filename"])"
    # figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    # figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")
    # savefig(p, figname)

    return
end

main(ARGS)