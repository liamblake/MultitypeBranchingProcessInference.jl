using StatsPlots
using MCMCChains
using YAML
using KernelDensity
using Distributions

using MultitypeBranchingProcessInference

include("./utils/config.jl")
include("./utils/figs.jl")

function read_datasets(filenames, datasetnames, applyrounding, nburnin=0)
    datasets = Dict{String, Array{Float64,2}}()
    fileid = 1
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        if applyrounding
            samples[end,:] .= round.(Int, samples[end,:])
        end
        datasets[datasetname] = samples[:, (nburnin+1):end]
        fileid += 1
    end
    return datasets
end

function datasetstochains(datasets, paramnames)
    chains = Dict{String, Chains}()
    for (datasetname, dataset) in datasets
        chain = Chains(dataset', paramnames)
        chains[datasetname] = chain
    end
    return chains
end

function maketraceplots(chains)
    plots = Dict{String, Any}()
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
    p = plot(xlabel="$(string(paramname))", ylabel="Density")
    chainid = 1
    for (datasetname, chain) in chains
        density!(p, chain[paramname]; label=datasetname, linestyle=smap(1), color=cmap(chainid))
        chainid += 1
    end
    yl, yh = ylims(p)
    chainid = 1
    for (datasetname, chain) in chains
        x = fill(mean(chain[paramname]), 2)
        y = [yl, yh]
        plot!(p, x, y; label=false, linestyle=smap(2), color=cmap(chainid))
        chainid += 1
    end
    if prior!==nothing
        plot!(p, x->Distributions.pdf(prior, x); label="Prior", color=cmap(chainid+1), linestyle=smap(3))
    end
    return p
end

function makechangepointpmf(chains, prior=nothing)
    paramname = :t_q
    p = plot(xlabel="$(string(paramname))", ylabel="Probability")
    chainid = 1
    for (datasetname, chain) in chains
        m = minimum(chain[paramname])
        M = maximum(chain[paramname])
        if (m%1) != 0 || (M%1) != 0
            error("t_q samples must be integers")
        end
        histogram!(p, chain[:t_q]; 
            bins=(m-0.5):1:(M+0.5), label=datasetname, color=cmap(chainid), alpha=0.5, normalize=:probability)
        chainid += 1
    end
    if prior!==nothing
        plot!(p, x->pdf(prior, x); label="Prior", color=cmap(chainid+1), linestyle=smap(3))
    end
    return p
end

function make2dposteriorpdf(chains, paramnames)
    p = plot()
    chainid = 1
    for (datasetname, chain) in chains
        chaindata = chain[paramnames].value.data
        densityestimate = kde(hcat([chaindata[:,:,i] for i in axes(chaindata, 3)]...))
        plot!(p, densityestimate; 
            color=cgrad(pmap(chainid)), levels=10, cbar=false)
        plot!(p, [NaN], [NaN]; color = cmap(chainid), label=datasetname)
        chainid += 1
    end
    return p
end

function main(argv)
    if length(argv)<2
        error("inference_analysis.jl program expects 2 or more arguments \
               \n    1. config file name.\
               \n    2... one or more strings of the form datasetname=filename where\
                        datasetname is a name to be used in plottinr and filename is a\
                        the name of a file containing samples.")
    end
    config = YAML.load_file(joinpath(pwd(), argv[1]))
    isinterventionmodel = "intervention" in keys(config["inference"]["prior_parameters"])

    dataset_metainfo = split.(argv[2:end], '=')
    dataset_names = [info[1] for info in dataset_metainfo] 
    dataset_filenames = [info[2] for info in dataset_metainfo] 
    
    nburnin = config["inference"]["mh_config"]["nadapt"]
    datasets = read_datasets(dataset_filenames, dataset_names, isinterventionmodel, nburnin)

    if isinterventionmodel
        paramnames = [:R_0; :T_E; :T_I; :q; :t_q]
    else 
        paramnames = [:R_0; :T_E; :T_I]
    end

    chains = datasetstochains(datasets, paramnames)
    
    traceplots = maketraceplots(chains)

    densityplots = Dict{Any, Any}()

    ctspriordists, discpriordists = makepriordists(config)

    paramid = 1
    for param in paramnames
        if param == :t_q
            continue
        end
        prior = ctspriordists[paramid]
        densityplots[param] = make1dposteriorpdf(chains, param, prior)
        paramid += 1
    end

    densityplots["R_0_vs_T_I"] = make2dposteriorpdf(chains, [:R_0, :T_I])

    if isinterventionmodel
        densityplots["R_0_vs_q"] = make2dposteriorpdf(chains, [:R_0, :q])
        prior = only(discpriordists)
        densityplots[:t_q] = makechangepointpmf(chains, prior)
    end

    caseidentifier = join(argv, "-")
    caseidentifier = replace(caseidentifier, 
        "." => "_", " " => "_", "/" => "_", "\\" => "_")
    for (name, plt) in traceplots
        figfilename = joinpath(pwd(), "figs", "$(caseidentifier)_traceplot_$(name).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    for (name, plt) in densityplots
        figfilename = joinpath(pwd(), "figs", "$(caseidentifier)_density_$(name).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end
    return 
end

main(ARGS)
