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
    datasets = Dict{String, Array{Float64,2}}()
    fileid = 1
    chains = Dict{String, Chains}()
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(pwd(), filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        if applyrounding
            samples[end,:] .= round.(Int, samples[end,:])
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
    if prior!==nothing
        plot!(p, x->Distributions.pdf(prior, x); label="Prior", color=cmap(chainid+1), linestyle=smap(3), linewidth=2)
    end
    return p
end

function makechangepointpmf(chains, prior, cases)
    paramname = :t_q
    p = plot(xlabel=L"t_q", ylabel="Probability")
    chainid = 1
    for (datasetname, chain) in chains
        m = minimum(chain[paramname])
        M = maximum(chain[paramname])
        if (m%1) != 0 || (M%1) != 0
            error("t_q samples must be integers")
        end
        histogram!(p, chain[:t_q]; 
            bins=(m-0.5):1:(M+0.5), label=datasetname, color=cmap(chainid), alpha=0.2, normalize=:probability, legend=:topleft)
        chainid += 1
    end
    x = 0:length(cases)-1
    y = cases
    scatter!(twinx(p), x, y; label="Daily cases", ylabel="Daily cases", legend=:topright,
        color=:black, linestyle=:solid)
    if prior!==nothing
        x = 1:29
        plot!(p, x, x->pdf(prior, x); label="Prior", color=cmap(chainid+1), linestyle=smap(3))
    end
    return p
end

function make2dposteriorpdf(chains, paramnames)
    p = plot()
    chainid = 1
    for (datasetname, chain) in chains
        chaindata = chain[paramnames].value.data
        data_matrix = vcat([chaindata[:,:,i] for i in axes(chaindata, 3)]...)
        # if chainid==1
        #     subsetidx = 1:40:size(data_matrix, 1)
        #     scatter!(p, data_matrix[subsetidx,1], data_matrix[subsetidx,2]; color = cmap(chainid), label=datasetname, alpha=0.3)
        # else
            # modify Silvermans rule to get smooth kde contours 
            bw1 = 1.3*sqrt(var(data_matrix[:,1]))*size(data_matrix,1)^-0.2
            bw2 = 1.3*sqrt(var(data_matrix[:,2]))*size(data_matrix,1)^-0.2
            densityestimate = kde(data_matrix; bandwidth=(bw1, bw2))
            plot!(p, densityestimate; 
                color=cgrad(pmap(chainid)), levels=8, cbar=false, linewidth=2)
            plot!(p, [NaN], [NaN]; color = cmap(chainid), label=datasetname, linewidth=2)
        # end
        chainid += 1
    end
    plot!(xlabel=L"%$(paramnames[1])", ylabel=L"%$(paramnames[2])")
    return p
end

function main(argv)
    if length(argv)<2
        error("analysis.jl program expects 2 or more arguments \
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
    if isinterventionmodel
        paramnames = [:R_0; :T_E; :T_I; :q; :t_q]
    else 
        paramnames = [:R_0; :T_E; :T_I]
    end
    chains = read_datasets(dataset_filenames, dataset_names, isinterventionmodel, nburnin, paramnames)
    
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

    for i in eachindex(dataset_names)
        for j in Iterators.drop(eachindex(dataset_names), i)
            keyi = dataset_names[i]
            keyj = dataset_names[j]
            chainpair = Dict(
                keyi => chains[keyi],
                keyj => chains[keyj],
            )
            densityplots["R_0_vs_T_I_$(keyi)_$(keyj)"] = make2dposteriorpdf(chainpair, [:R_0, :T_I])
        end
    end

    if isinterventionmodel
        for i in eachindex(dataset_names)
            for j in Iterators.drop(eachindex(dataset_names), i)
                keyi = dataset_names[i]
                keyj = dataset_names[j]
                chainpair = Dict(
                    keyi => chains[keyi],
                    keyj => chains[keyj],
                )
            densityplots["R_0_vs_q_$(keyi)_$(keyj)"] = make2dposteriorpdf(chainpair, [:R_0, :q])
            end
        end
        prior = only(discpriordists)
        
        path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
        # only need daily cases
        cases = pathtodailycases(path, 5)
        cases = [only(c) for c in cases]
        densityplots[:t_q] = makechangepointpmf(chains, prior, cases)
    end

    caseidentifier = "config_$(argv[1])_$(join(keys(chains), "-"))"
    caseidentifier = replace(caseidentifier, 
        "." => "_", " " => "_", "/" => "_", "\\" => "_")
    for (name, plt) in traceplots
        figfilename = joinpath(pwd(), "figs", "traceplot_$(name)_$(caseidentifier).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    for (name, plt) in densityplots
        figfilename = joinpath(pwd(), "figs", "density_$(name)_$(caseidentifier).$(FIGURE_FILE_EXT)")
        savefig(plt, figfilename)
    end

    tmp = joinpath(pwd(), "gr-temp")
    println("Press any key to remove the temporary folder at $tmp (or press Ctrl-c to cancel).")
    readline(stdin)
    rm(tmp; force=true, recursive=true)
    return 
end

main(ARGS)
