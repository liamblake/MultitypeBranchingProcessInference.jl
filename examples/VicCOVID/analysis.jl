using MultitypeBranchingProcessInference
using MCMCChains
using StatsPlots
using YAML

function read_datasets(filenames, datasetnames, nburnin=0)
    datasets = Dict{String, Array{Float64,2}}()
    fileid = 1
    for filename in filenames
        datasetname = datasetnames[fileid]
        samples = open(joinpath(@__DIR__, filename), "r") do io
            MetropolisHastings.read_binary_array_file(io)
        end
        datasets[datasetname] = samples[:, (nburnin+1):end]
        fileid += 1
    end
    return datasets
end

datasetnames = ["VicCOVID"]
datafilenames = [
    joinpath("data", "vic_covid_kalman_param_samples.f64_array.bin")
]

config = YAML.load_file(joinpath(@__DIR__,"config.yaml"))
samples = read_datasets(datafilenames, datasetnames, config["inference"]["mh_config"]["nadapt"])

function datasetstochains(datasets, paramnames)
    chains = Dict{String, Chains}()
    for (datasetname, dataset) in datasets
        chain = Chains(dataset', paramnames)
        chains[datasetname] = chain
    end
    return chains
end

paramnames = [
    :T_E, :T_I, :R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10
]
chains = datasetstochains(samples, paramnames)
chains["VicCOVID"]

plot(chains["VicCOVID"])

R0idx = [:R_0_1, :R_0_2, :R_0_3, :R_0_4, :R_0_5, :R_0_6, :R_0_7, :R_0_8, :R_0_9, :R_0_10]
violin(chains["VicCOVID"][R0idx].value.data[:,:,1], ylims=(0, 4.5))