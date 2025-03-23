using StatsPlots
using MultitypeBranchingProcessInference
using YAML
using LaTeXStrings

include("utils/config.jl")
include("utils/io.jl")
include("utils/figs.jl")

function main(argv)
    if length(argv) != 1
        error("plotobservations.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(@__DIR__, argv[1]))
    model, ~ = makemodel(config)

    path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
    # only need daily cases
    cases = pathtodailycases(path, obs_state_idx(model.stateprocess))
    cases_vec = vcat(cases...)
    p = scatter(t, cases_vec, label="Daily new observed cases", xlabel="Day, t", ylabel="Number of cases,  " * L"\,C_t^*", color=:black)

    figname = "observations_$(argv[1])"
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")

    savefig(p, figname)
    return p
end

main(ARGS)