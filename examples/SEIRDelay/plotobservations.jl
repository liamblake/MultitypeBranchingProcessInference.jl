using StatsPlots
using MultitypeBranchingProcessInference
using YAML
using LaTeXStrings

include("./utils/config.jl")
include("./utils/io.jl")
include("./utils/figs.jl")

function main(argv)
    if length(argv) != 1
        error("plotobservations.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(@__DIR__, argv[1]))
    model, _ = makemodel(config)

    path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))

    # Plot path through time
    p = scatter(t, hcat(path...)', xlabel="Day, t", ylabel="Number", labels=["E" "I" "O" "N" "IM"])

    figname = "simulation_path_$(argv[1])"
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")

    savefig(p, figname)

    # Daily observations and notifications
    counts = pathtodailycases(path, obs_state_idx(model.stateprocess) - 1)
    counts_vec = vcat(counts...)

    notifs = pathtodailycases(path, obs_state_idx(model.stateprocess))
    notifs_vec = vcat(notifs...)

    # Plot daily notifications
    p = scatter(t, notifs_vec, label="Notifications", xlabel="Day, t", ylabel="Number,  " * L"\,C_t^*", color=:black)
    scatter!(p, t, counts_vec, label="Counts (no delay)", color=:red)

    figname = "observations_$(argv[1])"
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")

    savefig(p, figname)
end

main(ARGS)