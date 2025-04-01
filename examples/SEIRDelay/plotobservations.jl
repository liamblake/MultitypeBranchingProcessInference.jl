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
	p = scatter(t, hcat(path...)', xlabel = "Day, t", ylabel = "Number", labels = ["E" "I" "O" "N" "IM"])

	figname = "simulation_path_$(argv[1])"
	figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
	figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")

	savefig(p, figname)

	# Daily notifications
	notifs = pathtodailycases(path, obs_state_idx(model.stateprocess))
	notifs_vec = vcat(notifs...)

	# Plot daily notifications
	p = scatter(t, notifs_vec, label = "Notifications", xlabel = "Day, t", ylabel = "Number,  " * L"\,C_t^*", color = :black)

	if config["inference"]["data"] == "simulated"
		# Plot R0
		tR = config["model"]["stateprocess"]["params"]["timestamps"]
		model_Rt = config["model"]["stateprocess"]["params"]["R_0"]
		plot!(twinx(), tR, model_Rt, label = L"R_0", ylabel = L"Simulated $R_t$", color = :red, seriestype = :steppost)
	end

	figname = "observations_$(argv[1])"
	figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
	figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")

	savefig(p, figname)
end

main(ARGS)
