using MCMCChains
using StatsPlots
using YAML
using Random
using MultitypeBranchingProcessInference
using LaTeXStrings
using DelimitedFiles
using Distributions

include("./utils/config.jl")
include("./utils/io.jl")
include("./utils/figs.jl")

function makestateestimateplot(tstep, stateidx, config)
    model, param_seq = makemodel(config)

    path, t = readparticles(joinpath(@__DIR__, config["simulation"]["outfilename"]))
    cases = pathtodailycases(path, obs_state_idx(model.stateprocess))

    pf_rng = makerng(config["state_inference"]["pfseed"])
    nparticles = config["state_inference"]["nparticles"]
    switch_rng = makerng(config["state_inference"]["switchseed"])
    threshold = config["state_inference"]["threshold"]

    hf = HybridFilterApproximation(
        model, pf_rng, switch_rng, nparticles, threshold, getrandomstateidx(model.stateprocess),
    )

    pf = hf.pfapprox
    kf = hf.kfapprox

    observations = Observations(t[1:tstep], cases[1:tstep])
    
    # ensure seed of pf is the same at the start of every run
    Random.seed!(pf_rng, config["state_inference"]["pfseed"])
    
    # compute state estimates for pf (implicit in logpdf call)
    logpdf!(model, param_seq, observations, pf, reset_obs_state_iter_setup!)
    
    # get state estimates from pf and add to plot
    pfstatesims = [p[stateidx] for p in pf.store.store]
    xl, xr = minimum(pfstatesims)-0.5, maximum(pfstatesims)+0.5
    histogram(pfstatesims, bins=xl:xr, normalize=:pdf, label="Particle", alpha=0.3)

    # compute state estimates for kf (implicit in logpdf call)
    logpdf!(model, param_seq, observations, kf, reset_obs_state_iter_setup!)

    # get distribution parameters for state estimates and add density to plot
    mu = kf.kalmanfilter.state_estimate[stateidx]
    sigma = sqrt(kf.kalmanfilter.state_estimate_covariance[stateidx,stateidx])
    plot!(xl:0.05:xr, x -> pdf(Normal(mu, sigma), x), label="Gaussian")

    # ensure seed of pf is the same at the start of every run
    Random.seed!(pf_rng, config["state_inference"]["pfseed"])

    # compute state estimates for hf (implicit in logpdf call)
    logpdf!(model, param_seq, observations, hf, reset_obs_state_iter_setup!)
    
    # to get state estimates we need to determine which filter was used at the last iteration
    if hf.switchparams.currfiltername == :pfapprox
        statesims = [p[stateidx] for p in pf.store.store]
        histogram!(statesims, bins=xl:xr, normalize=:pdf, label="Hybrid (s=$threshold)", alpha=0.3)
    elseif hf.switchparams.currfiltername == :kfapprox
        mu = kf.kalmanfilter.state_estimate[stateidx]
        sigma = sqrt(kf.kalmanfilter.state_estimate_covariance[stateidx,stateidx])
        plot!(xl:0.05:xr, x -> pdf(Normal(mu, sigma), x), label="Hybrid (s=$threshold)")
    end

    # additional plot formatting
    yl, yh = ylims(plot!())
    # plot mean of pf
    plot!(mean(pfstatesims)*[1;1], [yl; yh], colour=:darkblue, linewidth=3, linestyle=:dash, label=false)
    # add annotation for time stamp
    annotate!([xl+0.05*(xr-xl)], [0.9*yh], L"t=%$(t[tstep])")
    # other formatting
    plot!(ylabel=L"Probability/Density, p(z_{%$(stateidx),t}|\mathbf{y}_{1:t})", xlabel=L"z_{%$(stateidx),t}", grid=:off)
    return plot!()
end

function main(argv)
    if length(argv)!=1
        error("state_inference.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(@__DIR__, argv[1]))

    for stateidx in config["state_inference"]["states"]
        for tstep in config["state_inference"]["tsteps"]
            p = makestateestimateplot(tstep, stateidx, config)
            figname = "state_inference-config_file_$(argv[1])-state_idx_$(stateidx)-tstep_$(tstep)"
            figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
            figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")
            savefig(p, figname)
        end
    end
end

main(ARGS)