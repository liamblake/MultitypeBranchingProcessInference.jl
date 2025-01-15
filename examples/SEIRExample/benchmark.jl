using YAML
using Random
using StatsPlots
using MultitypeBranchingProcessInference
using LaTeXStrings
using BenchmarkTools
using LinearAlgebra

include("./utils/config.jl")
include("./utils/io.jl")
include("./utils/figs.jl")

const N_STD = 3

function makebenchmarks(config)
    model, param_seq = makemodel(config)

    path, t = readparticles(joinpath(pwd(), config["simulation"]["outfilename"]))
    cases = pathtodailycases(path, obs_state_idx(model.stateprocess))
    observations = Observations(t, cases)

    pf_rng = makerng(config["benchmarks"]["pfseed"])
    switch_rng = makerng(config["benchmarks"]["switchseed"])

    benchmarks = Dict()
    # pass nsamples as argument as the type cannot be inferred otherwise
    benchmark = (approx, nsamples::Int=config["benchmarks"]["nsamples"]) -> begin
        retval = Ref{Float64}()
        retvals = Float64[]
        bench::BenchmarkTools.Trial = @benchmark(
            $(retval)[] = logpdf!($model, $param_seq, $observations, $approx, $reset_obs_state_iter_setup!),
            teardown=(push!($retvals, $(retval)[])), 
            samples=nsamples, 
            evals=1,
        )::BenchmarkTools.Trial
        return (vals=retvals, times_ns=bench.times)
    end

    # define hf, pf and kf variables so we can use them after the loop(s)
    pf = nothing
    kf = nothing
    hf = nothing

    for nparticles in config["benchmarks"]["nparticles"]
        for threshold in config["benchmarks"]["thresholds"]
            # reseed rngs to ensure consistent results
            Random.seed!(pf_rng, config["benchmarks"]["pfseed"])
            Random.seed!(switch_rng, config["benchmarks"]["switchseed"])
            # define hf with the given threshold
            hf = HybridFilterApproximation(
                model, pf_rng, switch_rng, nparticles, threshold, getrandomstateidx(model.stateprocess),
            )
            # run benchmark
            benchmarks["filter=hybrid, nparticles=$nparticles, threshold=$threshold"] = benchmark(hf)
        end
        # reseed rngs to ensure consistent results
        Random.seed!(pf_rng, config["benchmarks"]["pfseed"])
        pf = hf.pfapprox
        benchmarks["filter=particle, nparticles=$nparticles"] = benchmark(pf)
    end
    kf = hf.kfapprox
    benchmarks["filter=kalman"] = benchmark(kf)
    return benchmarks
end

function bootvar(rng, vals, nboot)
    vars = [
        var(
            vals[rand(rng, eachindex(vals), length(vals))] # randomly sample vals
        ) # compute var of sample
        for _ in 1:nboot
    ] # vars is an empirical estimte of the distribution of var
    return var(vars) # compute the variance of the distribution of var
end

function bootci(rng, vals, nboot)
    vars = [
        var(
            vals[rand(rng, eachindex(vals), length(vals))] # randomly sample vals
        ) # compute var of sample
        for _ in 1:nboot
    ] # vars is an empirical estimte of the distribution of var
    return StatsPlots.quantile(vars, 0.005), StatsPlots.quantile(vars, 0.995)  
end

function benchmarkelipseparams(rng, benchmark)
    vals = benchmark.vals
    times_log2sec = log2.(benchmark.times_ns/1e9)

    # elipse center
    x = [var(vals); mean(times_log2sec)] 
    # construct variance(s)
    var_of_var = bootvar(rng, vals, length(vals))
    y = [var_of_var 0.0; 0.0 var(times_log2sec)]
    return x, y
end

function benchmarkciparams(rng, benchmark)
    vals = benchmark.vals
    times_log2sec = log2.(benchmark.times_ns/1e9)

    x = bootci(rng, vals, length(vals))
    x = (x[1], var(vals), x[2])
    y = (StatsPlots.quantile(times_log2sec, 0.005), StatsPlots.quantile(times_log2sec, 0.995))
    y = (y[1], mean(times_log2sec), y[2])
    return x, y
end

# p = maketimingplot(benches, config)
function addbenchmarks!(p, shapeparams, config, textpositions, type)
    plot_const_kwargs = (n_std=N_STD, alpha=0.5)
    textsize = 10
    texthalign = :left
    textvalign = :bottom

    colour_id = 0
    dolabel = true
    for threshold in config["benchmarks"]["thresholds"]
        colour_id += 1
        dolabel = true
        for nparticles in config["benchmarks"]["nparticles"]
            label = dolabel && (dolabel=false; "Hybrid (s=$threshold)")
            key = "filter=hybrid, nparticles=$nparticles, threshold=$threshold"
            if type==:covellipse
                x, y = shapeparams[key]
                covellipse!(p, x, y; color=cmap(colour_id), label=label,
                    plot_const_kwargs...
                )
            elseif type==:cisquare
                x, y = shapeparams[key]
                # xs = [x[1];x[1];x[2];x[2]]
                # ys = [y[1];y[2];y[2];y[1]]
                # shape = Shape(xs, ys)
                scatter!(p, [x[2]], [y[2]]; 
                    yerror=[(y[2]-y[1], y[3]-y[2])], xerror=[(x[2]-x[1], x[3]-x[2])], 
                    color=cmap(colour_id), label=label,
                    plot_const_kwargs...
                )
            end
            textpos = textpositions[key]
            annotate!(p, [textpos[1]], [textpos[2]], text(L"%$nparticles", textsize, valign=textvalign, halign=texthalign))
        end
    end
    colour_id += 1
    dolabel = true
    for nparticles in config["benchmarks"]["nparticles"]
        label = dolabel && (dolabel=false; "Particle")
        key = "filter=particle, nparticles=$nparticles"
        if type==:covellipse
            x, y = shapeparams[key]
            covellipse!(p, x, y; color=cmap(colour_id), label=label,
                plot_const_kwargs...
            )
        elseif type==:cisquare
            x, y = shapeparams[key]
            scatter!(p, [x[2]], [y[2]]; 
                yerror=[(y[2]-y[1], y[3]-y[2])], xerror=[(x[2]-x[1], x[3]-x[2])], 
                color=cmap(colour_id), label=label,
                plot_const_kwargs...
            )
        end
        textpos = textpositions[key]
        annotate!(p, [textpos[1]], [textpos[2]], text(L"%$nparticles", textsize, valign=textvalign, halign=texthalign))
    end
    return colour_id
end

function addformatting!(p)
    plot!(p, ylabel="Time (sec)", xlabel="Loglikelihood  estimate  variance")

    # log2 y-axis
    yl, yh = ylims(p)
    yticks_ = round(Int, yl, RoundUp):round(Int, yh, RoundDown)
    yticks!(p, yticks_, [L"2^{%$i}" for i in yticks_], tickfontsize=10)
    return 
end

function gettextpositions(shapeparams, shape)
    pos = Dict()
    for (key1, params) in shapeparams
        if shape==:covellipse
            x = copy(params[1])
            pos[key1] = x
        elseif shape==:cisquare
            x = params[1]
            y = params[2]
            pos[key1] = (x[2], y[2])
        end
    end
    return pos
end

function main(argv)
    if length(argv) != 1
        error("benchmark.jl program expects 1 argument \
               \n    - config file name.")
    end

    config = YAML.load_file(joinpath(pwd(), argv[1]))

    if "env" in keys(config) && "blas_num_threads" in keys(config["env"])
        LinearAlgebra.BLAS.set_num_threads(config["env"]["blas_num_threads"])
    end

    benches = makebenchmarks(config)

    rng = makerng(config["benchmarks"]["bootseed"])
    shapeparams = Dict(
        # key => benchmarkelipseparams(rng, benches[key]) 
        key => benchmarkciparams(rng, benches[key]) 
        for key in keys(benches) if key != "filter=kalman"
    )

    textpositions = gettextpositions(shapeparams, :cisquare)
    plot()
    colour_id = addbenchmarks!(plot!(), shapeparams, config, textpositions, :cisquare)
    
    colour_id += 1
    x = [var(benches["filter=kalman"].vals)]
    times_log2sec = log2.(benches["filter=kalman"].times_ns/1e9)
    y = [mean(times_log2sec)]
    yerr = [(
        only(y)-StatsPlots.quantile(times_log2sec, 0.005), 
        StatsPlots.quantile(times_log2sec, 0.995)-only(y)
    )]
    scatter!(plot!(), x, y; yerror=yerr,
        color=cmap(colour_id), label="Kalman")

    addformatting!(plot!())
    
    figname = "benchmarks-config_file_$(argv[1])"
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    figname = joinpath(pwd(), "figs", "$figname.$FIGURE_FILE_EXT")
    savefig(plot!(), figname)
    
    tmp = joinpath(pwd(), "gr-temp")
    println("Press any key to remove the temporary folder at $tmp (or press Ctrl-c to cancel).")
    readline(stdin)
    rm(tmp; force=true, recursive=true)
    return plot!()
end

main(ARGS)