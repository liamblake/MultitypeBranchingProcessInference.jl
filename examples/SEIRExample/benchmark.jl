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

# p = maketimingplot(benches, config)
function addbenchmarks!(p, ellipses, config, textpositions)
    plot_const_kwargs = (n_std=N_STD, alpha=0.5)
    textsize = 10
    textalign = :center

    colour_id = 1
    dolabel = true
    for threshold in config["benchmarks"]["thresholds"]
        colour_id += 1
        dolabel = true
        for nparticles in config["benchmarks"]["nparticles"]
            label = dolabel && (dolabel=false; "Hybrid (s=%$threshold)")
            key = "filter=hybrid, nparticles=$nparticles, threshold=$threshold"
            x, y = ellipses[key]
            # covellipse!(p, x, y; color=cmap(colour_id), label=label,
            #     plot_const_kwargs...
            # )
            s1 = sqrt(y[1,1])
            s2 = sqrt(y[2,2])
            xs = [x[1] - N_STD*s1;
                x[1] - N_STD*s1;
                x[1] + N_STD*s1;
                x[1] + N_STD*s1]
            ys = [x[2] - N_STD*s2;
                x[2] + N_STD*s2;
                x[2] + N_STD*s2;
                x[2] - N_STD*s2]
            shape = Shape(xs, ys)
            plot!(p, shape; color=cmap(colour_id), label=label,
                plot_const_kwargs...
            )
            textpos = textpositions[key]
            annotate!(p, [textpos[1]], [textpos[2]], text("%$nparticles", textsize, valign=textalign))
        end
    end
    colour_id += 1
    dolabel = true
    for nparticles in config["benchmarks"]["nparticles"]
        label = dolabel && (dolabel=false; "Particle")
        key = "filter=particle, nparticles=$nparticles"
        x, y = ellipses[key]
        # covellipse!(p, x, y; color=cmap(colour_id), label=label,
        #     plot_const_kwargs...
        # )
        s1 = sqrt(y[1,1])
        s2 = sqrt(y[2,2])
        xs = [x[1] - N_STD*s1;
            x[1] - N_STD*s1;
            x[1] + N_STD*s1;
            x[1] + N_STD*s1]
        ys = [x[2] - N_STD*s2;
            x[2] + N_STD*s2;
            x[2] + N_STD*s2;
            x[2] - N_STD*s2]
        shape = Shape(xs, ys)
        plot!(p, shape; color=cmap(colour_id), label=label,
            plot_const_kwargs...
        )
        textpos = textpositions[key]
        annotate!(p, [textpos[1]], [textpos[2]], text("%$nparticles", textsize, valign=textalign))
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

function gettextpositions(ellipses, mindistances = (0.2, 0.05))
    pos = Dict()
    for (key1, value1) in ellipses
        x = copy(value1[1])
        y = copy(value1[2])
        # x[2] += sqrt(y[2,2])*(N_STD-0.2)
        pos[key1] = x
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
        key => benchmarkelipseparams(rng, benches[key]) 
        for key in keys(benches) if key != "filter=kalman"
    )

    textpositions = gettextpositions(shapeparams)
    plot()
    colour_id = addbenchmarks!(plot!(), shapeparams, config, textpositions)
    
    colour_id += 1
    x = fill(var(benches["filter=kalman"].vals), length(benches["filter=kalman"].vals))
    y = log2.(benches["filter=kalman"].times_ns/1e9)
    scatter!(plot!(), x, y; 
        color=cmap(colour_id), label="Kalman")

    addformatting!(plot!())
    
    figname = joinpath(pwd(), "figs", "benchmarks-config_file_$(argv[1])-state_idx_$(stateidx)-tstep_$(tstep).$FIGURE_FILE_EXT")
    figname = replace(figname, "." => "_", " " => "_", "/" => "_", "\\" => "_")
    return plot!()
end

main(ARGS)