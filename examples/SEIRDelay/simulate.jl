using Random
using YAML
using MultitypeBranchingProcessInference

include("./utils/config.jl")
include("./utils/io.jl")

function main(args)
    if length(args) != 1
        error("simulate.jl program expects 1 argument \
               \n    - config file name.")
    end

    configfile = joinpath(pwd(), args[1])

    println("[INFO] Loading config file at $(configfile).")
    params = YAML.load_file(configfile)

    rng = makerng(params["simulation"]["seed"])

    model, param_seq = makemodel(params)
    mtbp = model.stateprocess

    println("MTBP prior to simulation:")
    println(mtbp)

    init!(rng, mtbp)

    tstep = params["simulation"]["tstep"]
    t = zero(tstep)

    writefilename = params["simulation"]["outfilename"]
    nsteps = params["simulation"]["nsteps"]

    println("[INFO] Simulating...")

    nextparamidx = firstindex(param_seq)
    nextparamtime = gettime(first(param_seq))
    iszero(nextparamtime) || error("First param timestamp in paramseq must be 0, got $nextparamtime")

    open(joinpath(pwd(), writefilename), "w") do io
        write(io, Int64(nsteps + 1))
        writeparticle(io, mtbp.state, t)
        for _ in 1:nsteps
            if t == nextparamtime
                params, nextparamidx = iterate(param_seq, nextparamidx)
                setparams!(model, params)
                nextparamtime = (nextparamidx > length(param_seq)) ? Inf : gettime(param_seq[nextparamidx])
            elseif t > nextparamtime
                error("Parameters at timestamp $nextparamtime. Parameter timestamp must equal an observation timestamp.")
            end

            t += tstep
            simulate!(rng, mtbp, tstep)
            writeparticle(io, mtbp.state, t)
        end
    end

    println("MTBP after to simulation:")
    println(mtbp)

    return 0
end

@time main(ARGS)