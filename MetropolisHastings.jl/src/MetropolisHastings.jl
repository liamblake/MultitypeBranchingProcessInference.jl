module MetropolisHastings

using Random
using Distributions
using LinearAlgebra

import Random.rand
import Distributions.logpdf

include("io.jl")
include("proposal.jl")
include("buffer.jl")

export MHConfig,
    metropolis_hastings,
    SymmetricProposalDistribution,
    MutableMvNormal,
    read_binary_array_file

struct MHConfig{F<:Real}
    samples_write_buffer_size::Int
    samples_write_file::String
    maxiters::Int
    nparams::Int
    max_time_sec::Float64
    init_sample::Vector{F}
    verbose::Bool
    info_file::String
    adaptive::Bool
    nadapt::Int
    adapt_cov_scale::F
    continue_from_write_file::Bool
end

function printconfig(io, mh_config)
    println(io, "[INFO]     MH Config:")
    println(io, "[INFO]         samples_write_buffer_size: ", mh_config.samples_write_buffer_size)
    println(io, "[INFO]         samples_write_file: ", mh_config.samples_write_file)
    println(io, "[INFO]         maxiters: ", mh_config.maxiters)
    println(io, "[INFO]         nparams: ", mh_config.nparams)
    println(io, "[INFO]         max_time_sec: ", mh_config.max_time_sec)
    println(io, "[INFO]         init_sample: ", mh_config.init_sample)
    println(io, "[INFO]         verbose: ", mh_config.verbose)
    println(io, "[INFO]         info_file: ", mh_config.info_file)
    println(io, "[INFO]         adaptive: ", mh_config.adaptive)
    println(io, "[INFO]         nadapt: ", mh_config.nadapt)
    println(io, "[INFO]         adapt_cov_scale: ", mh_config.adapt_cov_scale)
    println(io, "[INFO]         continue_from_write_file: ", mh_config.continue_from_write_file)
    return
end

function _printinfo(verbose, info_io, samples_count, samples_buffer, start_time_sec, max_time_sec, maxsamples, loglike, infloglikecount)
    if !verbose
        return 
    end
    accratio = samples_buffer.accepted_count/samples_buffer.bufferidx
    println(info_io, "[INFO] Iteration $(samples_count).")
    println(info_io, "[INFO] Elapsed time $(elapsedtime(start_time_sec)) seconds.")
    println(info_io, "[INFO] Acceptance ratio $(accratio).")
    println(info_io, "[INFO] Number of proposed samples with infinite loglikelihood: $(infloglikecount) of the last $(samples_buffer.bufferidx) iterations.")
    println(info_io, "[INFO] Remaining time $(max_time_sec-(time()-start_time_sec)) seconds")
    println(info_io, "[INFO] Remaining iterations $(maxsamples-samples_count)")
    println(info_io, "[INFO] Current sample $(viewlatest(samples_buffer)).")
    println(info_io, "[INFO] Current loglikelihood $(loglike).")
    if accratio == zero(accratio)
        println(info_io, "[WARN] None of the previous $(samples_buffer.bufferidx) samples were accepted.")
    end
    flush(info_io)
    return 
end

function _printinfo_endstatus(verbose, info_io, start_time_sec, mh_config, samples_count, maxsamples, end_of_samples_pos, eof_pos, totalinfiniteloglikelihoodscount, uniquesamplescount)
    if !verbose
        return 
    end

    header_size = length((mh_config.nparams, samples_count))+1 # add 1 for dim field in file which is not part of header
    header_size_nbytes = header_size*sizeof(Int64)
    
    samples_size_nbytes = end_of_samples_pos - header_size_nbytes
    samples_size = (samples_size_nbytes/sizeof(eltype(mh_config.init_sample))/mh_config.nparams)
    
    total_size_nbytes = samples_size_nbytes + header_size_nbytes

    println(info_io, "[INFO] END STATUS")
    println(info_io, "[INFO]     elapsed time: $(elapsedtime(start_time_sec)) seconds.")
    println(info_io, "[INFO]     timeout: $(timeout(start_time_sec, mh_config.max_time_sec)).")
    println(info_io, "[INFO]     max iters reached: $(maxitersreached(samples_count, maxsamples)).")
    println(info_io, "[INFO]     number of unique samples: $(uniquesamplescount).")
    println(info_io, "[INFO]     number of proposed samples with infinite loglikelihood: $(totalinfiniteloglikelihoodscount)")
    println(info_io, "[INFO]     samples filesize:")
    println(info_io, "[INFO]         header: $header_size ($header_size_nbytes bytes).")
    println(info_io, "[INFO]         samples: $samples_size ($samples_size_nbytes bytes).")
    println(info_io, "[INFO]         total: $total_size_nbytes bytes (eof at $(eof_pos)).")
    printconfig(info_io, mh_config)
    return 
end

function elapsedtime(starttime)
    return time()-starttime
end
function timeout(starttime, max_time)
    return elapsedtime(starttime) >= max_time
end

function maxitersreached(iteration, max_iterations)
    return iteration >= max_iterations
end

function adaptmaybe!(symmetric_proposal_distribution, mh_config, info_io, samples_count, samples_buffer)
    if mh_config.adaptive && samples_count<=mh_config.nadapt
        mh_config.verbose && println(info_io, "[INFO] Adapting proposal at iteration $(samples_count).")
        adapt!(symmetric_proposal_distribution, samples_buffer.buffer, mh_config.adapt_cov_scale)
        mh_config.verbose && println(info_io, "[INFO] Proposal at iteration $(samples_count): $(symmetric_proposal_distribution)")
        flush(info_io)
    end
    return 
end

function calc_log_accept_ratio_contribution(sample, loglikelihood_fn, prior_logpdf)
    logpdf_prior = prior_logpdf(sample)
    if isinf(logpdf_prior)
        # skip expensive likelihood evaluation when logpdf_prior = -Inf
        loglikelihood_value = logpdf_prior
        log_accept_ratio_contrib = logpdf_prior
    else
        loglikelihood_value = loglikelihood_fn(sample)
        log_accept_ratio_contrib = loglikelihood_value + logpdf_prior
    end
    return log_accept_ratio_contrib, loglikelihood_value
end

function read_init_sample_from_end_of_file!(mh_config)
    return open(mh_config.samples_write_file, "r") do io 
        seekend(io)
        skip(io, -sizeof(eltype(mh_config.init_sample))*mh_config.nparams)
        for i in 1:mh_config.nparams
            mh_config.init_sample[i] = read(io, eltype(mh_config.init_sample))
        end
        read_binary_array_file_header(io)
    end
end

function init_sample_setup!(mh_config)
    if mh_config.continue_from_write_file
        if !isfile(mh_config.samples_write_file)
            error("MH IO file does not exist: $(mh_config.samples_write_file).")
        end
        header = read_init_sample_from_end_of_file!(mh_config) 
        samples_count = header[2]
    else
        if isfile(mh_config.samples_write_file) || isfile(mh_config.info_file)
            error("MH IO file(s) already exist\n    $(mh_config.samples_write_file)\n    $(mh_config.info_file).")
        end
        samples_count = 0
    end
    return samples_count
end

function check_init_params(init_params, logpdf_fn)
    logpdf_prior_value = logpdf_fn(init_params)
    if isinf(logpdf_prior_value) && logpdf_prior_value<zero(logpdf_prior_value)
        error("Initial params are not in the support of the prior.")
    end
    return 
end

function metropolis_hastings(rng::AbstractRNG, loglikelihood_fn, prior_logpdf_fn, symmetric_proposal_distribution, mh_config::MHConfig)
    # set up parameters in cases when we do or do not continue from a file
    samples_count = init_sample_setup!(mh_config)
    maxsamples = samples_count + mh_config.maxiters

    # set up first sample from config
    current_sample = mh_config.init_sample
    check_init_params(current_sample, prior_logpdf_fn)
    setstate!(symmetric_proposal_distribution, current_sample)
    infiniteloglikelihoodscount = 0
    current_log_accept_ratio_contrib, current_loglikelihood_value = 
        calc_log_accept_ratio_contribution(current_sample, loglikelihood_fn, prior_logpdf_fn)
    infiniteloglikelihoodscount += isinf(current_loglikelihood_value)

    # allocate space for samples
    proposed_sample = Vector{eltype(mh_config.init_sample)}(undef, mh_config.nparams)
    samples_buffer = SamplesBuffer(mh_config.init_sample, mh_config.samples_write_buffer_size)

    open(mh_config.info_file, "a") do info_io
    open(mh_config.samples_write_file, "a") do samples_io
        # the file is empty and we need to leave space for the header
        if position(samples_io)==0
            skip_binary_array_file_header(samples_io, length(size(samples_buffer.buffer)))
        end 
        # else, we assume there is already space for the header
        
        totalinfiniteloglikelihoodscount = 0
        uniquesamplescount = 0

        # continue the MCMC chain until timeout or maxiters reached
        start_time_sec = time()
        while !timeout(start_time_sec, mh_config.max_time_sec) && !maxitersreached(samples_count, maxsamples)
            # propose
            proposed_sample .= rand(rng, symmetric_proposal_distribution)
            proposed_log_accept_ratio_contrib, proposed_loglikelihood_value = 
                calc_log_accept_ratio_contribution(proposed_sample, loglikelihood_fn, prior_logpdf_fn)
            infiniteloglikelihoodscount += isinf(proposed_loglikelihood_value)
            log_accept_ratio = proposed_log_accept_ratio_contrib - current_log_accept_ratio_contrib
            
            # accept/reject
            samples_count += 1
            if log(rand(rng)) <= log_accept_ratio
                current_sample = addsample!(samples_buffer, proposed_sample)
                setstate!(symmetric_proposal_distribution, current_sample)
                current_log_accept_ratio_contrib, current_loglikelihood_value = proposed_log_accept_ratio_contrib, proposed_loglikelihood_value
            else 
                repeatsample!(samples_buffer)
            end

            # write buffer 
            if isbufferfull(samples_buffer)
                _printinfo(mh_config.verbose, info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglikelihood_value, infiniteloglikelihoodscount)

                totalinfiniteloglikelihoodscount += infiniteloglikelihoodscount
                infiniteloglikelihoodscount = 0
                uniquesamplescount += samples_buffer.accepted_count

                adaptmaybe!(symmetric_proposal_distribution, mh_config, info_io, samples_count, samples_buffer)
                writebuffer!(samples_io, samples_buffer)
            end
        end
        _printinfo(mh_config.verbose, info_io, samples_count, samples_buffer, start_time_sec, mh_config.max_time_sec, maxsamples, current_loglikelihood_value, infiniteloglikelihoodscount)
        # write remaing samples from buffer
        writepartialbuffer!(samples_io, samples_buffer)
        _printinfo_endstatus(mh_config.verbose, info_io, start_time_sec, mh_config, samples_count, maxsamples, position(samples_io), position(seekend(samples_io)), totalinfiniteloglikelihoodscount, uniquesamplescount)
        # now that the number of samples is known, add the file header
        header = (mh_config.nparams, samples_count)
        write_binary_array_file_header(samples_io, header)
    end # close params io
    end # close info io
    return samples_count
end

end