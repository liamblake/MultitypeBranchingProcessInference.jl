mutable struct SamplesBuffer{T}
    const buffer::T
    bufferidx::Int
    accepted_count::Int
    const buffersize::Int
end

function SamplesBuffer(init_values, buffersize, ll=nothing)
    buffer = Array{eltype(init_values), 2}(undef, length(init_values)+(ll!==nothing), buffersize)
    if ll===nothing 
        for col in eachcol(buffer)
            col .= init_values
        end
    else
        for col in eachcol(buffer)
            col[1:end-1] .= init_values
            col[end] = ll
        end
    end
    return SamplesBuffer(
        buffer,
        0,
        0,
        buffersize,
    )
end

function getlatestidx(buffer)
    if isbufferempty(buffer)
        return buffer.buffersize
    end
    return buffer.bufferidx
end

function viewlatest(buffer)
    latestidx = getlatestidx(buffer)
    return @view buffer.buffer[:, latestidx]
end

function addsample!(buffer, sample, ll=nothing)
    buffer.bufferidx += 1
    buffer.accepted_count += 1
    if ll===nothing
        buffer.buffer[:, buffer.bufferidx] .= sample
        latest = @view buffer.buffer[:, buffer.bufferidx]
    else
        buffer.buffer[1:end-1, buffer.bufferidx] .= sample
        latest = @view buffer.buffer[1:end-1, buffer.bufferidx]
        buffer.buffer[end, buffer.bufferidx] = ll
    end
    return latest
end

function repeatsample!(buffer)
    latestidx = getlatestidx(buffer)
    buffer.bufferidx += 1
    buffer.buffer[:, buffer.bufferidx] .= buffer.buffer[:, latestidx]
    latest = @view buffer.buffer[:, buffer.bufferidx]
    return latest
end

function isbufferempty(buffer)
    return buffer.bufferidx == 0
end

function isbufferfull(buffer)
    return buffer.bufferidx >= buffer.buffersize
end

function writebuffer!(io, buffer)
    write(io, buffer.buffer)
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end

function writepartialbuffer!(io, buffer)
    write(io, buffer.buffer[:, 1:buffer.bufferidx])
    buffer.bufferidx = 0
    buffer.accepted_count = 0
    return 
end
