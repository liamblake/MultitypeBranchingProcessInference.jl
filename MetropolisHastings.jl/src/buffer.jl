mutable struct SamplesBuffer{T}
    const buffer::T
    bufferidx::Int
    accepted_count::Int
    const buffersize::Int
end

function SamplesBuffer(init_values, buffersize)
    buffer = Array{eltype(init_values), 2}(undef, length(init_values), buffersize)
    for col in eachcol(buffer)
        col .= init_values
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

function addsample!(buffer, sample)
    buffer.bufferidx += 1
    buffer.accepted_count += 1
    buffer.buffer[:, buffer.bufferidx] .= sample
    latest = @view buffer.buffer[:, buffer.bufferidx]
    return latest
end

function repeatsample!(buffer)
    latestidx = getlatestidx(buffer)
    buffer.bufferidx += 1
    buffer.buffer[:, buffer.bufferidx] .= buffer.buffer[:, latestidx]
    return 
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
