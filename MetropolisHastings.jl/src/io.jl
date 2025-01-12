function skip_binary_array_file_header(io, dim)
    @assert position(io)==0 "Expected io to be at the start of the file"
    write(io, Int64(dim))
    for _ in 1:dim
        write(io, Int64(0))
    end
    return 
end
function write_binary_array_file_header(io, array_size)
    seekstart(io)
    write(io, Int64(length(array_size)))
    for sz in array_size
        write(io, Int64(sz))
    end
    return 
end
function read_binary_array_file_header(io)
    seekstart(io)
    dim = read(io, Int64)
    array_size = Vector{Int64}(undef, dim)
    for i in 1:dim
        array_size[i] = read(io, Int64)
    end
    return array_size
end
function read_binary_array_file(io, dtype=Float64)
    array_size = read_binary_array_file_header(io)
    data_array = Array{dtype, length(array_size)}(undef, array_size...)
    for i in 1:prod(array_size)
        data_array[i] = read(io, dtype)
    end
    return data_array
end