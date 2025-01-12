struct MTBPMomentsOperator{C<:AbstractMatrix, M, S}
    ntypes::Int
    characteristicmatrix::C
    generator::C
    exp_method::M
    exp_malloc::S
    function MTBPMomentsOperator{C,M,S}(
        ntypes::Int, characteristicmatrix::C, generator::C, exp_method::M, exp_malloc::S,
    ) where {C<:AbstractMatrix, M, S}
        @assert size(generator)==(ntypes^2+ntypes, ntypes^2+ntypes)
        return new{C,M,S}(ntypes, characteristicmatrix, generator, exp_method, exp_malloc)
    end
    MTBPMomentsOperator(ntypes::Int, characteristicmatrix::C, generator::C, exp_method::M, exp_malloc::S) where {C<:AbstractMatrix, M, S} = 
        MTBPMomentsOperator{C,M,S}(ntypes, characteristicmatrix, generator, exp_method, exp_malloc)
end

function paramtype(o::MTBPMomentsOperator)
    return eltype(o.generator)
end

function getntypes(o::MTBPMomentsOperator)
    return o.ntypes
end

function MTBPMomentsOperator(p, T=Float64)
    generator = zeros(T, p^2+p, p^2+p)
    characteristicmatrix = zeros(T, p, p)
    method = ExpMethodHigham2005(generator)
    exp_malloc = ExponentialUtilities.alloc_mem(generator, method)
    return MTBPMomentsOperator(p, characteristicmatrix, generator, method, exp_malloc)
end

function mean!(m, op::MTBPMomentsOperator, state)
    p = getntypes(op)
    return @views mul!(m, op.generator[end-p+1:end, end-p+1:end], state)
end
mean(op::MTBPMomentsOperator, state::AbstractArray) = 
    mean!(similar(op.characteristicmatrix, getntypes(op)), op, state)

function variance_covariance!(v, op::MTBPMomentsOperator, state)
    v_flat = reshape(@view(v[axes(v)...]), :, 1)
    p = getntypes(op)
    @views mul!(v_flat, op.generator[1:end-p, end-p+1:end], state)
    return v
end
variance_covariance(op::MTBPMomentsOperator, state) = 
    variance_covariance!(similar(op.characteristicmatrix), op, state)

function moments!(op::MTBPMomentsOperator, bp, t=nothing)
    characteristicmatrix!(op.characteristicmatrix, bp)
    if t===nothing
        t = one(paramtype(op))
    end
    p = op.ntypes
    # Generator is 
    # +---------------+
    # |  Ω' ⊕ Ω'   C  |
    # |     0      Ω' |
    # +---------------+
    # (also multiply by t) where Ω' is the characteristic matric and 
    # the columns of C are the pairwise product moments of the offspring
    # distribution. ⊕ is the kronecker product, A ⊕ B = A ⊗ I + I ⊗ B, 
    # where ⊗ is the kronecker product.
    op.generator .= zeros(eltype(op.generator))
    # bottom right block, Ω'
    op.generator[end-p+1:end, end-p+1:end] .= op.characteristicmatrix*t
    t_characteristicmatrix = op.generator[end-p+1:end, end-p+1:end]
    # Top-left block. 
    # First, add kron(I, op.characteristicmatrix*t) = I ⊗ Ω'
    for i in 1:p
        offset = (i-1)*p
        op.generator[offset.+(1:p), offset.+(1:p)] .= t_characteristicmatrix
    end
    # Then, add kron(op.characteristicmatrix*t, I) = Ω' ⊗ I 
    for i in 1:p
        offset_i = (i-1)*p
        for j in 1:p
            offset_j = (j-1)*p
            for k in 1:p
                op.generator[offset_i+k, offset_j+k] += t_characteristicmatrix[i,j]
            end
        end
    end
    # Top-right block, add second moments of progeny distribution.
    for i in 1:p
        op.generator[1:end-p, end-p+i] .= bp.progeny[i].second_moments[:]
        op.generator[1:end-p, end-p+i] .*= bp.rates[i]*t
    end
    exponential!(op.generator, op.exp_method, op.exp_malloc)
    return 
end