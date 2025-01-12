@testset "ParticleFilter/observationmodel.jl" begin
    @testset "defaults" begin
        om = LinearGaussianObservationModel([1 0.])

        @test logpdf(om, [0.], [0.; 1.]) ≈ -0.5*log(2*pi)
        @test logpdf(om, [1.], [1.; 0.]) ≈ -0.5*log(2*pi)
        @test logpdf(om, [1.], [0.; 1.]) ≈ -0.5*log(2*pi)-0.5
        @test logpdf(om, [0.], [1.; 0.]) ≈ -0.5*log(2*pi)-0.5
        @test logpdf(om, ([0.],[1.]), [0.; 1.]) ≈ (-0.5*log(2*pi)) + (-0.5*log(2*pi)-0.5)
    end
    @testset "f32" begin
        om = LinearGaussianObservationModel([1f0 0.f0])

        @test logpdf(om, [0.f0], [0.f0; 1.f0]) ≈ -0.5f0*log(2f0*Float32(pi))
        @test logpdf(om, [1.f0], [1.f0; 0.f0]) ≈ -0.5f0*log(2f0*Float32(pi))
        @test logpdf(om, [1.f0], [0.f0; 1.f0]) ≈ -0.5f0*log(2f0*Float32(pi))-0.5f0
        @test logpdf(om, [0.f0], [1.f0; 0.f0]) ≈ -0.5f0*log(2f0*Float32(pi))-0.5f0
        @test logpdf(om, ([0.f0], [1.f0]), [0.f0; 1.f0]) ≈ (-0.5f0*log(2f0*Float32(pi))) + (-0.5f0*log(2f0*Float32(pi))-0.5f0)
    end
    @testset "multiple observations" begin
        om = LinearGaussianObservationModel([1 0.])

        @test logpdf(om, [[0.], [1.]], [0.; 1.]) ≈ (-0.5*log(2*pi)) + (-0.5*log(2*pi)-0.5)
    end
end