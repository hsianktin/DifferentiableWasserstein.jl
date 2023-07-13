using DifferentiableWasserstein
using Test
using Random
using PyCall
using Zygote
using Flux
using ExactOptimalTransport

@testset "DifferentiableWasserstein.jl" begin

    Random.seed!(1234)
    flag = true
    for i in 1:1000
        u_samples = rand(1000)
        v_samples = rand(1000)
        if !(W₁(u_samples, v_samples) ≈ ExactOptimalTransport.wasserstein(discretemeasure(u_samples), discretemeasure(v_samples)))
            flag = false
        end
    end
    @test flag == true

    Random.seed!(1234)
    pars = [1.0,2.0]
    u_samples = rand(1000)
    v_samples = rand(1000)
    function f(pars, u_samples, v_samples)
        par1, par2 = pars
        return W₁(u_samples, v_samples .* par2)
    end

    grads = gradient(Flux.params(pars)) do
        f(pars, u_samples, v_samples)
    end

    opt = ADAM(0.01)

    for i in 1:1000
        grads = gradient(Flux.params(pars)) do
            f(pars, u_samples, v_samples)
        end
        Flux.update!(opt, Flux.params(pars), grads)
    end

    @test (pars[2] - 1.0) < 0.1


    Random.seed!(1234)
    pars = [1.0,2.0]
    u_samples = randn(2000)
    v_samples = randn(1000)
    function f(pars, u_samples, v_samples)
        par1, par2 = pars
        return W₂(u_samples, v_samples .* par2)
    end

    grads = gradient(Flux.params(pars)) do
        f(pars, u_samples, v_samples)
    end

    opt = ADAM(0.01)

    for i in 1:1000
        grads = gradient(Flux.params(pars)) do
            f(pars, u_samples, v_samples)
        end
        Flux.update!(opt, Flux.params(pars), grads)
    end

    @test (pars[2] - 1.0) < 0.1

end
