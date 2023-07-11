# DifferentiableWasserstein
This repository implements the well-known exact solution for the 1D Wasserstein distance between two 1D distributions,
represented by discrete samples with uniform weights. Its implementation can be differentiated automatically by `Zygote.jl`.

The validity of the implementation is verified by comparing the results with the scipy implemenation of the 1D Wasserstein distance.
## Usage

```julia
using Random
Random.seed!(1234)
u_samples = randn(1000)
v_samples = randn(1000)
W₁(u_samples, v_samples) # 0.01903452841031397
```

## Installation
    
```julia
using Pkg; Pkg.add("https://github.com/hsianktin/DifferentiableWasserstein.jl")
```