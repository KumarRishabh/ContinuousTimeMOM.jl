module ParallelCTMOM
    using PrettyTables
    using Revise
    using Plots
    using Random
    using ProgressMeter
    # using JLD2 # for saving and loading the data efficiently
    include("ParallelContactProcess.jl")
    using .ContactProcess
    using JLD2
    using Distributions
    Random.seed!(10)
    using Parameters
    using Distributed
    using Serialization
    using FileIO
    using CUDA  

    @with_kw struct ModelParameters
        infection_rate::Float64 = 0.05
        recovery_rate::Float64 = 0.1
        time_limit::Int = 100
        prob_infections::Float64 = 0.3
        num_simulations::Int = 1000
        num_particles::Int = 20000
        r::Int = 5
        testing_mode::Bool = false
    end


end