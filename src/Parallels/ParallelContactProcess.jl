include("../ContactProcess.jl")
module ParallelContactProcess
    export initialize_state_and_rates, calculate_all_rates, sample_time_with_rates, update_states!, update_rates!, add_noise, run_simulation, generate_animation!
    using Random
    using Revise
    using Plots
    using Distributions
    using StatsBase
    using Parameters
    using ProgressMeter
    using JLD2
    using Base.Threads
    using CUDA
    # use CUDA to speed up the computation of every function in the ContactProcess module
    CUDA.allowscalar(false)
    
    @with_kw struct GridParameters
        width::Int = 15
        height::Int = 15
        width_with_padding::Int = width + 2
        height_with_padding::Int = height + 2

    end

    @with_kw struct ModelParameters
        infection_rate::Float64 = 0.05
        recovery_rate::Float64 = 0.1
        time_limit::Int = 100
        prob_infections::Float64 = 0.3
        num_simulations::Int = 1000
    end
    @inline function calculate_rate(state, i, j, grid_params, model_params)
        rate = 0.0
        if state[i, j]
            rate = model_params.recovery_rate
        else
            rate = model_params.infection_rate * (state[i - 1, j] + state[i + 1, j] + state[i, j - 1] + state[i, j + 1])
        end
        return rate
    end

    function initialize_state_and_rates_kernel(state, rates, rand_vals, grid_params, model_params, mode)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1# 0 based indexing
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1# 0 based indexing
        # println(i, j)
        if i < grid_params.width_with_padding && j < grid_params.height_with_padding
            if mode == 1
                if rand_vals[i, j] < 0.25
                    state[i, j] = true
                end
            elseif mode == 2
                if rand_vals[i, j] < model_params.prob_infections
                    state[i, j] = true
                end
            end
            rates[i, j] = calculate_rate(state, i, j, grid_params, model_params)
        end
        return
    end

    function initialize_state_and_rates(grid_params, model_params; mode = "fixed_probability")
        state = CUDA.zeros(Bool, grid_params.width_with_padding, grid_params.height_with_padding)
        rates = CUDA.zeros(grid_params.width_with_padding, grid_params.height_with_padding)
        rand_vals = CUDA.CuArray(CUDA.rand(grid_params.width_with_padding, grid_params.height_with_padding))
        numblocks = (grid_params.width, grid_params.height)
        numthreads = (1, 1)
        mode_int = mode == "complete_chaos" ? 1 : 2
        @cuda threads=(numthreads) blocks=(numblocks) initialize_state_and_rates_kernel(state, rates, rand_vals, grid_params, model_params, mode_int)

        return state, rates
    end

    function calculate_all_rates_kernel(state, rates, grid_params, model_params)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1# 0 based indexing
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1# 0 based indexing
        if i <= grid_params.width && j <= grid_params.height
            rates[i, j] = calculate_rate(state, i, j, grid_params, model_params)
        end
        return
    end

    function calculate_all_rates(state, rates, grid_params, model_params)
        numblocks = (grid_params.width, grid_params.height)
        numthreads = (1, 1)
        @cuda threads=(numthreads) blocks=(numblocks) calculate_all_rates_kernel(state, rates, grid_params, model_params)
        return
    end

    function sample_time_with_rates(rates, grid_params, model_params)
        total_rate = CUDA.sum(rates)
        time = rand(Exponential(1 / total_rate))
        return time
    end

    function update_states_kernel(state, rates, rand_vals, grid_params, model_params, time)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1# 0 based indexing
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1# 0 based indexing
        if i <= grid_params.width_with_padding && j <= grid_params.height_with_padding
            if state[i, j]
                if rand_vals[i, j] < model_params.recovery_rate * time
                    state[i, j] = false
                end
            else
                if rand_vals[i, j] < model_params.infection_rate * time * (state[i-1, j] + state[i+1, j] + state[i, j-1] + state[i, j+1])
                    state[i, j] = true
                end
            end
        end
        return
    end

    function update_states(state, rates, grid_params, model_params, time)
        numblocks = (grid_params.width, grid_params.height)
        numthreads = (1, 1)
        rand_vals = CUDA.CuArray(CUDA.rand(grid_params.width, grid_params.height_with_padding))
        @cuda threads=(numthreads) blocks=(numblocks) update_states_kernel(state, rates, rand_vals, grid_params, model_params, time)
        return
    end

    function update_rates_kernel(state, rates, grid_params, model_params)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + 1# 0 based indexing
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + 1# 0 based indexing
        if i <= grid_params.width && j <= grid_params.height
            rates[i, j] = calculate_rate(state, i, j, grid_params, model_params)
        end
        return
    end

    # function run_simulation(state, rates, grid_params, model_params)
    #     state_sequence = []
    #     transition_times = []
    #     updated_nodes = []
    #     push!(state_sequence, copy(state))
    #     time = 0.0
    #     while time < model_params.time_limit
    #         calculate_all_rates(state, rates, grid_params, model_params)
    #         time += sample_time_with_rates(rates, grid_params, model_params)
    #         update_states(state, rates, grid_params, model_params, time)
    #         push!(state_sequence, copy(state))
    #         push!(transition_times, time)
    #     end
    #     return state_sequence, transition_times, updated_nodes
    # end

    function run_simulation(grid_params, model_params)
        state = CUDA.zeros(Bool, grid_params.width_with_padding, grid_params.height_with_padding)
        rates = CUDA.zeros(grid_params.width_with_padding, grid_params.height_with_padding)
        rand_vals = CUDA.CuArray(CUDA.rand(grid_params.width_with_padding, grid_params.height_with_padding))
        numblocks = (grid_params.width, grid_params.height)
        numthreads = (1, 1)
        mode_int = mode == "complete_chaos" ? 1 : 2
        @cuda threads=(numthreads) blocks=(numblocks) initialize_state_and_rates_kernel(state, rates, rand_vals, grid_params, model_params, mode_int)
        state_sequence = []
        transition_times = []
        updated_nodes = []
        push!(state_sequence, copy(state))
        time = 0.0
        while time < model_params.time_limit
            calculate_all_rates(state, rates, grid_params, model_params)
            time += sample_time_with_rates(rates, grid_params, model_params)
            update_states(state, rates, grid_params, model_params, time)
            push!(state_sequence, copy(state))
            push!(transition_times, time)
        end
        return state_sequence, transition_times, updated_nodes
    end

    # run multiple simulations optimally using CUDA where num_simulations is the number of simulations to run


end

# time the standard initialization and the CUDA initialization
grid_params = ParallelContactProcess.GridParameters(width=1000, height=1000)
model_params = ParallelContactProcess.ModelParameters(infection_rate=0.05, recovery_rate=0.1, time_limit=1, prob_infections=0.3, num_simulations=1000)
@time state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params; mode="fixed_probability")
timing = @time new_state, new_rates = ParallelContactProcess.initialize_state_and_rates(grid_params, model_params; mode="fixed_probability");
state_cpu = Array(new_state)
rates_cpu = Array(new_rates)

state_sequence, transition_times, updated_nodes = ParallelContactProcess.run_simulation(grid_params, model_params)
ϵ = 1e-6
∈