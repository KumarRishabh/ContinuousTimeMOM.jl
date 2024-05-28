# Data Generating model as per contact process on a M \times M grid
# JULIA_NUM_THREADS=4 julia --project=experiments/ -t 4 experiments/MCSimulation%28Ex%202%29.jl
Threads.nthreads()
module ContactProcess
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
    # Initialize parameters
    dir_path = joinpath(@__DIR__, "../experiments/data") # directory path to save the data generated from the simulations
    @with_kw struct GridParameters
        width::Int64 = 200    
        height::Int64 = 200
        width_with_padding::Int64 = width + 2
        height_with_padding::Int64 = height + 2
    end
    
    @with_kw struct ModelParameters
        infection_rate::Float64 = 2.0
        recovery_rate::Float64 = 1.5
        prob_noise::Float64 = 0.05
        # num_steps::Int64 = 300
        time_limit::Float64 = 100.0
        prob_infections::Float64 = 0.01 # Bernoulli distributions
        num_simulations::Int64 = 1
    end
    
    grid_params = GridParameters()
    model_params = ModelParameters()


    # Initialize the grid
    #
    """
    initialize_state_and_rates(grid_params, model_params; mode = "fixed_probability")

    Initialize the state and rates for the contact process simulation.

    # Arguments
    - `grid_params`: The parameters of the grid.
    - `model_params`: The parameters of the contact process model.
    - `mode`: The mode for initializing the state. Default is "complete_chaos" or "fixed_probability".

    # Returns
    - `state`: The initial state of the grid.
    - `rates`: The initial rates for each node in the grid.
    """
    function initialize_state_and_rates(grid_params, model_params; mode = "fixed_probability")
        state = zeros(Bool, grid_params.width_with_padding, grid_params.height_with_padding)
        rates = zeros(grid_params.width_with_padding, grid_params.height_with_padding)

        if mode == "complete_chaos"
            # Sample randomly the nodes which will be infected (mode = "complete_chaos")
            for i in 2:grid_params.width_with_padding - 1
                for j in 2:grid_params.height_with_padding - 1
                    if rand() < 0.3
                        state[i, j] = true
                    end
                end
            end
        end
        # Sample randomly the nodes which will be infected (mode = "fixed_probability")
        if mode == "fixed_probability"
            for i in 2:grid_params.width + 1
                for j in 2:grid_params.height + 1
                    if rand() < model_params.prob_infections
                        state[i, j] = true
                    end
                end
            end
        end
        # update the rates for the infected nodes
        rates = calculate_all_rates(state, grid_params, model_params)
        return state, rates
    end



    # Function to update the state of the grid

    function calculate_all_rates(state, grid_params, model_params)
        # After calculating the rates, we need to select the position of the node 
        # which will be updated
        rates = zeros(grid_params.width_with_padding, grid_params.height_with_padding)
        for i in 2:grid_params.width_with_padding-1
            for j in 2:grid_params.height_with_padding-1
                rates[i, j] = model_params.infection_rate *(1 - state[i, j]) * (state[i, j+ 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + model_params.recovery_rate * state[i, j]
            end
        end
        # choose a grid position according to the rate 
        return rates
    end

    function sample_time_with_rates(state, rates)
        rate = sum(rates)
        # sample from the Exponential rate
        time = rand(Exponential(1 / rate))
        return time
    end

    # function to add two integers 

    # function to multiply any number of integers using SIMD instructions





    function update_states!(state, rates, grid_params; debug_mode::Bool = false)
        r = rand(); flag = false;
        debug_mode == true && println("Random number: ", r)  # Turn debug mode on
        cum_rate = 0.0
        updated_node = (0, 0)
        total_rate = sum(rates)
        for i in 2:grid_params.width_with_padding-1
            for j in 2:grid_params.height_with_padding-1
                # update the state of the node if the cumulative rate is greater than the random number
                cum_rate += rates[i, j]
                if cum_rate >= r * total_rate
                    debug_mode == true && println("State updated")
                    state[i, j] = !state[i, j]
                    updated_node = (i, j)
                    flag = true
                    break
                end
            end
            # Check this piece of code
            if flag == true
                break
            end 
            if cum_rate >= r * total_rate
                debug_mode == true && println("State updated")
                state[i, height] = !state[i, height]
                updated_node = (i, height)
                flag = true
            end
        end
        debug_mode && println("Random number: ", r) # Turn debug mode off
        return state, updated_node
    end


    #=
    After updating the state, we need to update the rates of the neighbors of the updated node 
    =#
    function update_rates!(state, rates, updated_node, grid_params, model_params)
        i, j = updated_node
        # If the updated_note is changed to infected, then the rate will be that of recovery and similary when the node is 
        # changed to uninfected, the rate will be that of infection

        rates[i, j] = model_params.infection_rate *(1 - state[i, j]) * (state[i, j+ 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + model_params.recovery_rate * state[i, j]
        if j + 1 == grid_params.width_with_padding
            rates[i, j + 1] = 0 # Boundary condition
        else
            rates[i, j + 1] = model_params.infection_rate * (1 - state[i, j + 1]) * (state[i, j + 2] + state[i, j] + state[i - 1, j + 1] + state[i + 1, j + 1]) + model_params.recovery_rate * state[i, j + 1]
        end
        
        if i + 1 == grid_params.height_with_padding
            rates[i + 1, j] = 0 # Boundary condition
        else
            rates[i + 1, j] = model_params.infection_rate * (1 - state[i + 1, j]) * (state[i + 1, j + 1] + state[i + 1, j - 1] + state[i, j] + state[i + 2, j]) + model_params.recovery_rate * state[i + 1, j]
        end
        
        if j - 1 == 1
            rates[i, j - 1] = 0 # Boundary condition
        else
            rates[i, j - 1] = model_params.infection_rate * (1 - state[i, j - 1]) * (state[i, j] + state[i, j - 2] + state[i - 1, j - 1] + state[i + 1, j - 1]) + model_params.recovery_rate * state[i, j - 1]
        end

        if i - 1 == 1
            rates[i - 1, j] = 0 # Boundary condition
        else
            rates[i - 1, j] = model_params.infection_rate * (1 - state[i - 1, j]) * (state[i - 1, j + 1] + state[i - 1, j - 1] + state[i - 2, j] + state[i, j]) + model_params.recovery_rate * state[i - 1, j]
        end
        
        return rates
    end
        

    # Function to add noise to observations
    function add_noise(state, grid_params)
        noisy_state = copy(state)
        for i in 2:grid_params.width
            for j in 2:grid_params.height
                if rand() < prob_noise
                    noisy_state[i, j] = !noisy_state[i, j]
                end
            end
        end
        return noisy_state
    end
    # print(state)
    # Initialize the plot

    # function plot_noisy_state(noisy_state)
        
    # heatmap(state, color=:green)
    # Run the simulation

    function run_simulation!(state, rates, grid_params, model_params; debug_mode::Bool = false)
        # initialize states 
        state_sequence = []
        times = [0.0]
        updated_nodes = [(1, 1)] # don't have to count the (1, 1) node, just a placeholder
        push!(state_sequence, state)
        time_elapsed = 0.0
        while time_elapsed < model_params.time_limit
            rates = calculate_all_rates(state_sequence[end], grid_params, model_params)
            time = sample_time_with_rates(state_sequence[end], rates)
            if time + times[end] > model_params.time_limit
                break
            end
            new_state, updated_node = update_states!(copy(state_sequence[end]), copy(rates), grid_params; debug_mode = debug_mode)
            rates = update_rates!(new_state, copy(rates), updated_node, grid_params, model_params)
            push!(state_sequence, new_state)
            push!(times, times[end] + time)
            push!(updated_nodes, updated_node)
            time_elapsed += time
        end

        return state_sequence, times, updated_nodes
    end

    # function to run multiple simulations according to the model parameters and re-using the simulations function
    # use array of array to store the state sequences and times
    function multiple_simulations(grid_params, model_params)
        state_sequences = Array{Vector{Any}, 1}(undef, model_params.num_simulations)
        times = Array{Array{Float64, 1}, 1}(undef, model_params.num_simulations)
        updated_nodes = Array{Array{Tuple{Int64, Int64}, 1}, 1}(undef, model_params.num_simulations)
        for i in 1:model_params.num_simulations
            state, rates = initialize_state_and_rates(grid_params, model_params)
            interim_state_sequences, interim_times, interim_updated_node = run_simulation!(state, rates, grid_params, model_params)
            # state_sequences[i], times[i] = run_simulation!(state, rates, grid_params, model_params)
            state_sequences[i] = interim_state_sequences
            times[i] = interim_times
            updated_nodes[i] = interim_updated_node
            println("Simulation: $i")

            
        end
        return state_sequences, times, updated_nodes
    end

    function save_multiple_simulations(grid_params, model_params; dir_path = dir_path)
        progress = Progress(model_params.num_simulations, 1, "Running simulations...")
        if has_cuda_gpu()
            @threads for i in 1:model_params.num_simulations
                state, rates = initialize_state_and_rates(grid_params, model_params)
                state_sequences, times, updated_nodes = run_simulation!(state, rates, grid_params, model_params)
                # save("$dir_path/state_sequences_$i.jld", "state_sequences", state_sequences)
                save("$dir_path/initial_state_$i.jld", "initial_state", state)
                save("$dir_path/times_$i.jld", "times", times)
                save("$dir_path/updated_nodes_$i.jld", "updated_nodes", updated_nodes)
            end
        else
            for i in 1:model_params.num_simulations
                state, rates = initialize_state_and_rates(grid_params, model_params)
                state_sequences, times, updated_nodes = run_simulation!(state, rates, grid_params, model_params)
                # save("$dir_path/state_sequences_$i.jld", "state_sequences", state_sequences)
                save("$dir_path/initial_state_$i.jld", "initial_state", state)
                save("$dir_path/times_$i.jld", "times", times)
                save("$dir_path/updated_nodes_$i.jld", "updated_nodes", updated_nodes)
                next!(progress)
            end
        end
    end

    function make_state_sequence(initial_state, updated_nodes, grid_params, model_params) # will make things easier while loading the saved files
        state_sequence = [initial_state]
        for i ∈ 2:length(updated_nodes)
            updated_node = updated_nodes[i]
            temp_state = deepcopy(state_sequence[end])  # Copy the last state
            # if initial_state[updated_node[1], updated_node[2]] = false then state has to be updated to true and vice-versa
            if state_sequence[end][updated_node[1], updated_node[2]] == false
                temp_state[updated_node[1], updated_node[2]] = true
            else
                temp_state[updated_node[1], updated_node[2]] = false
            end
            push!(state_sequence, temp_state)
        end
        return state_sequence
    end
    # function multiple_simulations(grid_params, model_params)
    #     state_sequences = [] # Array{Array{Bool, 2}, model_params.num_simulations}
    #     state_sequences = Array{Array{Bool, 2}}(undef, model_params.num_simulations)
    #     times = Array{Float64}(undef, model_params.num_simulations)
    #     for i in 1:model_params.num_simulations
    #         state, rates = initialize_state_and_rates(grid_params, model_params)
    #         state_sequences[i], times[i] = run_simulation!(state, rates, grid_params, model_params)
    #     end
    #     return state_sequences, times
    # end

    function generate_animation(state_sequence, times)

        # p = heatmap(state_sequence[1], color=:grays, legend = false)
        p = heatmap(state_sequence[1], color=:grays, legend = false)

        anim = @animate for step in eachindex(state_sequence)
            println("Step: $step")
            state = state_sequence[step]
            # print("State:", state)

            heatmap!(state, color=:grays, legend=false)
            # Put time step in the title
            title!("Time step: $(times[step])")
        end
        return anim
    end
    
end # module
