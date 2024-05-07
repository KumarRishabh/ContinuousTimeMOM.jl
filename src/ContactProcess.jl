# Data Generating model as per contact process on a torus
module ContactProcess
    export initialize_state_and_rates, calculate_all_rates, sample_time_with_rates, update_states!, update_rates!, add_noise, run_simulation, generate_animation!
    using Random
    using Plots
    using Distributions
    using StatsBase

    # Initialize parameters
    width, height = 200, 200# dimensions of the grid
    # add paddings to the grid 
    width_with_padding = width + 2
    height_with_padding = height + 2
    # prob_infect = 0.2       # probability of infecting a neighbor
    infection_rate = 2 # a
    # prob_recover = 0.1 
    recovery_rate = 1.5 # s   # probability of recovery
    prob_noise = 0.05       # probability of observation noise
    num_steps = 300       # number of time steps to simulate


    # Initialize the grid
    function initialize_state_and_rates(num_infections)
        global width_with_padding, height_with_padding
        state = zeros(Bool, width_with_padding, height_with_padding)
        rates = zeros(width_with_padding, height_with_padding)
        # randomly infect a according to num_infections without replacemt and without paddings
        infected_nodes = sample(1:width * height, num_infections, replace=false)
        # convert the infected nodes to the padded grid
        print("Infected nodes: ", length(infected_nodes))
        for node in infected_nodes
            i, j = round(Int, node / width) + 1, node % width + 2
            state[i, j] = true
        end
        # update the rates for the infected nodes
        rates = calculate_all_rates(state)
        return state, rates
    end

    # Function to update the state of the grid

    function calculate_all_rates(state)
        # After calculating the rates, we need to select the position of the node 
        # which will be updated
        global width_with_padding, height_with_padding
        rates = zeros(width_with_padding, height_with_padding)
        for i in 2:width_with_padding-1
            for j in 2:height_with_padding-1
                rates[i, j] = infection_rate *(1 - state[i, j]) * (state[i, j+ 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + recovery_rate * state[i, j]
            end
        end
        # choose a grid position according to the rate 
        return rates
    end

    function sample_time_with_rates(state, rates)
        global width_with_padding, height_with_padding
        rate = sum(rates)
        # sample from the Exponential rate
        time = rand(Exponential(1 / rate))
        return time
    end

    # function to add two integers 

    # function to multiply any number of integers using SIMD instructions





    function update_states!(state, rates; debug_mode::Bool = false)
        global width_with_padding, height_with_padding; r = rand(); flag = false;
        debug_mode == true && println("Random number: ", r)  # Turn debug mode on
        cum_rate = 0.0
        updated_node = (0, 0)
        total_rate = sum(rates)
        for i in 2:width_with_padding-1
            for j in 2:height_with_padding-1
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
    function update_rates!(state, rates, updated_node)
        global width_with_padding, height_with_padding
        i, j = updated_node
        # If the updated_note is changed to infected, then the rate will be that of recovery and similary when the node is 
        # changed to uninfected, the rate will be that of infection

        rates[i, j] = infection_rate *(1 - state[i, j]) * (state[i, j+ 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + recovery_rate * state[i, j]
        if j + 1 == width_with_padding
            rates[i, j + 1] = 0 # Boundary condition
            
        elseif i + 1 == height_with_padding
            rates[i + 1, j] = 0 # Boundary condition
        
        elseif j - 1 == 1
            rates[i, j - 1] = 0 # Boundary condition

        elseif i - 1 == 1
            rates[i - 1, j] = 0 # Boundary condition
        
        else 
            rates[i, j + 1] = infection_rate * (1 - state[i, j]) * (state[i, j + 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + recovery_rate * state[i, j]
        end 
        
        return rates
    end
        

    # Function to add noise to observations
    function add_noise(state)
        noisy_state = copy(state)
        for i in 1:width
            for j in 1:height
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

    function run_simulation!(state, rates, num_steps; debug_mode::Bool = false)
        # initialize states 
        state_sequence = []
        times = [0.0]
        push!(state_sequence, state)
        for _ in 1:num_steps
            rates = calculate_all_rates(state_sequence[end])
            time = sample_time_with_rates(state_sequence[end], rates)
            new_state, updated_node = update_states!(copy(state_sequence[end]), copy(rates); debug_mode = debug_mode)
            rates = update_rates!(new_state, copy(rates), updated_node)
            push!(state_sequence, new_state)
            push!(times, times[end] + time)
        end
        return state_sequence, times
    end

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
