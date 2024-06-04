module CTMOM
    using PrettyTables
    using Revise
    using Plots
    using Random
    using ProgressMeter
    # using JLD2 # for saving and loading the data efficiently
    include("ContactProcess.jl")
    using .ContactProcess
    using JLD2
    using Distributions
    Random.seed!(10)
    using Parameters
    using Distributed
    using Serialization
    using FileIO


    export Particle, initialize_particles_as_vectors, biased_initialize_particles_as_vectors, initialize_particles_as_hashmap, initialize_particles_with_signals_as_hashmap, observe_node, observe_state, get_observations_from_state_sequence, get_observations, likelihood, count_total_particles, branching_particle_filter, estimate_infection_grid, actual_infection_grid, calculate_error, calculate_error_trajectory, get_heatmap_data, plot_heatmap, save_particle_history, save_simulation_to_csv

    dir_path = joinpath(@__DIR__, "./filtered_data")
    @with_kw mutable struct Particle
        # state of the particle is defined as a 2D array with the same dimensions as the grid
        state::Array{Bool, 2} = zeros(Bool, 12, 12) # 10 x 10 grid with all zeros
        weight::Float64 = 1.0
        # age::Int = 0 # age of the particle
        # child::Int = 0 # 0 means the particle has no parent
        # id::Int = 0 # parent of the particle

    end



    # Select a node in the M \times M grid at random 
    # with rate = 100 and observe the state of the node with some error 
    # For example, if X[i, j] = 1 (infected) then the observtion is correct with probability 0.80
    # and if X[i, j] = 0 (not infected) then the observation is correct with probability 0.95

    function initialize_particles_as_vectors(num_particles, grid_params, model_params)
        particles = Vector{Particle}()
        for i ∈ 1:num_particles
            initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode="fixed_probability")
            push!(particles, Particle(state = initial_state, weight = 1.0))
        end
        return particles
    end
    function biased_initialize_particles_as_vectors(num_particles, grid_params, model_params)
        particles = Vector{Particle}()
        for i ∈ 1:3
            initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode="fixed_probability")
            push!(particles, Particle(state = initial_state, weight = 1.0))
        end
        for i ∈ 4:num_particles
            initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode="fixed_probability")
            push!(particles, Particle(state=initial_state, weight=10.0))
        end
        return particles
    end


    function initialize_particles_as_hashmap(num_particles, grid_params, model_params)::Dict{Int,Vector{Particle}}
        particles = Dict{Int,Vector{Particle}}()
        for i ∈ 1:num_particles
            initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode="complete_chaos")
            particles[i] = [Particle(state = initial_state, weight = 1.0)]
        end
        return particles
    end

    function initialize_particles_with_signals_as_hashmap(num_particles, grid_params, model_params, signal_state, signal_rate)::Dict{Int,Vector{Particle}}
        particles = Dict{Int,Vector{Particle}}()
        for i ∈ 1:Int(num_particles)
            initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode="fixed_probability")
            particles[i] = [Particle(state = initial_state, weight = 1.0)]
        end
        return particles
    end

    function observe_node(state, i, j; infection_error_rate = 0.80, recovery_error_rate = 0.95)
        if state[i, j] == true # if the node is infected
            return rand() < infection_error_rate
        else # if the node is not infected
            # return true with probability 0.05
            return rand() < 1 - recovery_error_rate
        end
    end

    function observe_state(state; infection_error_rate = 0.80, recovery_error_rate = 0.95, rate = 100)
        # select a node at random and observe the state of the node
        i, j = rand(2:(size(state, 1) - 1)), rand(2:(size(state, 2)-1))
        time = rand(Exponential(1/rate)) 
        observed_state = (observe_node(state, i, j, infection_error_rate = infection_error_rate, recovery_error_rate = recovery_error_rate), (i ,j))

        return observed_state, time
    end



    function get_observations_from_state_sequence(state_sequence, time_limit, transition_times; infection_error_rate = 0.80, recovery_error_rate = 0.95, rate = 100, testing_mode = false)
        observations = []
        times = []
        current_time = 0.0
        while current_time < time_limit
            if current_time == 0.0
                observed_state, time = observe_state(state_sequence[1], infection_error_rate=infection_error_rate, recovery_error_rate=recovery_error_rate, rate=rate)
            else    
                observed_state, time = observe_state(state_sequence[findlast(x -> x <= current_time, transition_times)], infection_error_rate=infection_error_rate, recovery_error_rate=recovery_error_rate, rate=rate)
            end
            push!(observations, observed_state)
            push!(times, time + current_time)
            current_time = times[end]
        end
        return observations, times
    end

    function get_observations(state, time_limit; infection_error_rate = 0.80, recovery_error_rate = 0.95, rate = 100)
        observations = []
        times = []
        current_time = 0
        while current_time < time_limit
            observed_state, time = observe_state(state, infection_error_rate = infection_error_rate, recovery_error_rate = recovery_error_rate, rate = rate)
            push!(observations, observed_state)
            # if times is empty then push the time otherwise push the time + the last element in times (write a ternary if-else)
            push!(times, isempty(times) ? time : time + times[end])
            current_time = times[end]
        end
        return observations, times
    end

    function likelihood(hidden_state, observed_state)
        # q_bar is the canonical probability of the observed state given the hidden state which is 1/2
        if hidden_state[observed_state[2][1], observed_state[2][2]] == true
            if observed_state[1] == true
                return 0.8
            else
                return 0.2
            end
        end 

        if observed_state[1] == true
            return 0.05
        else
            return 0.95
        end
        
    end

    function count_total_particles(particle_history, time_stamp)
        total_particles = 0
        time_stamp = Float32(time_stamp)
        for key in keys(particle_history[Float32(time_stamp)])
            total_particles += length(particle_history[time_stamp][key])
        end
        return total_particles
    end

    function count_total_particles(particles)
        return length(particles)
    end

    function save_particle_history(particles, time_stamp)
        time_stamp = Float32(time_stamp)
        save("$dir_path/particle_history_$(time_stamp).jld2", "particles", particles)
    end
    # Let's initialize states and rates and observe the state

    # instead of storing the particle_history as a dictionary, write the particle_history to a file
    # Also, instead of storing the particle_history as a dictionary, store it as a vector
    function branching_particle_filter(initial_num_particles, grid_params, model_params, observation_time_stamps, observations ; r=1.5, signal_state=zeros(Bool, 12, 12), signal_rate=100.0, testing_mode = false)

        t_curr, t_next = 0, 0
        initial_particles = initialize_particles_as_vectors(initial_num_particles, grid_params, model_params)
        # particle_history = Dict{Float32,Dict{Int,Vector{Particle}}}()
        # Use a vector{vector{Particle}} instead of a dictionary
        # particle_history = Vector{Vector{Particle}}()
        # particle_history = Dict{Float32,Vector{Particle}}() # use a vector instead of a dictionary   
        new_particles = deepcopy(initial_particles)
        average_weights = sum([particle.weight for particle in new_particles]) / initial_num_particles
        progress = Progress(length(observation_time_stamps), 1, "Starting the particle filter...")

        for i ∈ eachindex(observation_time_stamps)
            next!(progress)
            if i != length(observation_time_stamps)
                t_curr = t_next
                t_next = observation_time_stamps[i]
            else
                t_curr = t_next
                t_next = model_params.time_limit
            end

            new_model_params = ContactProcess.ModelParameters(infection_rate=0.05, recovery_rate=0.1, time_limit=t_next - t_curr, prob_infections=0.05, num_simulations=1000) # rates are defined to be per day
            total_particles = count_total_particles(new_particles)
    

            for j ∈ 1:total_particles
                state = deepcopy(new_particles[j].state)
                rates = ContactProcess.calculate_all_rates(state, grid_params, new_model_params)
                X_sequence, _, _ = ContactProcess.run_simulation!(state, rates, grid_params, new_model_params)
                new_particles[j].state = copy(X_sequence[end])

                if average_weights > exp(20) || average_weights < exp(-20)
                    new_particles[j].weight = new_particles[j].weight * likelihood(X_sequence[end], observations[i]) * 2 / average_weights
                    # new_particles[j].weight = 2 / average_weights
                else
                    new_particles[j].weight = new_particles[j].weight * likelihood(X_sequence[end], observations[i]) * 2
                    # new_particles[j].weight = 2
                end
                if testing_mode == true
                    open("$dir_path/output.txt", "a") do file
                        println(file, "Particle weight: ", new_particles[j].weight, " for $j th particle at time $t_next")
                    end
                end
            end

            average_weights = sum([particle.weight for particle in new_particles]) / initial_num_particles
            open("$dir_path/output.txt", "a") do file
                println(file, "Average weights at time $t_next: ", average_weights)
                println(file, "Total Particles at the beginning of the iteration $t_next: ", count_total_particles(new_particles))
            end
            temp_particles = deepcopy(new_particles)
            deleted_particles = 0
            total_offsprings = 0
            for j ∈ 1:total_particles
                V = 0.2 * rand() - 0.1
                if ((average_weights / r <= new_particles[j].weight + V) && (new_particles[j].weight + V <= average_weights * r)) == false # To branch
                    num_offspring = flag = floor(Int, new_particles[j].weight / average_weights)
                    # println("The Bernoulli parameter is:", (new_particles[j].weight / average_weights) - num_offspring)
                    num_offspring += rand(Bernoulli((new_particles[j].weight / average_weights) - num_offspring))
                    temp_particles[j - deleted_particles].weight = deepcopy(average_weights)
                    if num_offspring > 0 # branch the particle
                        # println("Branching particle: ", j, " with ", num_offspring, " offsprings")
                        if flag == 0
                            if testing_mode == true
                                open("$dir_path/output.txt", "a") do file
                                    println(file, "Killing candidate ", j, " but not killed with weight: ", new_particles[j].weight/average_weights)
                                end
                            end
                        end 
                        for _ ∈ 1:(num_offspring-1)
                            push!(temp_particles, deepcopy(new_particles[j]))
                            
                        end
                        total_offsprings += num_offspring - 1
                    
                    else # kill the particle
                        # println("Killing particle: ", j)
                        deleteat!(temp_particles, j - deleted_particles)
                        # println("Total Particles: ", count_total_particles(temp_particles))
                        if testing_mode == true
                            open("$dir_path/output.txt", "a") do file
                                println(file, "Killing particle: ", j, " with weight: ", new_particles[j].weight/average_weights , " at time $t_next")
                            end
                        end

                        deleted_particles += 1
                    end

                end
            end

            new_particles = deepcopy(temp_particles) # update the particles
            if testing_mode == true
                open("$dir_path/output.txt", "a") do file
                    println(file, "Destroyed particles: ", deleted_particles, " Total offsprings: ", total_offsprings)
                    println(file, "Total Particles at the end of the iteration $i: ", count_total_particles(new_particles))
                end
            end
            # push!(particle_history, new_particles)
            # save the particle history to a file and don't push it to the particle history
            save_particle_history(new_particles, Float32(t_next))

        end
        return
    end

end # module Observations


