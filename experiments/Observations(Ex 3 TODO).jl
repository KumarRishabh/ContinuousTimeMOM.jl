using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter
# using JLD2 # for saving and loading the data efficiently
include("../src/ContactProcess.jl")
using .ContactProcess
using JLD2
using Distributions
Random.seed!(10)
using Parameters


@with_kw struct Particle
    # state of the particle is defined as a 2D array with the same dimensions as the grid
    state::Array{Bool, 2} = zeros(Bool, 20, 20) # 20 x 20 grid with all zeros
    weight::Float64 = 1.0
end
state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode = "complete_chaos")
p₁ = Particle(state = state, weight = 1.0)
particles = Dict{Matrix{Bool}, Vector{Particle}}()
initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params, mode = "complete_chaos")
for i ∈ 1:100
    particles[i] = [Particle(state = initial_state, weight = 1.0)]
end
# Select a node in the M \times M grid at random 
# with rate = 100 and observe the state of the node with some error 
# For example, if X[i, j] = 1 (infected) then the observtion is correct with probability 0.80
# and if X[i, j] = 0 (not infected) then the observation is correct with probability 0.95
function initialize_particles(num_particles, grid_params, model_params)::Dict{Array{Bool,2},Vector{Particle}}
    particles = Dict{Array{Bool,2},Vector{Particle}}()
    initial_state, initial_rate = ContactProcess.initialize_state_and_rates(grid_params, model_params)
    for i ∈ 1:num_particles
        particles[i] = [Particle(state = initial_state, 1.0)]
    end
    return particles
end

function observe_node(state, i, j; infection_error_rate = 0.80, recovery_error_rate = 0.95)
    if state[i, j] == 1
        return rand() < infection_error_rate
    else
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
    if hidden_state == 1
        return hidden_state[observed_state[2][1], observed_state[2][2]] ? 0.80 : 0.20
    else
        return hidden_state[observed_state[2][1], observed_state[2][2]] ? 0.05 : 0.95
    end
end
# Let's initialize states and rates and observe the state

grid_params = ContactProcess.GridParameters(width = 20, height = 20)
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 10, prob_infections = 0.05, num_simulations = 1000) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params; mode = "complete_chaos")

observed_state, time = observe_state(state)
observations, observation_time_stamps = get_observations(state, 100)
num_particles = [100]
# between the observation time stamps, simulate the contact process with state and rates
t_curr, t_next = 0, 0
V = 0.2*rand() - 0.1
# need to create a data structure to associate how many particles are at each observation time stamp
# since the number of particles can change at each observation time stamp due to branching process 
# we need to store the number of particles at each observation time stamp
# particles = {particle_1 => [instance_1, instance_2, instance_3], particle_2 => [instance_1, instance_2, instance_3, instance_4]} for 100 particles


particles = initialize_particles(num_particles[1], grid_params, model_params)
for i ∈ eachindex(observation_time_stamps)
    for j ∈ num_particles[1]
        if i != length(observation_time_stamps)
            t_curr = t_next
            t_next = observation_time_stamps[i]
        else 
            t_curr = t_next
            t_next = model_params.time_limit
        end

        # TODO: Implement the branching process
        new_model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = t_next, prob_infections = 0.05, num_simulations = 1000) # rates are defined to be per day
        state = particles[j][end].state
        rates = ContactProcess.calculate_all_rates(state, grid_params, new_model_params)
        X_sequence, times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, new_model_params)
        # update the particle weight
        for k in eachindex(particles[j])
            particles[j][k].weight = particles[j][k].weight * ContactProcess.compute_likelihood(X_sequence, observations[i][1], updated_nodes, observations[i][2])
        end

        # simulate the contact process with the state and rates
        # between the t_curr and t_next
        
        # update the state and rates
        # update the particles


            # simulate the contact process with the state and rates
            # between the observation_time_stamps[i] and observation_time_stamps[j]
            # update the state and rates
    end
end

