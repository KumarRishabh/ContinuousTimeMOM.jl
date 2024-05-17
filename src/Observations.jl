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

# Select a node in the M \times M grid at random 
# with rate = 100 and observe the state of the node with some error 
# For example, if X[i, j] = 1 (infected) then the observtion is correct with probability 0.80
# and if X[i, j] = 0 (not infected) then the observation is correct with probability 0.95

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
    time = rand(Exponential(rate)) 
    observed_state = (observe_node(state, i, j, infection_error_rate = infection_error_rate, recovery_error_rate = recovery_error_rate), (i ,j))

    return observed_state, time
end

function get_observations(state, num_observations; infection_error_rate = 0.80, recovery_error_rate = 0.95, rate = 100)
    observations = []
    times = []
    for i in 1:num_observations
        observed_state, time = observe_state(state, infection_error_rate = infection_error_rate, recovery_error_rate = recovery_error_rate, rate = rate)
        push!(observations, observed_state)
        # if times is empty then push the time otherwise push the time + the last element in times (write a ternary if-else)
        push!(times, isempty(times) ? time : time + times[end])

    end
    return observations, times
end
# Let's initialize states and rates and observe the state

grid_params = ContactProcess.GridParameters(width = 20, height = 20)
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 10, prob_infections = 0.05, num_simulations = 1000) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

observed_state, time = observe_state(state)
observations, observation_time_stamps = get_observations(state, 100)
i, j = rand(2:(size(state, 1) - 1)), rand(2:(size(state, 2)-1))

# plot 100 simulations from Exponential distribution with rate 100

# plot the histogram of the 100 simulations
histogram(rand(Exponential(100), 100), bins = 20, label = "Exponential(100)", xlabel = "x", ylabel = "Frequency", title = "Histogram of 100 simulations from Exponential(100)", legend = :topleft)