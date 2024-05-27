using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter
using FileIO
include("../src/ContactProcess.jl")
using .ContactProcess
using Base.Threads

Random.seed!(10)

#=We consider the same contact process disease spread model as the prior
example but now with M = 200 and time up to T = 500 days. Using the same four different
possibilities of b and d but still only simulating the one average one where b = 0.05 and d =
0.1, we are interested in estimating the probability P(XT ≤ 10). We will now use 100, 000
samples but, instead of resampling, we just weight.=#

function sum_edges(matrix)
    return sum(matrix[2, :]) + sum(matrix[end-1, :]) + sum(matrix[:, 2]) + sum(matrix[:, end-1])
end

"""
    computing the likelihood of the data given the following model settings: 
    1.) infection_rate, recovery_rate = 0.0499, 0.1001
    2.) infection_rate, recovery_rate = 0.0501, 0.1001
    3.) infection_rate, recovery_rate = 0.0499, 0.0999
    4.) infection_rate, recovery_rate = 0.0501, 0.0999

    The function returns the likelihood (along the time-series/path 📈) of the data given the model settings
"""
function compute_loglikelihood(model_params, state_sequence, times, updated_nodes)
    loglikelihoods = [0.0] # likelihood computed at jump times
    bar_X = sum(state_sequence[1])
    hat_X = sum_edges(state_sequence[1])
    time = times[1]
    for i ∈ 2:length(state_sequence) # start from 2 because the first element is the initial element which is a placeholder
        updated_node = updated_nodes[i]
        if i == length(state_sequence)
            new_time = model_params.time_limit
        else
            new_time = times[i]
        end
        last_loglikelihood = loglikelihoods[end]
        δt = new_time - time
        # println(loglikelihoods, (0.3 - model_params.recovery_rate - 4 * model_params.recovery_rate) * bar_X * (new_time - time) - (0.05 - model_params.recovery_rate) * hat_X * (new_time - time))
        loglikelihoods = push!(loglikelihoods, last_loglikelihood + (0.3 - model_params.recovery_rate - 4 * model_params.infection_rate) * bar_X * δt - (0.05 - model_params.infection_rate) * hat_X * δt)
        # if the state is updated to 1 then add to the loglikelihood log(model_params["infection_rate"]/0.05) else add log(model_params["infection_rate"]/0.01)
        # println(state_sequence)
        if state_sequence[i][updated_node[1], updated_node[2]] == 1 && i != length(state_sequence)
            loglikelihoods[end] += log(model_params.infection_rate / 0.05) # birth
        elseif state_sequence[i][updated_node[1], updated_node[2]] == 0 && i != length(state_sequence)
            loglikelihoods[end] += log(model_params.recovery_rate / 0.1) # death
        end
        time = new_time
        bar_X = sum(state_sequence[i])
        hat_X = sum_edges(state_sequence[i])
    end
    return loglikelihoods
end


model_params = ContactProcess.ModelParameters(infection_rate=0.05, recovery_rate=0.1, time_limit=500, prob_infections=0.05, num_simulations=10_000) # rates are defined to be per day
grid_params = ContactProcess.GridParameters(width=20, height=20)

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
# ContactProcess.save_multiple_simulations(grid_params, model_params) # TURN ON CAREFULLY

model_params_1 = ContactProcess.ModelParameters(infection_rate=0.0499, recovery_rate=0.1001, time_limit=500, prob_infections=model_params.prob_infections, num_simulations=model_params.num_simulations) # rates are defined to be per day
model_params_2 = ContactProcess.ModelParameters(infection_rate=0.0501, recovery_rate=0.1001, time_limit=500, prob_infections=model_params.prob_infections, num_simulations=model_params.num_simulations) # rates are defined to be per day
model_params_3 = ContactProcess.ModelParameters(infection_rate=0.0499, recovery_rate=0.0999, time_limit=500, prob_infections=model_params.prob_infections, num_simulations=model_params.num_simulations) # rates are defined to be per day
model_params_4 = ContactProcess.ModelParameters(infection_rate=0.0501, recovery_rate=0.0999, time_limit=500, prob_infections=model_params.prob_infections, num_simulations=model_params.num_simulations) # rates are defined to be per day

# compute weights for each model parameters
# what is the size of the all_state_sequences variable? 
dir_path = joinpath(@__DIR__, "./data")

loglikelihoods_with_model_params_1 = []
loglikelihoods_with_model_params_2 = []
loglikelihoods_with_model_params_3 = []
loglikelihoods_with_model_params_4 = []
# Always ignore the first element
P = Progress(model_params.num_simulations, 1, "Computing loglikelihoods...")
# the first element is 0.0

initial_state = load("$dir_path/initial_state_1.jld", "initial_state")
updated_nodes = load("$dir_path/updated_nodes_1.jld", "updated_nodes") # always
times = load("$dir_path/times_1.jld", "times")
state_sequence = ContactProcess.make_state_sequence(initial_state, updated_nodes, grid_params, model_params)

prob_estimate_1 = 0
prob_estimate_2 = 0
prob_estimate_3 = 0
prob_estimate_4 = 0

for i ∈ 1:model_params.num_simulations
    initial_state = load("$dir_path/initial_state_$i.jld", "initial_state")
    updated_nodes = load("$dir_path/updated_nodes_$i.jld", "updated_nodes") # always
    times = load("$dir_path/times_$i.jld", "times")

    state_sequence = ContactProcess.make_state_sequence(initial_state, updated_nodes, grid_params, model_params)

    loglikelihood1 = compute_loglikelihood(model_params_1, state_sequence, times, updated_nodes)
    loglikelihood2 = compute_loglikelihood(model_params_2, state_sequence, times, updated_nodes)
    loglikelihood3 = compute_loglikelihood(model_params_3, state_sequence, times, updated_nodes)
    loglikelihood4 = compute_loglikelihood(model_params_4, state_sequence, times, updated_nodes)
    
    push!(loglikelihoods_with_model_params_1, loglikelihood1[end])
    push!(loglikelihoods_with_model_params_2, loglikelihood2[end])
    push!(loglikelihoods_with_model_params_3, loglikelihood3[end])
    push!(loglikelihoods_with_model_params_4, loglikelihood4[end])

    if sum(state_sequence[end]) ≤ 10
        prob_estimate_1 += exp(loglikelihood1[end])
        prob_estimate_2 += exp(loglikelihood2[end])
        prob_estimate_3 += exp(loglikelihood3[end])
        prob_estimate_4 += exp(loglikelihood4[end])
    end

    next!(P)
end

prob_estimate_1 /= model_params.num_simulations
prob_estimate_2 /= model_params.num_simulations
prob_estimate_3 /= model_params.num_simulations
prob_estimate_4 /= model_params.num_simulations

# println(loglikelihoods_with_model_params_1)

println("loglikelihoods_with_model_params_1: ", loglikelihoods_with_model_params_1)
println("loglikelihoods_with_model_params_2: ", loglikelihoods_with_model_params_2)
println("loglikelihoods_with_model_params_3: ", loglikelihoods_with_model_params_3)
println("loglikelihoods_with_model_params_4: ", loglikelihoods_with_model_params_4)


println("prob_estimate_1: ", prob_estimate_1)
println("prob_estimate_2: ", prob_estimate_2)
println("prob_estimate_3: ", prob_estimate_3)
println("prob_estimate_4: ", prob_estimate_4)




# test functions
