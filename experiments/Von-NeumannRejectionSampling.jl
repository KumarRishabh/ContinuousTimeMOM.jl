
# plot all the simulations one by one using gifs
using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter

include("../src/ContactProcess.jl")
using .ContactProcess

function sum_edges(matrix)
    return sum(matrix[2, :]) + sum(matrix[end - 1, :]) + sum(matrix[:, 2]) + sum(matrix[:, end - 1])
end

Random.seed!(10)
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 30, prob_infections = 0.05, num_simulations = 1000) # rates are defined to be per day
grid_params = ContactProcess.GridParameters(width = 20, height = 20)
state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

all_state_sequences, all_times, all_updated_nodes= ContactProcess.multiple_simulations(grid_params, model_params)

model_params_1 = ContactProcess.ModelParameters(infection_rate = 0.045, recovery_rate = 0.105, time_limit = 1, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day
model_params_2 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.1001, time_limit = 1, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day
model_params_3 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.0999, time_limit = 1, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day
model_params_4 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.0999, time_limit = 1, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day

# write a function to compute the sum of the edges of a matrix 


"""
    computing the likelihood of the data given the following model settings: 
    1.) infection_rate, recovery_rate = 0.0499, 0.1001
    2.) infection_rate, recovery_rate = 0.0501, 0.1001
    3.) infection_rate, recovery_rate = 0.0499, 0.0999
    4.) infection_rate, recovery_rate = 0.0501, 0.0999

    The function returns the likelihood of the data given the model settings
"""
function compute_loglikelihood(model_params, state_sequence, times, updated_nodes)
    loglikelihoods = [0.0] # likelihood computed at jump times
    bar_X = sum(state_sequence[1]) 
    hat_X = sum_edges(state_sequence[1])
    time = times[1]
    for i ∈ 2:length(state_sequence)
        updated_node = updated_nodes[i]
        if i == length(state_sequence)
            new_time = model_params.time_limit
        else
            new_time = times[i]
        end
        last_loglikelihood = loglikelihoods[end]
        δt = new_time - time
        # println(loglikelihoods, (0.3 - model_params.recovery_rate - 4 * model_params.recovery_rate) * bar_X * (new_time - time) - (0.05 - model_params.recovery_rate) * hat_X * (new_time - time))
        loglikelihoods = push!(loglikelihoods, last_loglikelihood + (0.3 - model_params.recovery_rate - 4*model_params.infection_rate) * bar_X * δt - (0.05 - model_params.infection_rate) * hat_X * δt)
        # if the state is updated to 1 then add to the loglikelihood log(model_params["infection_rate"]/0.05) else add log(model_params["infection_rate"]/0.01)
        # println(state_sequence)
        if state_sequence[i][updated_node[1], updated_node[2]] == 1
            loglikelihoods[end] += log(model_params.infection_rate/0.05) # birth
        else
            loglikelihoods[end] += log(model_params.recovery_rate/0.1) # death
        end
        time = new_time
        bar_X = sum(state_sequence[i])
        hat_X = sum_edges(state_sequence[i])
    end
    return loglikelihoods
end

# what is the shape of state_sequence

# permute the 20_000 simulations and do acceptance rejection sampling to the sample for 
# model_params_1, model_params_2, model_params_3, model_params_4 

# write a function to permute the simulations

param_1_samples = []
param_2_samples = []
param_3_samples = []
param_4_samples = []


progress = Progress(100, 1, "Computing loglikelihoods")
C = -Inf
for i ∈ 1:100
    # sample from the permuted indices
    next!(progress)
    state_sequence = all_state_sequences[i]
    times = all_times[i]
    # compute the loglikelihood for the current state sequence
    loglikelihoods = compute_loglikelihood(model_params_1, state_sequence, times, all_updated_nodes[i])
    # if likelihood is greater than the threshold then reject the sample
    println(loglikelihoods[end])
    if exp(loglikelihoods[end]) > C
        C = exp(loglikelihoods[end])
    end
end

println("Upper Bound: ", C)

C = -Inf
for i ∈ 1:100
    # sample from the permuted indices
    next!(progress)
    permuted_index = permuted_indices[i]
    state_sequence = all_state_sequences[permuted_index]
    times = all_times[permuted_index]
    # compute the loglikelihood for the current state sequence
    loglikelihoods = compute_loglikelihood(model_params_2, state_sequence, times, all_updated_nodes[permuted_index])
    # if likelihood is greater than the threshold then reject the sample
    # compute C such that it bounds the loglikelihoods
    if exp(loglikelihoods[end]) > C
        C = exp(loglikelihoods[end])
    end
end
println("Upper Bound: ", C)
# simulate uniform [0, C]



C = -Inf
for i ∈ 1:100
    # sample from the permuted indices
    next!(progress)
    permuted_index = permuted_indices[i]
    state_sequence = all_state_sequences[permuted_index]
    times = all_times[permuted_index]
    # compute the loglikelihood for the current state sequence
    loglikelihoods = compute_loglikelihood(model_params_3, state_sequence, times, all_updated_nodes[permuted_index])
    # if likelihood is greater than the threshold then reject the sample
    if exp(loglikelihoods[end]) > C
        C = exp(loglikelihoods[end])
    end
end
println("Upper Bound: ", C)

C = - Inf
for i ∈ 1:100
    # sample from the permuted indices
    next!(progress)
    permuted_index = permuted_indices[i]
    state_sequence = all_state_sequences[permuted_index]
    times = all_times[permuted_index]
    # compute the loglikelihood for the current state sequence
    loglikelihoods = compute_loglikelihood(model_params_4, state_sequence, times, all_updated_nodes[permuted_index])
    # if likelihood is greater than the threshold then reject the sample
    if exp(loglikelihoods[end]) > C
        C = exp(loglikelihoods[end])
    end

end
println("Upper Bound: ", C)


println(length(param_1_samples))
println(length(param_2_samples))
println(length(param_3_samples))
println(length(param_4_samples))

