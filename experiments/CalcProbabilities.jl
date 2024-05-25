using PrettyTables
using Revise
using Plots
using Random
using ProgressMeter

include("../src/ContactProcess.jl")
using .ContactProcess

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


model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 500, prob_infections = 0.05, num_simulations = 1000) # rates are defined to be per day
grid_params = ContactProcess.GridParameters(width = 20, height = 20)

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
all_state_sequences, all_times, all_updated_nodes = ContactProcess.multiple_simulations(grid_params, model_params)

model_params_1 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.1001, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day
model_params_2 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.1001, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day
model_params_3 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.0999, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day
model_params_4 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.0999, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day

# compute weights for each model parameters
# what is the size of the all_state_sequences variable? 

loglikelihoods_with_model_params_1 = []
loglikelihoods_with_model_params_2 = []
loglikelihoods_with_model_params_3 = []
loglikelihoods_with_model_params_4 = []
P = Progress(length(all_state_sequences), 1, "Computing loglikelihoods...")

for i ∈ eachindex(all_state_sequences)
    next!(P)
    loglikelihood = compute_loglikelihood(model_params_1, all_state_sequences[i], all_times[i], all_updated_nodes[i])
    push!(loglikelihoods_with_model_params_1, loglikelihood[end])
end
# println(loglikelihoods_with_model_params_1)
P = Progress(length(all_state_sequences), 1, "Computing loglikelihoods...")
for i ∈ eachindex(all_state_sequences)
    next!(P)
    loglikelihood = compute_loglikelihood(model_params_2, all_state_sequences[i], all_times[i], all_updated_nodes[i])
    push!(loglikelihoods_with_model_params_2, loglikelihood[end])
end

P = Progress(length(all_state_sequences), 1, "Computing loglikelihoods...")
for i ∈ eachindex(all_state_sequences)
    next!(P)
    loglikelihood = compute_loglikelihood(model_params_3, all_state_sequences[i], all_times[i], all_updated_nodes[i])
    push!(loglikelihoods_with_model_params_3, loglikelihood[end])
end
print(loglikelihoods_with_model_params_3)
P = Progress(length(all_state_sequences), 1, "Computing loglikelihoods...")
for i ∈ eachindex(all_state_sequences)
    next!(P)
    loglikelihood = compute_loglikelihood(model_params_4, all_state_sequences[i], all_times[i], all_updated_nodes[i])
    push!(loglikelihoods_with_model_params_4, loglikelihood[end])
end


println("loglikelihoods_with_model_params_1: ", loglikelihoods_with_model_params_1)
println("loglikelihoods_with_model_params_2: ", loglikelihoods_with_model_params_2)
println("loglikelihoods_with_model_params_3: ", loglikelihoods_with_model_params_3)
println("loglikelihoods_with_model_params_4: ", loglikelihoods_with_model_params_4)


function compute_probabilities(loglikelihoods, model_params, grid_params)
    estimate = 0
    for i ∈ 1:model_params.num_simulations
        # if check_edges > 10 add exp(loglikelihood[i]) to the estimate else add
        if sum(all_state_sequences[1][i]) ≤ 10
            estimate += exp(loglikelihoods[i])
        end
    end
    return estimate/model_params.num_simulations
end

prob_1 = compute_probabilities(loglikelihoods_with_model_params_1, model_params_1, grid_params)
prob_2 = compute_probabilities(loglikelihoods_with_model_params_2, model_params_2, grid_params)
prob_3 = compute_probabilities(loglikelihoods_with_model_params_3, model_params_3, grid_params)
prob_4 = compute_probabilities(loglikelihoods_with_model_params_4, model_params_4, grid_params)


loglikelihoods_2 = compute_loglikelihood(model_params_2, all_state_sequences[1], all_times[1], all_updated_nodes[1])
loglikelihoods_3 = compute_loglikelihood(model_params_3, all_state_sequences[1], all_times[1], all_updated_nodes[1])
loglikelihoods_4 = compute_loglikelihood(model_params_4, all_state_sequences[1], all_times[1], all_updated_nodes[1])


# test functions
