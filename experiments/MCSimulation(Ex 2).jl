using Pkg; Pkg.activate(".")
using PrettyTables
using Revise
using Plots
include("../src/ContactProcess.jl")
using .ContactProcess
using Random
using ProgressMeter

# simulate the contact process for height, width = 200, 200
# set the seed for reproducibility
# time_limit - 500
# infection_rate = 0.05
# recovery_rate = 0.1
# num_simulations = 100_000
function sum_edges(matrix)
    return sum(matrix[2, :]) + sum(matrix[end-1, :]) + sum(matrix[:, 2]) + sum(matrix[:, end-1])
end

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
        loglikelihoods = push!(loglikelihoods, last_loglikelihood + (0.3 - model_params.recovery_rate - 4 * model_params.infection_rate) * bar_X * δt - (0.05 - model_params.infection_rate) * hat_X * δt)
        # if the state is updated to 1 then add to the loglikelihood log(model_params["infection_rate"]/0.05) else add log(model_params["infection_rate"]/0.01)
        # println(state_sequence)
        if state_sequence[i][updated_node[1], updated_node[2]] == 1
            loglikelihoods[end] += log(model_params.infection_rate / 0.05) # birth
        else
            loglikelihoods[end] += log(model_params.recovery_rate / 0.1) # death
        end
        time = new_time
        bar_X = sum(state_sequence[i])
        hat_X = sum_edges(state_sequence[i])
    end
    return loglikelihoods
end



Random.seed!(1234)
grid_params = ContactProcess.GridParameters(width = 20, height = 20)
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 500, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

all_state_sequences, all_times, all_updated_nodes = ContactProcess.multiple_simulations(grid_params, model_params)
