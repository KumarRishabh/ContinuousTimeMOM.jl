using Pkg; Pkg.activate(".")
using PrettyTables
using Revise
using Plots
include("../src/ContactProcess.jl")
using .ContactProcess
using Random
using ProgressMeter
using JLD2
using FileIO
dir_path = joinpath(@__DIR__)

# simulate the contact process for height, width = 200, 200
# set the seed for reproducibility
# time_limit - 500
# infection_rate = 0.05
# recovery_rate = 0.1
# num_simulations = 100_000
function sum_edges(matrix)
    return sum(matrix[2, :]) + sum(matrix[end-1, :]) + sum(matrix[:, 2]) + sum(matrix[:, end-1])
end

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
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 500, prob_infections = 0.01, num_simulations = 1000) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

ContactProcess.multiple_simulations(grid_params, model_params)

MC_estimate = 0.0

# check if the sum(state) ≤ 10 
function check_state(state)
    return sum(state) ≤ 10
end
model_params_1 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.1001, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day

for i ∈ 1:model_params.num_simulations
    state_sequence = load("$dir_path/data/state_sequences_$i.jld")["interim_state_sequences"]
    times = load("$dir_path/data/times_$i.jld")["interim_times"]
    updated_nodes = load("$dir_path/data/updated_nodes_$i.jld")["interim_updated_nodes"]
    # println("State Sequence: ", state_sequence)
    loglikelihoods = compute_loglikelihood(model_params_1, state_sequence, times, updated_nodes)
    MC_estimate += check_state(state_sequence[end]) ? exp(loglikelihoods[end]) : 0
    println("Loglikelihood for simulation $i: ", loglikelihoods[end])
end

MC_estimate /= model_params.num_simulations
println("MC estimate for P(X̄_T ≤ 10 | b = 0.0499, d = 0.1001 ): ", MC_estimate)

model_params_2 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.1001, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day

MC_estimate = 0.0

for i ∈ 1:model_params.num_simulations
    state_sequence = load("$dir_path/data/state_sequences_$i.jld")["interim_state_sequences"]
    times = load("$dir_path/data/times_$i.jld")["interim_times"]
    updated_nodes = load("$dir_path/data/updated_nodes_$i.jld")["interim_updated_nodes"]
    loglikelihoods = compute_loglikelihood(model_params_2, state_sequence, times, updated_nodes)
    MC_estimate += check_state(state_sequence[end]) ? exp(loglikelihoods[end]) : 0
    println("Loglikelihood for simulation $i: ", loglikelihoods[end])
end

MC_estimate /= model_params.num_simulations
println("MC estimate for P(X̄_T ≤ 10 | b = 0.0501, d = 0.1001 ): ", MC_estimate)

model_params_3 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.0999, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day

MC_estimate = 0.0

for i ∈ 1:model_params.num_simulations
    state_sequence = load("$dir_path/data/state_sequences_$i.jld")["interim_state_sequences"]
    times = load("$dir_path/data/times_$i.jld")["interim_times"]
    updated_nodes = load("$dir_path/data/updated_nodes_$i.jld")["interim_updated_nodes"]
    loglikelihoods = compute_loglikelihood(model_params_3, state_sequence, times, updated_nodes)
    MC_estimate += check_state(state_sequence[end]) ? exp(loglikelihoods[end]) : 0
    println("Loglikelihood for simulation $i: ", loglikelihoods[end])
end

MC_estimate /= model_params.num_simulations
println("MC estimate for P(X̄_T ≤ 10 | b = 0.0499, d = 0.0999 ): ", MC_estimate)

model_params_4 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.0999, time_limit = 500, prob_infections = 0.01, num_simulations = model_params.num_simulations) # rates are defined to be per day

MC_estimate = 0.0

for i ∈ 1:model_params.num_simulations
    state_sequence = load("$dir_path/data/state_sequences_$i.jld")["interim_state_sequences"]
    times = load("$dir_path/data/times_$i.jld")["interim_times"]
    updated_nodes = load("$dir_path/data/updated_nodes_$i.jld")["interim_updated_nodes"]
    loglikelihoods = compute_loglikelihood(model_params_4, state_sequence, times, updated_nodes)
    MC_estimate += check_state(state_sequence[end]) ? exp(loglikelihoods[end]) : 0
    println("Loglikelihood for simulation $i: ", loglikelihoods[end])
end

MC_estimate /= model_params.num_simulations
println("MC estimate for P(X̄_T ≤ 10 | b = 0.0501, d = 0.0999 ): ", MC_estimate)


