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

Random.seed!(1234)
grid_params = ContactProcess.GridParameters(width = 20, height = 20)
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 500, prob_infections = 0.01, num_simulations = 100) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

all_state_sequences, all_times, all_updated_nodes = ContactProcess.multiple_simulations(grid_params, model_params)
