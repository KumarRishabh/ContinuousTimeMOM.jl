using Distributed
@everywhere using PrettyTables
@everywhere using Revise
@everywhere using Plots
@everywhere include("../src/ContactProcess.jl")
@everywhere using .ContactProcess
@everywhere using Random
@everywhere using ProgressMeter

# Set the number of workers
addprocs(4)  # Replace 4 with the desired number of workers

# Set the seed for reproducibility
@everywhere Random.seed!(1234)

# Set b, d with the same values as in the paper
grid_params = ContactProcess.GridParameters(width = 200, height = 200) 
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day 

# Initialize state and rates
@everywhere state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
pretty_table(state)

# Run the simulation
@everywhere state_sequence, times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, model_params)

# Generate the animation
@everywhere anim = ContactProcess.generate_animation(state_sequence, times)

# Save the animation
@everywhere gif(anim, "contact_process.gif", fps = 10)

# Parallelize multiple simulations
@everywhere all_state_sequences, all_times, all_updated_nodes = @distributed (vcat) for i in 1:model_params.num_simulations
    state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
    state_sequence, times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, model_params)
    (state_sequence, times, updated_nodes)
end
