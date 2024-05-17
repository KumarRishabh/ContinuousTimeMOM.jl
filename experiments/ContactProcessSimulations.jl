using PrettyTables
using Profile
using Revise
using Plots
include("../src/ContactProcess.jl")
using .ContactProcess
using Random
using ProgressMeter
# set the seed for reproducibility

Random.seed!(1234)

# set b, d with the same values as in the paper
grid_params = ContactProcess.GridParameters(width = 50, height = 50) 
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 4, prob_infections = 0.05, num_simulations = 20_000) # rates are defined to be per day 


state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
pretty_table(state)

# Run the simulation
state_sequence, times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, model_params)
      
# Generate the animation
anim = ContactProcess.generate_animation(state_sequence, times)

# Save the animation
@profile gif(anim, "contact_process.gif", fps = 10)

@profile all_state_sequences, all_times, all_updated_nodes= ContactProcess.multiple_simulations(grid_params, model_params)

# select a node at random 