using PrettyTables
using Revise
using Plots
include("../src/ContactProcess.jl")
using .ContactProcess
using Random
# set the seed for reproducibility

Random.seed!(1234)

# set b, d with the same values as in the paper
grid_params = ContactProcess.GridParameters(width = 20, height = 20) 
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 1, prob_infections = 0.01, num_simulations = 20) # rates are defined to be per day 


state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
pretty_table(state)

# Run the simulation
state_sequence, times = ContactProcess.run_simulation!(state, rates, grid_params, model_params)

# Generate the animation
anim = ContactProcess.generate_animation(state_sequence, times)

# Save the animation
gif(anim, "contact_process.gif", fps = 10)

all_state_sequences, all_times = ContactProcess.multiple_simulations(grid_params, model_params)

# plot all the simulations
for simulation ∈ all_state_sequences
    plot(simulation, legend = false)
end 