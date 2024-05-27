import Pkg; Pkg.add("Revise"); Pkg.add("PrettyTables")
using PrettyTables
using Revise
using Plots
include("../src/ContactProcess.jl")
using .ContactProcess

# Run the simulation and plot each step

# Initialize the game state with random values

grid_params = ContactProcess.GridParameters(width = 20, height = 20)
model_params = ContactProcess.ModelParameters(infection_rate = 2, recovery_rate = 0.5, num_steps = 50, prob_infections = 0.01)
# times = sample_time_with_rates(state, rates)

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)

p = heatmap(state, color=:grays, legend=false)
# Run the simulation
num_steps = 200
# pretty_table print the state with a string saying that it is the initial state
print("Initial State: ")
pretty_table(state)
state_sequence, times = ContactProcess.run_simulation!(state, rates, num_steps,  ; debug_mode = true)


# plot the state_sequence using the heatmap in a loop to show the spread of the infection
p = heatmap(state_sequence[1], color=:grays, legend=false)
# Issue: Why isn't the state_sequence gets updated in the loop?
# Solution: The state_sequence is not getting updated because the state is not being updated in the loop.
num_steps = 50
anim = @animate for i ∈ 1:num_steps
    heatmap!(p, state_sequence[i], color=:grays, legend=false)
    # sleep(1)
end

gif(anim, "contact_process.gif", fps=1)



# check if all states in the state sequence are different


