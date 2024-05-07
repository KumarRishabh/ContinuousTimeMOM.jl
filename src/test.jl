import Pkg; Pkg.add("Revise"); Pkg.add("PrettyTables")
using PrettyTables
using Revise
using Plots
include("ContactProcess.jl")
using .ContactProcess

# Run the simulation and plot each step

# Initialize the game state with random values
num_infections = 2
# times = sample_time_with_rates(state, rates)
height, width = 6, 6
width_with_padding, height_with_padding = width + 2, height + 2
state = zeros(Bool, width_with_padding, height_with_padding)
rates = zeros(width_with_padding, height_with_padding)
state, rates = ContactProcess.initialize_state_and_rates(num_infections)

p = heatmap(state, color=:grays, legend=false)
# Run the simulation
num_steps = 20
# pretty_table print the state with a string saying that it is the initial state
print("Initial State: ")
pretty_table(state)
state_sequence, times = ContactProcess.run_simulation!(state, rates, num_steps; debug_mode = true)
print("Updated State: ")
pretty_table(state_sequence[1])
# plot the state_sequence using the heatmap in a loop to show the spread of the infection
p = heatmap(state_sequence[1], color=:grays, legend=false)
# Issue: Why isn't the state_sequence gets updated in the loop?
# Solution: The state_sequence is not getting updated because the state is not being updated in the loop.

anim = @animate for i in 1:num_steps
    heatmap!(p, state_sequence[i], color=:grays, legend=false)
    # sleep(1)
end


anim = ContactProcess.generate_animation(state_sequence, times)

gif(anim, "contact_process.gif", fps=1)

# check if all states in the state sequence are different
