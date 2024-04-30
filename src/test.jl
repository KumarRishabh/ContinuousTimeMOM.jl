import Pkg; Pkg.add("Revise")
using Revise
include("ContactProcess.jl")
using .ContactProcess



# Run the simulation and plot each step

# Initialize the game state with random values
num_infections = 10

# times = sample_time_with_rates(state, rates)
height, width = 50, 50
width_with_padding, height_with_padding = width + 2, height + 2
state = zeros(Bool, width_with_padding, height_with_padding)
rates = zeros(width_with_padding, height_with_padding)
state, rates = ContactProcess.initialize_state_and_rates(state, rates, num_infections)
state, rates = Base.invokelatest(ContactProcess.initialize_state_and_rates(num_infections))