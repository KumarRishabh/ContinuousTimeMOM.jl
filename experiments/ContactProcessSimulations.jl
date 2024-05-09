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
model_params = ContactProcess.ModelParameters(infection_rate = 0.05, recovery_rate = 0.1, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day 


state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params)
pretty_table(state)

# Run the simulation
state_sequence, times = ContactProcess.run_simulation!(state, rates, grid_params, model_params)

# Generate the animation
anim = ContactProcess.generate_animation(state_sequence, times)

# Save the animation
gif(anim, "contact_process.gif", fps = 1)

all_state_sequences, all_times = ContactProcess.multiple_simulations(grid_params, model_params)


# plot all the simulations one by one using gifs

@docstring """
    computing the likelihood of the data given the following model settings: 
    1.) infection_rate, recovery_rate = 0.0499, 0.1001
    2.) infection_rate, recovery_rate = 0.0501, 0.1001
    3.) infection_rate, recovery_rate = 0.0499, 0.0999
    4.) infection_rate, recovery_rate = 0.0501, 0.0999

    The function returns the likelihood of the data given the model settings
"""
model_params_1 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.1001, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day
model_params_2 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.1001, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day
model_params_3 = ContactProcess.ModelParameters(infection_rate = 0.0499, recovery_rate = 0.0999, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day
model_params_4 = ContactProcess.ModelParameters(infection_rate = 0.0501, recovery_rate = 0.0999, time_limit = 1, prob_infections = 0.01, num_simulations = 20_000) # rates are defined to be per day
function likelihood(model_params, all_state_sequences, all_times)
    likelihoods = []
    for i in 1:length(all_state_sequences)
        state_sequence = all_state_sequences[i]
        times = all_times[i]
        likelihood = ContactProcess.compute_likelihood(model_params, state_sequence, times)
        push!(likelihoods, likelihood)
    end
    return likelihoods
end
