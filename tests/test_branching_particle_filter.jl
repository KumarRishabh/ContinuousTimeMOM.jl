# Create tests for the branching particle filter
using Random
using Distributions
using Plots 
using Revise
using ProgressMeter 
using FileIO 
include("../src/ContactProcess.jl")
include("../src/CTMOM.jl")
using .CTMOM
using .ContactProcess
using JLD2
Random.seed!(10)

grid_params = ContactProcess.GridParameters(width=10, height=10)
model_params = ContactProcess.ModelParameters(infection_rate=0.025, recovery_rate=0.05, time_limit=2, prob_infections=0.5, num_simulations=1000) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params; mode="fixed_probability")
state_sequence, transition_times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, model_params)
observed_state, time = observe_state(state)
observed_state
observations, observation_time_stamps = get_observations_from_state_sequence(state_sequence, model_params.time_limit, transition_times)
observed_dict = Dict(observation_time_stamps .=> observations)
initial_num_particles = 50 # initially start with 100 particles
# between the observation time stamps, simulate the contact process with s tate and rates
V = 0.2 * rand() - 0.1
U = 0.2 * rand() - 0.1

initialize_particles(initial_num_particles, grid_params, model_params)

particle_history = CTMOM.branching_particle_filter(initial_num_particles, grid_params, model_params, observation_time_stamps, observations, r=5, signal_state=state, signal_rate=rates)

# total_particles(particle_history, 0.0)

function estimate_infection_grid(particle_history, time_stamp)
    estimate = zeros(Float64, 12, 12)
    normalization_factor = 0
    for j ∈ 1:initial_num_particles
        for particle in particle_history[Float32(time_stamp)][j]
            estimate .+= particle.weight * particle.state
            normalization_factor += particle.weight
        end
    end
    estimate /= normalization_factor

    return estimate
end

function actual_infection_grid(state_sequence, time_stamp, transition_times)
    return state_sequence[findlast(x -> x <= time_stamp, transition_times)]
end

function calculate_error(estimate, actual_signal)
    error = 0
    for i in 1:size(estimate, 1)
        for j in 1:size(estimate, 2)
            error += (estimate[i, j] - actual_signal[i, j])^2
        end
    end

    return error / (size(estimate, 1) * size(estimate, 2))
end

function calculate_error_trajectory(particle_history, observation_time_stamps, state_sequence, transition_times)
    errors = []
    for i ∈ eachindex(observation_time_stamps)
        estimate = estimate_infection_grid(particle_history, observation_time_stamps[i])
        actual_signal = actual_infection_grid(state_sequence, observation_time_stamps[i], transition_times)
        error = calculate_error(estimate, actual_signal)
        push!(errors, error)
    end
    return errors
end
length(observation_time_stamps)
errors = calculate_error_trajectory(particle_history, observation_time_stamps, state_sequence, transition_times)
plot(observation_time_stamps, errors, label="Error", xlabel="Time", ylabel="Error", title="Error in estimating the infection grid")
savefig("error_plot_2.png")
function get_heatmap_data(particle_history, time_stamp)
    heatmap_data = []
    i = Float32(time_stamp)
    estimate = zeros(Float64, 17, 17)
    normalization_factor = 0
    for j ∈ 1:initial_num_particles
        # estimate += sum([particle.weight * particle.state for particle in particle_history[i][j]])
        # do a scalar multiplication of the state with the weight of the particle and then do element-wise addition
        for particle in particle_history[i][j]
            # check if particle.state is a 17 x 17 matrix

            estimate .+= particle.weight * particle.state
        end

        normalization_factor += sum([particle.weight for particle in particle_history[i][j]])
    end
    estimate /= normalization_factor
    push!(heatmap_data, estimate)

    return heatmap_data
end

# sample random time_stamps from the observation_time_stamps
function plot_heatmap(heatmap_data, time_stamp)
    P = heatmap(heatmap_data, color=:grays)
    return P
end

sample_time_stamps = sample(observation_time_stamps, 5)

particle_history = load("$dir_path/particle_history.jld")

# save particle history to a csv file
function save_particle_history(particle_history, file_name)
    open(file_name, "w") do io
        for time in keys(particle_history)
            total_particles = count_total_particles(particle_history[time])
            for i ∈ 1:total_particles
                particle = particle_history[time][i]

                write(io, "$time, $i, $(particle.state), $(particle.weight)\n")

            end
        end
    end
end

save_particle_history(particle_history, "$dir_path/particle_history.csv")
function save_simulation_to_csv(state_sequence, transition_times, file_name)
    open(file_name, "w") do io
        for i ∈ 1:length(transition_times)
            write(io, "$transition_times[i], $(state_sequence[i])\n")
        end
    end
end

save_particle_history(particle_history, "$dir_path/particle_history.csv")
