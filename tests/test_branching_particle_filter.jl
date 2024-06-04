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
Random.seed!(1)
dir_path = joinpath(@__DIR__, "../src/filtered_data")
save_fig_path = joinpath(@__DIR__, "../src/figures")
grid_params = ContactProcess.GridParameters(width=10, height=10) # changed to 15 × 15 grid
model_params = ContactProcess.ModelParameters(infection_rate=0.05, recovery_rate=0.1, time_limit=100, prob_infections=0.3, num_simulations=1000) # rates are defined to be per day

state, rates = ContactProcess.initialize_state_and_rates(grid_params, model_params; mode="fixed_probability")
state_sequence, transition_times, updated_nodes = ContactProcess.run_simulation!(state, rates, grid_params, model_params)
observed_state, time = observe_state(state)
observed_state
observations, observation_time_stamps = get_observations_from_state_sequence(state_sequence, model_params.time_limit, transition_times)
observed_dict = Dict(observation_time_stamps .=> observations)
initial_num_particles = 14000 # initially start with 100 particles
# between the observation time stamps, simulate the contact process with s tate and rates
V = 0.2 * rand() - 0.1
U = 0.2 * rand() - 0.1

# initialize_particles(initial_num_particles, grid_params, model_params)

CTMOM.branching_particle_filter(initial_num_particles, grid_params, model_params, observation_time_stamps, observations, r=5, signal_state=state, signal_rate=rates, testing_mode = false)

# total_particles(particle_history, 0.0)
function estimate_infection_grid(time_stamp)
    estimate = zeros(Float64, 12, 12)
    normalization_factor = 0
    time_stamp = Float32(time_stamp)
    particles = load("$dir_path/particle_history_$(time_stamp).jld2", "particles")
    total_particles = count_total_particles(particles)
    for j ∈ 1:total_particles
        particle = particles[j]
        # print("Particle: $j", particle)
        estimate .+= particle.weight * particle.state
        normalization_factor += particle.weight
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

function calculate_error_trajectory(observation_time_stamps, state_sequence, transition_times; time_limit = Inf)
    errors = []
    for i ∈ eachindex(observation_time_stamps)
        if observation_time_stamps[i] > time_limit
            break
        end
        println("Reading file $dir_path/particle_history_$(observation_time_stamps[i]).jld2")
        estimate = estimate_infection_grid(observation_time_stamps[i]) # TODO: This is a hack, refactor
        actual_signal = actual_infection_grid(state_sequence, observation_time_stamps[i + 1], transition_times)
        error = calculate_error(estimate, actual_signal)
        push!(errors, error)
    end
    return errors
end
observation_time_stamps[1]
errors = calculate_error_trajectory(observation_time_stamps, state_sequence, transition_times, time_limit=model_params.time_limit)
plot(observation_time_stamps[1:length(errors)], errors, label="Error", xlabel="Time", ylabel="Error", title="Error in estimating the infection grid")
savefig("$save_fig_path/error_plot_3.png")

function get_heatmap_data(time_stamp)
    heatmap_data = []
    i = Float32(time_stamp)
    estimate = zeros(Float64, 12, 12)
    normalization_factor = 0
    particles = load("$dir_path/particle_history_$(i).jld2", "particles")
    total_particles = count_total_particles(particles)
    println("Particles: $(particles[1])")
    for j ∈ 1:total_particles
        particle = particles[j]
        estimate += particle.weight * particle.state
        normalization_factor += particle.weight
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

# sample_time_stamps = sample(observation_time_stamps, 5)

# heatmap_data = get_heatmap_data(sample_time_stamps[1])
# P = plot_heatmap(heatmap_data, sample_time_stamps[1])
# save_particle_history(particle_history, "$dir_path/particle_history.csv")

