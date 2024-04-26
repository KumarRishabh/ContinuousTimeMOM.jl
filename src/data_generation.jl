# Data Generating model as per contact process on a torus

using Random
using Plots
using Distributions

# Initialize parameters
width, height = 50, 50 # dimensions of the grid
# add paddings to the grid 
width_with_padding = width + 2
height_with_padding = height + 2
# prob_infect = 0.2       # probability of infecting a neighbor
infection_rate = 2 # a
# prob_recover = 0.1 
recovery_rate = 1 # s   # probability of recovery
prob_noise = 0.05       # probability of observation noise
num_steps = 300       # number of time steps to simulate

# Initialize the grid
state = zeros(Bool, width_with_padding, height_with_padding)
initial_infections = 10  # start with 10 infected individuals
for _ in 1:initial_infections
    state[rand(1:width), rand(1:height)] = true
end

# Function to update the state of the grid



function calculate_all_rates(state)
    # After calculating the rates, we need to select the position of the node 
    # which will be updated
    global width_with_padding, height_with_padding
    rates = zeros(width_with_padding, height_with_padding)
    for i in 2:width_with_padding-1
        for j in 2:height_with_padding-1
            rates[i, j] = infection_rate *(1 - state[i, j]) * (state[i, j+ 1] + state[i, j - 1] + state[i - 1, j] + state[i + 1, j]) + recovery_rate * state[i, j]
        end
    end
    # choose a grid position according to the rate 

end

function sample_time_with_rates(state, rates)
    global width_with_padding, height_with_padding
    rate = sum(rates)
    # sample from the Exponential rate
    time = rand(Exponential(1 / rate))
    return time
end

function update_states!(state, rates)
    global width_with_padding, height_with_padding
    r = rand()
    cum_rate = 0.0
    total_rate = sum(rates)
    for i in 2:width_with_padding-1
        for j in 2:height_with_padding-1
            cum_rate += rates[i, j]
            if cum_rate >= r * total_rate
                state[i, j] = !state[i, j]
                break
            end
        end
        if cum_rate >= r * total_rate
            break
        end
    end
    return (i, j)
end

function update_rates!(state, rates)
# rate = calculate_rates(state)
    # calculate the rate for the min(Exponential) [addition of rates]
    # sample from the Exponential rate

# Function to add noise to observations
function add_noise(state)
    noisy_state = copy(state)
    for i in 1:width
        for j in 1:height
            if rand() < prob_noise
                noisy_state[i, j] = !noisy_state[i, j]
            end
        end
    end
    return noisy_state
end
# print(state)
# Initialize the plot

# function plot_noisy_state(noisy_state)
    
# heatmap(state, color=:green)
# Run the simulation


function generate_animation!(state, num_steps)
    p = heatmap(state, color=:grays, legend = false)

    anim = @animate for step in 1:num_steps
        state = update_state(state)
        # noisy_state = add_noise(state)
        
        # clf(p)
        heatmap!(p, state, color=:grays, lengend = false)
        # Put time step in the title
        title!("Time step: $step")
    end
    return anim
end

@time anim = generate_animation!(state, num_steps)
gif(anim, "contact_process_tori.gif", fps=2)
println("Simulation completed.")
# gif(anim, "contact_process_tori.gif", fps=10)
# println("Simulation completed.")
