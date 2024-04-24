using Plots

# Define the size of the game grid
width, height = 20, 20

# Initialize the game state with random values
state = rand(Bool, width, height)

# Function to count the number of live neighbors around a given cell
function count_neighbors(state, x, y)
    sum([state[mod1(x+i,width), mod1(y+j,height)] for i in -1:1 for j in -1:1]) - state[x, y]
end

# Function to update the game state
function update_state(state)
    new_state = copy(state)
    for x in 1:width
        for y in 1:height
            live_neighbors = count_neighbors(state, x, y)
            new_state[x, y] = live_neighbors == 3 || (state[x, y] && live_neighbors == 2)
        end
    end
    return new_state
end

# Run the simulation and plot each step
anim = @animate for step in 1:100
    state = update_state(state)
    heatmap(state, color=:grays, legend=false)
end

# Save the animation as a gif
gif(anim, "game_of_life.gif", fps = 10)