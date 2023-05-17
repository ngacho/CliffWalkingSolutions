import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import matplotlib.pyplot as plt


def return_state_utility(reward, utility_vector, actions, gamma, observation, transition_probability=1.0):
    """
    Calculates the utility of a state based on future rewards

        @param utility_vector: list, utilities of each state
        @param reward: int, reward for the current state
        @param gamma: float, discount factor
        @param env: gym.env, environment
    """
    
    action_array = np.zeros(len(actions))

    for action in actions:
        action_array[action] = calculate_action_utility(action, observation, utility_vector, transition_probability)
       
    # calculate the utility of each action
    return reward + gamma * np.max(action_array)
    
def calculate_action_utility(action, observation, utility_vector, transition_probability):
    utility = 0
    if transition_probability == 1:
        next_state = step_action(action, observation)
        if 0 <= next_state < len(utility_vector): 
            # source the utility of each state 
            utility = utility_vector[next_state]
    else:
        probable_actions = [[1, 3], [0, 2]]
        for actions in probable_actions[action % 2]:
            next_state = step_action(actions, observation)
            if 0 <= next_state < len(utility_vector): 
                # source the utility of each state 
                utility += (utility_vector[next_state] * ( (1 -transition_probability) / 2))
        
        next_state = step_action(action, observation)
        if 0 <= next_state < len(utility_vector): 
            # source the utility of each state
            utility += (utility_vector[next_state] *  transition_probability)

    return utility



def step_action(action, observation):

    """
        # 0: Move up 
        # 1: Move right
        # 2: Move down
        # 3: Move left
    
    """
    if action == 0:
        return observation - 12
    elif action == 1 and (observation + 1) % 12 != 0:
        return observation + 1
    elif action == 2:
        return observation + 12
    elif action == 3 and observation % 12 != 0:
        return observation - 1  
    else:
        return -1
    

    




env = gym.make('CliffWalking-v0', render_mode="human")
n_state = env.observation_space.n
# observation, info = env.reset()


gamma = 0.4 #Discount factor
transition_probability = 1 # Transition probability

for i in range(0, 5):
    values = np.full((n_state), 0.0).astype(np.float32)
    # initialize reward matrix as -1 
    reward_matrix = np.full((n_state), -1.0).astype(np.float32)
    # for everything in the cliff, the reward is -10
    for i in range(37, 47):
        reward_matrix[i] = -100.0
    # the reward of the goal is 10
    reward_matrix[47] = 10.0

    # permitted actions
    actions = np.arange(env.action_space.n)

    gamma += 0.1
    iteration = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value


    while True:
        delta = 0
        u_values = values.copy()
        iteration += 1

        print(f"iteration ... {iteration}")
        for state in range(n_state):
            # get the reward of being in that state
            reward = reward_matrix[state]
            # update the value of that state in our and save it separately
            values[state] = return_state_utility(reward, values, np.arange(env.action_space.n), 0.9, state, transition_probability)
            delta = max(delta, np.abs(u_values[state] - values[state])) #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
            break

    print("Iterations: " + str(iteration))
    print("Delta: " + str(delta))
    values = values.reshape((4, 12))

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the data with colors
    heatmap = ax.imshow(values, cmap='Greens')

    # Add the colorbar
    cbar = plt.colorbar(heatmap)

    # Show the values in the cells
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center', color='w')

    # Set the axis labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Value Iteration Plot')
    # Adjust spacing to avoid overlapping
    
    info_text = f'Transition Probability: {transition_probability}\nGamma: {round(gamma, 2)}'
    # fig.subplots_adjust(bottom=0.05)
    ax.text(0.5, 0.1, info_text, transform=ax.transAxes, ha='center')

    plt.savefig(f"plots/gamma-{round(gamma, 2)}-transition-probability-{transition_probability}.png")
    # Clear with clf() function:
    plt.clf()



# print(f"inital observation: {observation}")
# print(f"info: {info}")

# actions = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
# for action in actions:
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(f"observation {observation}")

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close()

