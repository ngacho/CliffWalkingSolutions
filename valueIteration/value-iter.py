import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
import operator


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


def run_value_iteration(gamma, transition_probability, reward_matrix, n_state, env):
    values = np.full((n_state), 0.0).astype(np.float32)

    iterations = 0 #Iteration counter
    epsilon = 0.01 #Stopping criteria small value

    while True:
        delta = 0
        u_values = values.copy()
        iterations += 1

        for state in range(n_state):
            # get the reward of being in that state
            reward = reward_matrix[state]
            # update the value of that state in our and save it separately
            values[state] = return_state_utility(reward, values, np.arange(env.action_space.n), 0.9, state, transition_probability)
            delta = max(delta, np.abs(u_values[state] - values[state])) #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma or iterations > 100000:
            break


    return values, iterations, delta



def create_reward_matrix(x, y, standard_reward, punishment, goal_reward):
    total_states = x * y
    reward_matrix = np.full((total_states), standard_reward).astype(np.float32)
    
    # for everything in the cliff, the reward is -10
    for i in range((total_states-1)-10, total_states - 1):
        reward_matrix[i] = -punishment
    # the reward of the goal is 10
    reward_matrix[59] = goal_reward

    return reward_matrix

def create_multiplot(value_iteration_results):
    # sort by the discount factor
    sorted_list = sorted(value_iteration_results, key=lambda x: x[2])

    # split the list into a group based on learning rate.
    for key,group in itertools.groupby(sorted_list, operator.itemgetter(2)):
        # for each group, create a plot with transition probability on the x axis and goal/punishment ratio on the y axis.

        # Create a grid of subplots
        # Define the number of goal/punishment ratios and transition probabilities
        num_ratios = 10
        num_probs = 4

        # Create a grid of subplots
        num_plots = num_ratios * num_probs
        num_cols = num_ratios  # Number of columns in the subplot grid
        num_rows = num_probs  # Number of rows in the subplot grid

        fig, axes = plt.subplots(num_cols, num_rows, figsize=(60, 48))  # Adjust the figure size as needed
        # Set the background color of the figure
        fig.patch.set_facecolor('#999999')  # Set to the desired color

        # FLatten the axes
        axes = axes.flatten()

        # Plot the color maps
        for i, result in enumerate(group):
            ((values, iterations, delta), transition_prob, gamma, goal_reward, punishment) = result

            # Calculate the subplot indices
            row_idx = i // num_ratios
            col_idx = i % num_ratios
            # col_idx = num_ratios - 1 - (i % num_ratios)

            # Plot the color map
            ax = axes[i]
            heatmap = ax.imshow(values, cmap='Greens')
            ax.set_title(f'Transition Prob: {transition_prob}, Goal/Punish Ratio: {goal_reward/punishment:.2f}')
            ax.axis('off')
            fig.colorbar(heatmap, ax=ax)

            # Set the x and y axis labels only on the bottom row and left column
            if row_idx == num_rows - 1:
                ax.set_xlabel('Goal/Punish Ratio')
            if col_idx == 0:
                ax.set_ylabel('Transition Probability')

        # Adjust the spacing between subplots
        fig.tight_layout()

        plt.savefig(f"learning-rate-{gamma}.png")
        print(f"Saved plot for {gamma}")
        # Clear with clf() function:
        plt.clf()

def main():
    # create the values
    value_iteration_results = list()
    env = gym.make('CliffWalking-v0', render_mode="human")
    n_state = 60
    # observation, info = env.reset()

    # PLAY Aroung with reward and punishment.
    gamma = 0.4 #Discount factor
    transition_probability = 0.7 # Transition probability


    standard_reward = -1.0
    punishment = 100
    goal_reward = 10

    # effects of the punishment and goal reward ratio.
    for _ in range(0, 10):
        # print(f"Running Value Iteration reward: {goal_reward}. Punishment: {punishment}")
        reward_matrix = create_reward_matrix(12, 5, standard_reward, punishment, goal_reward)

        # play with discount factor and transition probability
        gamma = 0.4 #Discount factor

        for _ in range(0, 6):
            transition_probability = 0.8 # Transition probability
            for _ in range(0, 4):
                # print(f"\tgamma : {gamma}, transition_probability: {transition_probability}")
                values, iterations, delta = run_value_iteration(gamma, transition_probability, reward_matrix, n_state, env)
                # plot maps
                values = values.reshape((5, 12))
                value_iteration_results.append(((values, iterations, delta), transition_probability, gamma, goal_reward, punishment))
                transition_probability = round(transition_probability - 0.2, 2)
            gamma = round(gamma + 0.1, 2)

        punishment -= 10
        goal_reward += 10

    create_multiplot(value_iteration_results)



if __name__ == "__main__":
    main()