# This code is following gupreet's tutorial on reinforcement learning
# The code can be found here: 
# https://gurpreet-ai.github.io/policy-evaluation-deep-reinforcement-learning-series/

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import pprint as pp

def print_policy(p, shape): 
    """Printing utility.
        Print the policy actions using symbols: 
        ^, v, <, > up, down, left, right
        * terminal states
        # obstacles
    """
    # Example action dictionary mapping
    counter = 0
    policy_string = ""
    for _ in range(shape[0]):
        for _ in range(shape[1]): 
            if (p[counter]== -1): policy_string+=" * "
            elif(p[counter]==0) : policy_string+=" ^ "
            elif(p[counter]==1) : policy_string+=" > "
            elif(p[counter]==2) : policy_string+=" v "
            elif(p[counter]==3) : policy_string+=" < "
            elif(np.isnan(p[counter])) : policy_string+="# " 
            counter += 1
        policy_string += '\n' 
    print(policy_string)



def policy_evaluation(env, action_probability, gamma, theta, state_values, policy, transition_model, goal_reward = 100):
    """
    policy_evaluation: 		estimate state values based on the policy
    
    @param env:       		OpenAI Gym environment
    @param policy:    		policy matrix containing actions and their probability in each state
    @param gamma:     		discount factor
    @param theta: 			evaluation will stop once values for all states are less than theta
    @param state_values: 	initial state values

    @return:         		new state values of the given policy
    """
    delta = theta*2 ## comparison to theta for stopping values.
    state_len = env.nS
    action_len = env.nA
    while (delta > theta): # while we are not at our stopping value
        delta = 0
        # for each state
        for state in range(state_len):
            state_transition_model = transition_model[state]
            # value of new state
            new_state_val = 0
            # for each action get the value of that state
            for action in range(action_len):
                possible_transitions = state_transition_model[action]
                # for each possible transition
                for transition in possible_transitions:
                    transition_prob, next_state, reward, done = transition
                    if next_state == 47:
                        reward = goal_reward

                    new_state_val += action_probability[state][action] * transition_prob * (reward + gamma * state_values[next_state])
            # update delta
            delta = max(delta, np.abs(new_state_val - state_values[state])) 
            # update state values
            state_values[state] = new_state_val
            # iterate policy
            policy[state] = policy_iteration(env, action_probability, state, state_values, transition_model)
    
    return state_values           

def policy_iteration(env, action_probability, state, state_values, transition_model):
    """Return the expected action.

    It returns an action based on the
    expected utility of doing a in state s,
    """
    state_transition_model = transition_model[state]
    state_len = env.nS
    action_len = env.nA
    actions_array = np.zeros(action_len)
    for action in range(action_len):
        transition_list = state_transition_model[action]
        value = 0
        for transition in transition_list:
            transition_prob, next_state, reward, done = transition
            value += action_probability[state][action] * transition_prob * state_values[next_state]
       
        #Expected utility of doing a in state s, according to T and u.
        actions_array[action] = value
    return np.argmax(actions_array)



def plot_values(state_values, goal_reward):
    fig = plt.figure()
    fig.patch.set_facecolor('#999999')
    plt.imshow(state_values, interpolation='none',  cmap='Greens')
    plt.axis('off')
    plt.savefig(f"plots/policy-evaluation-goal-reward-{goal_reward}.png")
    plt.clf()

def plot_policies(policy_array, shape, goal_reward):
    # Example action dictionary mapping
    action_dict = {0: "^", 1: ">", 2: "v", 3: "<"}
    # Convert numbers to arrows
    convert_to_arrow = np.vectorize(lambda x: action_dict[x])
    arrow_array = convert_to_arrow(policy_array)

    # Reshape the array to (4, 12) and reverse the order of rows
    reshaped_arrows = np.flipud(arrow_array.reshape(4, 12))

    # Create a figure and subplot
    fig, ax = plt.subplots(figsize=(8, 2))

    # Set a gray background
    ax.imshow([[0]], cmap='Greys', extent=(0, 12, 0, 4))

    # Iterate over the reshaped array and plot the arrows in each cell
    for i in range(4):
        for j in range(12):
            arrow = reshaped_arrows[i, j]
            ax.text(j + 0.5, i + 0.5, arrow, fontsize=12, ha='center', va='center')
            rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    ax.axis('off')
    plt.savefig(f"plots/policy-iteration-goal-reward-{goal_reward}.png")
    plt.clf()


    
def main():
    env = gym.make('CliffWalking-v0', render_mode="human")
    observation, info = env.reset(seed=42)
    transition_model = env.P


    goal_reward = 100
    for i in range(5): 
        # discount factor gamma
        gamma = 0.99
        state_len = env.nS
        action_len = env.nA
        # Probability of the agent taking an action from state.
        # for each state, agent can take any action with equal probability
        action_probability = np.ones((state_len, action_len))/action_len
        # initial state values
        state_values = np.zeros(state_len)
        theta = 0.0001

        # initial policy
        policy = np.random.randint(low=0, high=4, size=state_len, dtype="int64")
    
        print("****** Intial policy ******")
        print("****** Gamma (discount factor) ******")
        print(gamma)
        print("****** Theta ******")
        print(theta)
        state_values = policy_evaluation(env, action_probability, gamma, theta, state_values, policy, transition_model, goal_reward)
        print("****** Plot policies ******")
        plot_policies(policy, (4, 12), goal_reward)
        print("****** Best state values ******")
        state_values = np.round(state_values.reshape(4,12), 3)
        print(state_values)
        plot_values(state_values, goal_reward)
        print("NOTICE: state values are higher as you get closer to the goal.")
        goal_reward += 100
    


if __name__ == "__main__":
    main()