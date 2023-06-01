import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pprint as pp
import sys
import time
    

def print_policy(p, shape): 
    """Printing utility.
        Print the policy actions using symbols: 
        ^, v, <, > up, down, left, right
        * terminal states
        # obstacles
        0 up, 1:  right, 2: down, 3: left
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

def plot_values(state_values, shape, epoch):
    state_values = np.round(state_values.reshape(shape), 3)
    fig = plt.figure()
    fig.patch.set_facecolor('#999999')
    plt.imshow(state_values, interpolation='none', cmap='Greens')
    
    # Add text values to the plot
    for i in range(shape[0]):
        for j in range(shape[1]):
            plt.text(j, i, "{:.2f}".format(state_values[i, j]), color='black', ha='center', va='center')
    
    plt.axis('off')
    plt.savefig(f"plots/utility-estimate-after-epoch-{epoch}.png")
    plt.clf()

def play_game(optimal_policy):
    env = gym.make('CliffWalking-v0', render_mode="human")
    observation, info = env.reset(seed=42)
    num_actions = 0
    time.sleep(5)
    while True:
        action = int(optimal_policy[observation]) # take action based on policy.
        new_observation, _, terminated, truncated, _ = env.step(action)
        num_actions += 1
        while observation == new_observation:
            # take random action
            action = env.action_space.sample()
            new_observation, _, terminated, truncated, _ = env.step(action)
            num_actions += 1

        if num_actions > 50:
            print(f"Terminated without optimal policy\n")
            observation, _ = env.reset()
            break

        if terminated or truncated:
            print(f"Terminated normally")
            observation, _ = env.reset()
            break
            
        observation = new_observation
    env.close()

def normalize_list(non_normalized_list):
    normalized_list = non_normalized_list.copy()


    """
        Takes a 2d list and normalizes it's values.
    """
    max_value = np.amax(normalized_list)
    min_value = np.amin(normalized_list)
    range_value = max_value - min_value

    for i in range(len(normalized_list)):
        normalized_list[i] = normalized_list[i] if range_value == 0 else (normalized_list[i] - min_value) / range_value

    return normalized_list

def find_optimal_policy(utility_matrix, env):
    """

    Note we are using the transition model to know where we end up.
    We don't do any calculations using the values we get from the transition model.
    
    """
    state_len = env.nS
    action_len = env.nA
    transition_model = env.P
    optimal_policy = np.zeros(state_len)
    ## for each state, find the action that maximizes the utility
    for state in range(state_len):
        state_transition_model = transition_model[state]
        actions_array = np.zeros(action_len)
        for action in range(action_len):
            transition_list = state_transition_model[action]
            value = 0
            for transition in transition_list:
                transition_prob, next_state, reward, done = transition
                value += (utility_matrix[next_state] * transition_prob)                
            actions_array[action] = value
        optimal_policy[state] = np.argmax(actions_array)

    # graph optimal policy
    print_policy(optimal_policy, (4, 12))
    return optimal_policy

def simulate_explore_starts(env):
    # reset the environment.
    observation, info = env.reset()
    # simulate exploring starts
    for _ in range(50):
        action = np.random.randint(0, high=4, dtype='int32')
        new_observation, _, terminated, truncated, _ = env.step(action)
        observation = new_observation
        if terminated or truncated: 
            observation, info = env.reset()

    return observation

def update_utility(utility_matrix, trace_matrix, alpha, delta):
    '''Return the updated utility matrix

    @param utility_matrix the matrix before the update
    @param alpha the step size (learning rate)
    @param delta the error (Taget-OldEstimte) 
    @return the updated utility matrix
    '''
    utility_matrix += alpha * delta * trace_matrix
    return utility_matrix

def update_eligibility(trace_matrix, gamma, lambda_):
    '''Return the updated trace_matrix

    @param trace_matrix the eligibility traces matrix
    @param gamma discount factor
    @param lambda_ the decaying value
    @return the updated trace_matrix
    '''
    trace_matrix = trace_matrix * gamma * lambda_
    return trace_matrix


def main():
    ## discount factor
    gamma = 0.9
    ## step size
    alpha = 0.1
    ## decaying value
    lambda_ = 0.2
    ## initialize episodes
    tot_epoch = 1000000
    # make the environment
    env = gym.make('CliffWalking-v0', render_mode="None")
    state_len = env.nS
    action_len = env.nA
    ## initialize utility matrix to estimate value of utility
    utility_matrix = np.zeros(state_len)
    utility_matrix = np.float64(utility_matrix)
    # reward matrix
    reward_matrix = np.full(state_len, -1)
    reward_matrix[37:47] = -10
    reward_matrix[47] = 100
    # trace matrix
    trace_matrix = np.full(state_len, 1)
    
    optimal_policy = [2 for _ in range(state_len//4)] + [2 for _ in range(state_len//4)] + [1 for _ in range((state_len//4) - 1)] + [2] + [0 for _ in range((state_len//4) - 1)] + [2]
    last_changed = 0
    for epoch in range(tot_epoch + 1):
        old_utility_matrix = utility_matrix.copy()
        observation = simulate_explore_starts(env)
        # print(f"\tStarting episode {epoch} from {observation}")   
        while True:
             # take action from policy
            action = optimal_policy[observation]
            new_observation, _, terminated, truncated, _ = env.step(action)
            reward = reward_matrix[new_observation]
            delta = reward + (gamma * (utility_matrix[new_observation] - utility_matrix[observation]))
            #Adding +1 in the trace matrix (only the state visited)
            trace_matrix[observation] += 1
            # update the utility matrix
            utility_matrix = update_utility(utility_matrix, trace_matrix, alpha, delta)
            #Update the trace matrix (decaying) (all the states)
            trace_matrix = update_eligibility(trace_matrix, gamma, lambda_)
            # update the observation
            observation = new_observation
            if terminated or truncated: break

        # draw q-values
        if epoch % (tot_epoch / 10) == 0:
            utility_matrix[37:47] = -10
            utility_matrix[47] = 100
            values = normalize_list(utility_matrix)
            # print(utility_matrix)
            plot_values(values, (4, 12), epoch)
            print(f"******** Episode {epoch} completed ********")

        # rounded_normalized_old_values = np.round(old_utility_matrix, 4)
        # rounded_normalized_changed_values = np.round(utility_matrix, 4)

        if((old_utility_matrix != utility_matrix).any()):
            last_changed = epoch


    optimal_policy = find_optimal_policy(utility_matrix, env)
    print(f"utility matrix last changed at epoch {last_changed}")
    play_game(optimal_policy)

    # utility matrix stopped changing at epoch 393694

    env.close()

if __name__ ==  "__main__":
    main()