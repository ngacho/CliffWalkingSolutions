import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pprint as pp
import time

def print_policy(p, shape=(4, 12)): 
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

def get_return(statelist, gamma):
    """
        @param statelist : list of (state, reward) pairs from current state until the terminal state
        @param gamma : discount factor
        @return : value of the state.
        Takes a list of (state, reward) pairs and computes the return
    """

    counter = 0
    return_value = 0
    for _, reward in statelist:
        return_value += reward * (gamma ** counter)
        counter += 1
    return return_value

def plot_values(state_values, epoch):
    fig = plt.figure()
    fig.patch.set_facecolor('#999999')
    plt.imshow(state_values, interpolation='none',  cmap='Greens')
    plt.axis('off')
    plt.savefig(f"plots/utility-estimate-after-epoch-{epoch}.png")
    plt.clf()

def plot_policies(policy_array, shape):
    # Example action dictionary mapping
    action_dict = {-1:"*", 0: "^", 1: ">", 2: "v", 3: "<"}
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
    for i in range(shape[0]):
        for j in range(shape[1]):
            arrow = reshaped_arrows[i, j]
            ax.text(j + 0.5, i + 0.5, arrow, fontsize=12, ha='center', va='center')
            rect = Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    ax.axis('off')
    plt.savefig(f"plots/Optimal-policy.png")
    plt.close()


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

    optimal_policy[37:48] = -1
    # graph optimal policy
    plot_policies(optimal_policy, (4, 12))
    return optimal_policy

        
def calculate_state_value_first_visit(episode_list, utility_matrix, running_mean_matrix, gamma, state_len):
    # if the episode is finished, try estimating utility
    counter = 0
    # checkup to identify if it's the first time to visit the state
    checkup_matrix = np.zeros(state_len)       
    
    # first visit monte carlo
    # for each state it stored, it checks if it's the first time to visit the state
    # if it's the first time, it computes the return and update the utility matrix
    for visit in episode_list:
        observation, reward = visit
        if checkup_matrix[observation] == 0:
            # compute return
            return_value = get_return(episode_list[counter:], gamma)
            # update utility matrix
            utility_matrix[observation] += return_value
            # update running mean matrix
            running_mean_matrix[observation] += 1
            # update checkup matrix
            checkup_matrix[observation] = 1
        counter += 1

    return utility_matrix / running_mean_matrix

def calculate_state_value_every_visit(episode_list, utility_matrix, running_mean_matrix, gamma, state_len):
    # if the episode is finished, try estimating utility
    counter = 0
    # checkup to identify if it's the first time to visit the state
    checkup_matrix = np.zeros(state_len)       
    
    # first visit monte carlo
    # for each state it stored, it checks if it's the first time to visit the state
    # if it's the first time, it computes the return and update the utility matrix
    for visit in episode_list:
        observation, reward = visit
        if checkup_matrix[observation] == 0:
            # compute return
            return_value = get_return(episode_list[counter:], gamma)
            # update utility matrix
            utility_matrix[observation] += return_value
            # update running mean matrix
            running_mean_matrix[observation] += 1
            # update checkup matrix
            checkup_matrix[observation] = 1
        counter += 1

    return utility_matrix / running_mean_matrix

def play_game(optimal_policy):
    time.sleep(5)
    env = gym.make('CliffWalking-v0', render_mode="human")
    observation, info = env.reset(seed=42)
    num_actions = 0
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

def main():
    """
        At each step, the agent records the reward obtained 
        and saves a history of all states visited until reaching a terminal state.

        we call an episode the sequence of states from the starting state to the terminal state.    
    """
    ## discount factor
    gamma = 0.9
    ## initialize episodes
    tot_epoch = 5000000
    # make the environment
    env = gym.make('CliffWalking-v0', render_mode="none")
    state_len = env.nS
    action_len = env.nA
    ## initialize utility matrix to estimate value of utility
    utility_matrix = np.zeros(state_len)
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full(state_len, 1.0e-10) 

    # define policy (this is the optimal policy)
    policy = [2 for _ in range(state_len//4)] + [2 for _ in range(state_len//4)] + [1 for _ in range((state_len//4) - 1)] + [2] + [0 for _ in range((state_len//4) - 1)] + [2]
    policy = np.array(policy)
    policy[37:48] = -1

    last_changed = 0

    print_policy(policy)

    ## for each episode:
    for epoch in range(tot_epoch + 1):
        old_average_returns = utility_matrix / running_mean_matrix
        # reset the environment.
        observation = simulate_explore_starts(env)
        # start new episode
        episode_list = list()
        while True:
            action = policy[observation]
            # episode_list.append((observation, reward))
            new_observation, reward, terminated, truncated, info = env.step(action)
            # increase the reward
            # if observation == 47: reward = 100
            episode_list.append((observation, reward))
            observation = new_observation
            if terminated or truncated: break

        average_returns = calculate_state_value_first_visit(episode_list, utility_matrix, running_mean_matrix, gamma, state_len)

        # graph the utility matrix after an epoch.
        if epoch % (tot_epoch / 10) == 0:
            print("****** Plot policies ******")
            state_values = average_returns.copy()
            state_values = np.round(state_values.reshape(4,12), 3)
            plot_values(state_values, epoch)
            print(f"******** Episode {epoch} completed ********")

        if epoch % (tot_epoch / 50000) == 0:
            print(f"<++++++++ episode {epoch} ++++++++>")

        if((old_average_returns != average_returns).any()):
            last_changed = epoch



    # close environment
    env.close()

    ## after all episodes are done, find and plot the optimal policy
    optimal_policy = find_optimal_policy(average_returns, env)
    print_policy(optimal_policy, shape=(4, 12))

    print(f"average returns last changed at epoch {last_changed}")

    #
    # play game
    print("****** Play game with optimal policy ******")
    play_game(optimal_policy)
    

if __name__ == "__main__":
    main()