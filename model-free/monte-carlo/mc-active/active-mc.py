import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import random

def get_return(state_list, gamma):
    """
        @param statelist : list of (state, reward) pairs from current state until the terminal state
        @param gamma : discount factor
        @return : value of the state.
        Takes a list of (state, reward) pairs and computes the return
    """

    counter = 0
    return_value = 0
    for visit in state_list:
        #  (observation, action, reward) = visit. we just care about observation.
        _, _, reward = visit
        return_value += reward * np.power(gamma, counter)

    return return_value


def update_policy(episode_list,policy, average_state_action_return):
    for visit in episode_list:
        observation, _, _ = visit
        if policy[observation] != -1:
            # update policy to be the one with the highest value for each state
            policy[observation] = np.argmax(average_state_action_return[:, observation])

    return policy
    

def plot_policies(policy_array, shape, epoch):
    # Example action dictionary mapping
    action_dict = {0: "^", 1: ">", 2: "v", 3: "<"}
    # Convert numbers to arrows
    convert_to_arrow = np.vectorize(lambda x: action_dict[x])
    arrow_array = convert_to_arrow(policy_array)
    arrow_array[37:48] = ""

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
    plt.savefig(f"plots/Optimal-policy-for-epoch-{epoch}.png")
    plt.clf()


def triangulation_for_triheatmap(col, row):
    xv, yv = np.meshgrid(np.arange(-0.5, col), np.arange(-0.5, row))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, col), np.arange(0, row))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (col + 1) * (row + 1)  # indices of the centers

    trianglesN = [(i + j * (col + 1), i + 1 + j * (col + 1), cstart + i + j * col)
                  for j in range(row) for i in range(col)]
    trianglesE = [(i + 1 + j * (col + 1), i + 1 + (j + 1) * (col + 1), cstart + i + j * col)
                  for j in range(row) for i in range(col)]
    trianglesS = [(i + 1 + (j + 1) * (col + 1), i + (j + 1) * (col + 1), cstart + i + j * col)
                  for j in range(row) for i in range(col)]
    trianglesW = [(i + (j + 1) * (col + 1), i + j * (col + 1), cstart + i + j * col)
                  for j in range(row) for i in range(col)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

def plot_state_action(state_action_values, action_len, epoch):
    col, row = 12, 4  # e.g. 5 columns, 4 rows
    values = state_action_values.reshape((action_len, col, row))
    triangul = triangulation_for_triheatmap(col, row)
    norms = [plt.Normalize(-0.5, 1) for _ in range(4)]
    fig, ax = plt.subplots(figsize=(20, 16))
    imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', ec='white')
        for t, val in zip(triangul, values)]

    ax.set_xticks(range(col))
    ax.set_yticks(range(row))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    cbar = fig.colorbar(imgs[0], ax=ax)
    ax.axis('off')
    plt.savefig(f"plots/q-values-heatmap-{epoch}.png")
    plt.clf()

def get_policy(state_action_values, state_len):
    policy = np.zeros(state_len, dtype="int64")
    for i in range(state_len):
        policy[i] = np.argmax(state_action_values[:, i])
    
    return policy

def play_game(optimal_policy):
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


def calculate_state_value_first_visit(episode_list, state_action_values, running_mean_matrix, gamma, action_len, state_len):
    # The episode is finished, now estimating the utilities
    counter = 0
    # Checkup to identify if it is the first visit to a state-action
    checkup_matrix = np.zeros((action_len,state_len))
    # For each visit in the episode
    for visit in episode_list:
        observation, action, _ = visit
        # If it is the first visit to the state-action, estimate the return
        if(checkup_matrix[action, observation] == 0):
            # get the return value of that episode
            return_value = get_return(episode_list[counter:], gamma)
            # Update the running mean matrix
            running_mean_matrix[action, observation] += 1
            # Update the state-action matrix
            state_action_values[action, observation] += return_value
            # update the check up matrix to visited
            checkup_matrix[action, observation] = 1

        counter += 1

    return state_action_values / running_mean_matrix



def main():
    """
        At each step, the agent records the reward obtained 
        and saves a history of all states visited until reaching a terminal state.

        we call an episode the sequence of states from the starting state to the terminal state.    
    """

     ## discount factor
    gamma = 0.9
    ## initialize episodes
    tot_epoch = 1000000
    # e-greedy policy
    epsilon = 0.7
    # make the environment
    env = gym.make('CliffWalking-v0', render_mode="None")
    state_len = env.nS
    action_len = env.nA
    # state-action values
    state_action_values = np.zeros((action_len, state_len))
    # random policy
    policy = np.random.randint(low=0, high=action_len, size=state_len, dtype="int64")
    # init with low numbers to avoid division by zero.
    running_mean_matrix = np.full((action_len, state_len), 1.0e-10)
     ## for each episode:
    for epoch in range(tot_epoch + 1):
        # reset the environment.
        observation, info = env.reset()
        # save episodes in a list
        episode_list = list()
        # a thousand random actions e-greedy
        for i in range(1000):
            action = policy[observation]
            # randomly take an action sometimes.
            if random.random() > epsilon: 
                action = env.action_space.sample()

            new_observation, reward, terminated, truncated, info = env.step(action)
            #Append the visit in the episode list
            episode_list.append((observation, action, reward))
            observation = new_observation
            if terminated or truncated: break


        # episode ended, calculate the state-action values
        average_state_action_return = calculate_state_value_first_visit(episode_list, state_action_values, running_mean_matrix, gamma, action_len, state_len)
        # update the policy matrices
        policy = update_policy(episode_list, policy, average_state_action_return)
        # graph state action values.
        if epoch % (tot_epoch / 10) == 0:
            ## minimize state action values of terminal states
            
            print("****** Plot state action values ******")
            plot_state_action(average_state_action_return, action_len, epoch)
            print("****** Plot policy ******")
            plot_policies(policy, (4, 12), epoch)

            
        print(f"Episode {epoch} done...")
    
    # get the correct policy
    plot_policies(policy, (4, 12), 'final')


    # close environment
    env.close()


    ## play the game
    print("************ Playing game with optimal policies ************")
    play_game(policy)

if __name__ == "__main__":
    main()