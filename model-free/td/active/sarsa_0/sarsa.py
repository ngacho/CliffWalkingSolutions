import numpy as np
import gymnasium as gym
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import time

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

def normalize_list(non_normalized_list, shape=(4, 12)):
    non_normalized_list = non_normalized_list.copy()


    """
        Takes a 2d list and normalizes it's values.
    """
    max_value = np.amax(non_normalized_list)
    min_value = np.amin(non_normalized_list)
   
    for i in range(shape[0]):
        non_normalized_list[i] = (non_normalized_list[i] - min_value) / (max_value - min_value)
    return non_normalized_list

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
    # normalize state action values
    state_action_values = normalize_list(state_action_values)
    col, row = 12, 4  # e.g. 5 columns, 4 rows
    values = state_action_values.reshape((action_len, col, row))
    triangul = triangulation_for_triheatmap(col, row)
    fig, ax = plt.subplots(figsize=(20, 16))
    imgs = [ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
        for t, val in zip(triangul, values)]
    
    # Add text annotations
    # for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
    #     for i in range(col):
    #         for j in range(row):
    #             for k in range(action_len):
    #                 v = round(values[k, i, j], 2)
    #                 ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v}', color='b', ha='center', va='center')
    

    ax.set_xticks(range(col))
    ax.set_yticks(range(row))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    cbar = fig.colorbar(imgs[0], ax=ax)
    ax.axis('off')
    plt.savefig(f"plots/normalized-q-values-heatmap-{epoch}.png")
    plt.close()

def get_return(alpha, gamma, reward, state_action_val, new_state_new_action_val):
    """
        Q(st,at) ← Q(st,at) + α[rt+1 + γQ(st+1,at+1) − Q(st,at)]

    """
    return state_action_val + alpha * (reward + gamma * (new_state_new_action_val - state_action_val))

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
    
    ## discount factor
    gamma = 0.9
    ## step size
    alpha = 0.1
    # make the environment
    env = gym.make('CliffWalking-v0')
    state_len = env.nS
    action_len = env.nA
    # state-action values
    state_action_values = np.zeros((action_len, state_len))
    # initiate random policy
    # optimal_policy = [2 for _ in range(state_len//4)] + [2 for _ in range(state_len//4)] + [1 for _ in range((state_len//4) - 1)] + [2] + [0 for _ in range((state_len//4) - 1)] + [2]
    policy = np.random.randint(low=0, high=action_len, size=state_len, dtype="int64")
    ## initialize episodes
    tot_epoch = 50000
    last_value_change = 0
    last_policy_change = 0
    for epoch in range(tot_epoch + 1):
        old_state_action = state_action_values.copy()
        old_policy = policy.copy()
        # reset the environment.
        observation = simulate_explore_starts(env)
        # save episodes in a list
        while True:
            action = policy[observation]
            new_observation, reward, terminated, truncated, info = env.step(action)
            # update the state action values
            state_action_values[action, observation] = get_return(alpha, gamma, reward, state_action_values[action, observation], state_action_values[policy[new_observation], new_observation])
            # update the policy
            policy[observation] = np.argmax(state_action_values[:, observation])
            observation = new_observation
            if terminated or truncated: break

        # graph state action values.
        if epoch % (tot_epoch / 10) == 0:
            ## minimize state action values of terminal states
            
            print(f"****** Plot state action values for episode {epoch} ******")
            plot_state_action(state_action_values, action_len, epoch)
            print("****** Plot policy ******")
            plot_policies(policy, (4, 12), epoch)

        if((old_state_action != state_action_values).any()):
            last_value_change = epoch

        if((old_policy !=  policy).any()):
            last_policy_change = epoch


    print(f"********** Values last changed in epoch: {last_value_change} **********")
    print(f"********** Policy last changed in epoch: {last_policy_change} **********")
    play_game(policy)
    


    




    pass

if __name__ == "__main__":
    main()