import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt
import numpy as np
import copy
import pprint as pp





def return_policy_evaluation(env, state, state_values, discount_factor, transition_model):

    """
        Latex expression: $v_{k+1} = \sum_{a} \pi(a|s) \sum_{s', r} p(s', a |s, a) \\big[ r + \gamma v_{k}(s') \\big]$
        @param env : gymnasium environment
        @param state : current state we are in
        @param state_values : values of each state defined in an array
        @discount_factor : discount factor for learning
        @transition_model : transition model of moving from one state to another

        This method calculates the utility of a state based on the bellman equation expressed above.
    
    """

    state_transition_model = transition_model[state]
    action_space = env.action_space.n
    action_array = np.full(action_space, 0, dtype="float32")
    # sum of getting to other states from our state.
    for action in range(action_space):
        action_value = 0
        transition_list = state_transition_model[action]
        # current transition list has 1 possibility as transition probability is 1
        for i in transition_list:
            # transition probability to the next state from our state 
            probability, next_state, reward, _ = i
            
            # get value of that state
            Rt_1 = probability * (reward + (discount_factor * state_values[next_state]))
        
            action_value += Rt_1 
        
        action_array[action] = action_value
    
    return sum(action_array) 

    
def update_policy(env, state, state_values, policy_array, transition_model):
    # action_dict =  {None : "Nothing", -1 : "Terminal", 0 : "Up", 1 : "Left" , 2 : "Down" , 3 : "Right"}
    state_transition_model = transition_model[state]

    action_space = env.action_space.n
    # empty array
    action_array = np.full(action_space, 0, dtype="float32")
    # sum of getting to other states from our state.
    for action in range(action_space):
        action_value = 0
        transition_list = state_transition_model[action] 
        for i in transition_list:
            _, next_state, _, _ = i
            action_value += state_values[next_state]

        action_array[action] = action_value

    # print(f"sum of actions possibilities: {sum(action_array)}")

    policy_array[state] = np.argmax(action_array)

    # get the action with the highest value.

def main():
    
    env = gym.make('CliffWalking-v0', render_mode="human")
    n_state = env.observation_space.n
    action_space = env.action_space

    # transition model
    transition_model = env.P
    
    # define random policy.
    policy_array = np.random.randint(low=0, high=4, 
                                    size=n_state, dtype="int64")
    
    
    # define random policy values.
    state_values = np.full(n_state, 0, dtype="float32")


    # policy iteration
    theta = 0.01# stopping criteria
    delta = theta * 2
    discount_factor = 0.8
    iterations = 0
    stop = False

    while(delta < theta * (1 - discount_factor) / discount_factor):
        delta = 0
        old_state_values = state_values.copy()
        # calculate value for each state
        for state in range(n_state):
            # calculate the new state value
            new_state_value = return_policy_evaluation(env, state, state_values, discount_factor, transition_model)
            # get the difference
           
            delta = max(delta, np.abs(old_state_values[state] - new_state_value)) #Stopping criteria
            print(f"Delta {delta} v Theta {theta}")
            
            # update the state value matrix
            state_values[state] = new_state_value
            # update the policy.
            update_policy(env, state, state_values, policy_array, transition_model)

            iterations += 1
            
        # print(f"iteration {iterations}, delta {delta} vs stopping criteria : {theta * ((1 - discount_factor) / discount_factor)}")
        # print(f"stop iterations: {'Yes' if delta < theta * (1 - discount_factor) / discount_factor else 'No'}")
        # print()
        # if iterations > 1000 or stop:
        
    print("=================== RESULT ==================") 
    print("Iterations: " + str(iterations))
    print("Delta: " + str(delta))
    print("Gamma: " + str(discount_factor))
    print("Theta: " + str(theta)) 
    
    print("===================================================") 
    print(f"Policy array : {policy_array}")
        # 0: Move up 
        # 1: Move right
        # 2: Move down
        # 3: Move left

    print(f"Policy array {policy_array.reshape(4, 12)}")
    print_policy(policy_array, (4, 12))

        # if iterations > 5:
        #     break

if __name__ == "__main__":
    main()