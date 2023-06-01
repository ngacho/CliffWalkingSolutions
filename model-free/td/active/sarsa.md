In TD control the estimation is based on the tuple State-Action-Reward-State-Action hence the name SARSA.

The procedure is as follows
Move one step selecting at from π(st)
Observe: rt+1, st+1, at+1
Update the state-action function Q(st,at)
Update the policy π(st)← argmax aQ(st,at)

In algorithmic form for one episode:

``` python
# state-action values
state_action_values = np.zeros((action_len, state_len))
# initiate random policy
policy = np.random.randint(low=0, high=action_len, size=state_len, dtype="int64")
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
```

SARSA is FUCKING Insane. By far the fastest and most efficient way of solving cliffwalking solutions.
My algorithm shows that sarsa converges after just 17,371 episodes. THAT's magnitudes lower than anything we have had so far.