# Plots of value iterations
The following plots investigate the relationship between transition probability (the probability that an agent gets the action it chooses) v goal/punishment ratio for each learning rate. This is for the cliffwalking example shown in the main readme.

As you can see, for each learning ratio, the agent tends to work best when the goal/punishment is greater than 1 for all transition probabilities.

When the transition probabilities:
    - Are high
        - and the reward/punishment ratio is low, 
            - an agent following a policy using the highest values around it favors no movement due to low rewards.
            - note the even distribution of greens.
        - and the reward/punishment ratio is high,
            - the agent favors a policy that will get it to the reward as fast as possible.
            - note the distribution of greens close to the cliff
    - Are low
        - and the reward/punishment ratio is low, 
            - an agent following a policy using the highest values around it favors no movement as well due to low rewards
            - not the lighter evenly distributed greens
        -  and the reward/punishment ratio is high.
            - the agent favors policies that get it close to the reward using longer paths.
            - sometimes the agent favors no movement, the risk is tooo high.
                - note the distribution of greens in the bottom right.
