# Plots of value iterations
The following plots investigate the relationship between transition probability (the probability that an agent gets the action it chooses) v goal/punishment ratio for each learning rate. This is for the cliffwalking example shown in the main readme.

As you can see, for each learning ratio, the agent tends to work best when the goal/punishment is greater than 1 for all transition probabilities.

As you can see, when the transition probabilities are high, the values are high in areas that have the shortest probabilities to the goal. When the transition probabilities are low, the highest values are found near the goal and nowhere else. 