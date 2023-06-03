Applying the TD algorithm means to move step by step considering only the state at t and the state at t+1. 

That’s it, after each step we get the utility value and the reward at t+1 and we update the value at t. The TD(0) algorithm ignores the past states as shown by the shadow I added above those states.

Convergence is in the range of 370,000 - 420,000.

Note that the terminal values in our utility estimates are always one. This is because I normalized the actual utility values. The terminal states have no utilities and therefore they remain zero, evaluating to 1 when normalized.