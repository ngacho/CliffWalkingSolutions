
What matters in TD(0) is the current state and the state at t+1. To accelerate learning, it could be useful to extend what has been learned at t+1 to previous states.

When 0 < λ < 1 the traces decrease in time, giving a small weight to infrequent states.

For λ = 0, only the previous prediction is updated TD(0).
For λ = 1, we have TD(1) where all previous predicitons are equally updated.

TD(1) can be considered an extension of MC methods using a TD framework. 

In MC methods we need to wait the end of the episode in order to update the states. 
In TD(1), we do not need the end of the episode. We can update all the previous states online

This ensures an early convergence and learning is faster.

For numbers, running a 1,000,000 epochs shows that convergence for td_trace is at 393694.
