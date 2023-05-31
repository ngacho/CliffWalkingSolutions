# TD-methods.

With MC it is necessary to wait until the end of the episode before updating the utility function. This is a problem because some applications can have long episodes with learning delayed to the end of each one. Moreover, in some environments the completion of the episode is not guaranteed.

In TD learning we want to update the utility function after each visit, therefore we do not have all the states and we do not have the values of the rewards. The only information available is rt+1 the reward at t+1 and the utilities estimated before.

General rule
NewEstimate ← OldEstimate + StepSize[Target − OldEstimate]

The Target is the expected return of the state.


TD trace offers early convergence. 




