# TD-methods.

With MC it is necessary to wait until the end of the episode before updating the utility function. This is a problem because some applications can have long episodes with learning delayed to the end of each one. Moreover, in some environments the completion of the episode is not guaranteed.

General rule
NewEstimate ← OldEstimate + StepSize[Target − OldEstimate]

The Target is the expected return of the state.




