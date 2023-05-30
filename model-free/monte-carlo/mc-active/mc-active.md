## Monte-carlo solution

The monte-carlo solution is proving to be useful in solving the cliff-walking problem.

As we can see in the q-value heatmap, the q-values begin to converge early on.

Unlike other methods that I've had to catalyze convergence with a bigger reward, using q-values ensures that we get a converging policy.

Even though all terminal states have a value of 0 (which is high), the q-value algorithm is motivated to find a good early on (at about 100000) episodes.