# Method 1: Linear model mixing

This method is derived from [Coleman's thesis](https://dukespace.lib.duke.edu/items/80db7117-94a7-45cc-83c6-1489c090821f), and 
uses a mixing function with hyperparameters to be estimated using data, to construct the mixed model. There are several possible mixing functions given in the code below for a user to play with and build off of to write their own mixing function. There are also priors to choose from (in the `priors.py` file) for the hyperparameters of each mixing function.

Once the mixing function has been chosen, and data supplied or simulated, the user can construct the mixed model by sampling the parameter space using the sampler wrapper below, and then building the posterior predictive distribution (PPD). This is given as

$$
p(\tilde y(g)|\theta, \mathbf{D}) = \sum_{j=1}^{M} \alpha(g; \theta_{j}) F^{N_s}_s(g) + (1 - \alpha(g; \theta_{j})) F^{N_l}_{l}(g),
$$

where $\alpha(g; \theta_{j})$ is the chosen mixing function with hyperparameters $\theta_{j}$.

::: samba.mixing
