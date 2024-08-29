# Method 2: Bivariate model mixing

In this method, we use pointwise bivariate model mixing, or precision-weighted mixing, requiring models to be evaluated at every point in the input space
where we desire a prediction to be made. This method can be written succinctly as

$$
f_{\dagger} = \frac{1}{Z_P}\sum_{k=1}^{K} \frac{1}{v_k}f_k,
   \qquad Z_P \equiv \sum_{k=1}^{K}\frac{1}{v_k},
$$

where we can also define

$$
f_{\dagger} \sim \normal\bigl(Z_P^{-1}\sum_k \frac{1}{v_k}f_k, Z_P^{-1}\bigr).
$$

This method is precision-weighted because it uses the variances of the models at each point in the input space as the inverse weights of the corresponding model prediction, hence the model with the smallest variance at a given point will dominate the mixed model at that location.

::: samba.discrepancy
