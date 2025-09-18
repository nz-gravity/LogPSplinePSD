import jax.numpy as jnp
import numpy as np
from log_psplines.psplines.initialisation import init_basis_and_penalty

# Test basis and penalty scaling
knots = jnp.linspace(0, 1, 10)
degree = 3
n_grid_points = 511
diff_matrix_order = 2

basis, penalty = init_basis_and_penalty(knots, degree, n_grid_points, diff_matrix_order)

print("Basis shape:", basis.shape)
print("Penalty shape:", penalty.shape)
print("Basis max:", basis.max())
print("Basis min:", basis.min())
print("Penalty max:", penalty.max())
print("Penalty min:", penalty.min())

# Simulate weights ~ Normal(0,1)
weights = np.random.normal(0, 1, basis.shape[1])
log_delta_sq = basis @ weights
print("log_delta_sq range:", log_delta_sq.min(), "to", log_delta_sq.max())
print("delta_sq (exp) range:", np.exp(log_delta_sq).min(), "to", np.exp(log_delta_sq).max())

# Simulate with regularization
from sklearn.linear_model import Ridge

# Ridge regression equivalent to the prior structure
ridge_alpha = 1.0  # based on phi=1, k*alpha = penalty strength

# But in the model, the prior is log_prior = 0.5 k log phi - 0.5 phi wPw

# To optimize, often use Maximum a posteriori or something
# But for simulation, let's see the scale
# If we assume phi =1, then it's N(0, 1/phi * (penalty)^{-1} ) but actually the prior is not standard normal.

# Anyway, the point is to see if with normalization, the scale is reasonable.
print("Example basis values at first few points:")
for i in range(min(10, basis.shape[0])):
    print(f"Point {i}: basis has values up to {basis[i].max()}")
