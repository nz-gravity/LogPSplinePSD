import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy
from jax.scipy.stats import gamma as jax_gamma
from scipy.stats import gamma as scipy_gamma

x = np.linspace(0.01, 1, 100)
alpha = 1.3
beta = 3
key = jax.random.PRNGKey(0)


pdf = scipy_gamma.pdf(x, a=alpha, scale=1 / beta)
pdf_jax = np.array(jax_gamma.pdf(x, a=alpha, scale=1 / beta))
scipy_samples = scipy.stats.gamma.rvs(a=alpha, scale=1 / beta, size=1000)
jax_samples = jax.random.gamma(key, alpha, shape=(1000,)) * 1 / beta


np.testing.assert_allclose(pdf, pdf_jax, atol=1e-5)
plt.plot(
    x,
    pdf,
    color="green",
)
plt.plot(x, pdf_jax, linestyle="--", color="red")
plt.hist(scipy_samples, bins=100, alpha=0.25, color="green", density=True)
plt.hist(jax_samples, bins=100, alpha=0.25, color="red", density=True)
plt.show()
