from jax import random
from jax.scipy.stats import gamma as jax_gamma
from scipy.stats import gamma as scipy_gamma
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
alpha = 0.2
beta = 0.9
key = random.PRNGKey(0)


pdf = scipy_gamma.logpdf(x, a=alpha, scale=1 / beta)
pdf_jax = jax_gamma.logpdf(x, a=alpha, scale=1 / beta)
plt.plot(x, pdf)
plt.show()
