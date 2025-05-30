


LogPSplinePSD
=============

Log-spline representation of the power spectral density (PSD) in the frequency domain, using penalized B-splines with a discrete penalty on spline coefficients.

**GitHub Repository**: https://github.com/nz-gravity/LogPSplinePSD


Overview
--------

`LogPSplinePSD` implements a Bayesian model for PSD estimation by fitting a log-spline to the periodogram. Main features:

- **Log-frequency representation**: Works on the log-scale of frequencies for numerical stability and improved resolution.
- **P-spline prior**: Applies a discrete difference penalty to log B-spline coefficients, enforcing smoothness in the log-PSD domain.
- **Whittle likelihood**: Employs Whittle's approximation for fast likelihood evaluation on periodogram ordinates.
- **HMC sampling**: Uses NumPyro (JAX) to perform efficient Hamiltonian Monte Carlo inference.

Methodology
-----------

The approach follows the P-spline framework for spectral density estimation described by Maturana-Russel & Meyer (2021) (arXiv:1905.01832):

1. **Basis construction**
   Define order-r B-spline basis functions \\(B_k(\\omega)\\), \\(k=1,\\dots,K+r\\), on an equidistant grid of interior knots in the log-frequency domain.

2. **Penalized prior**
   Apply a discrete \\(D\\)th-order difference penalty to the spline coefficients \\(\\{\\beta_k\\}\\), which induces smoothness in the estimated log-PSD.

3. **Knot placement (optional)**
   For spectra with sharp features, knot locations can be set based on quantiles of the raw periodogram values to allocate flexibility where needed.

4. **Model and likelihood**
   - The log-PSD is modeled as:
     \\[
     \\log f(\\lambda_l) = \\sum_k \\beta_k \\, B_k(\\log \\lambda_l).
     \\]
   - Whittle’s approximation for the periodogram \\(I_n(\\lambda_l)\\) yields the log-likelihood:
     \\[
     \\log L(\\beta) \\propto -\\sum_{l=1}^{\\nu} \\bigl[ \\log f(\\lambda_l) + I_n(\\lambda_l)/f(\\lambda_l) \\bigr].
     \\]

5. **Inference**
   Jointly sample the spline coefficients and penalty precision using NumPyro’s NUTS sampler.

This fixed-basis P-spline approach avoids reversible-jump MCMC over knot numbers and positions, reducing computational cost while retaining flexibility to capture complex spectral features.

Installation
------------

Clone the repository::

    git clone https://github.com/avivajpeyi/LogPSplinePSD.git
    cd LogPSplinePSD

Install dependencies via pip::

    pip install -r requirements.txt

Or with conda::

    conda env create -f environment.yml
    conda activate logpsplinepsd

Basic Usage
-----------

Save the following as `demo.py` and run to perform a quick demonstration of PSD fitting::

    import time
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from log_psplines.datasets import Periodogram
    from log_psplines.psplines import LogPSplines
    from log_psplines.bayesian_model import whittle_lnlike
    from log_psplines.mcmc import run_mcmc
    from log_psplines.plotting import plot_pdgrm, plot_trace

    # Generate sample periodogram
    freqs, power = Periodogram.generate_sample(n=1024, fs=1.0)
    pdgrm = Periodogram(freqs=freqs, power=power)

    # Initialize model
    model = LogPSplines.from_periodogram(pdgrm, n_knots=25, degree=3, diffMatrixOrder=2)

    # Compute initial log-likelihood
    lnl0 = whittle_lnlike(jnp.log(pdgrm.power), model(jnp.zeros(model.weights.shape)))
    print(f"Initial log-likelihood: {lnl0:.2f}")

    # Run MCMC sampling
    mcmc, _ = run_mcmc(pdgrm, n_knots=25, num_samples=500, num_warmup=500)

    # Plot and save results
    samples = mcmc.get_samples()
    fig1, _ = plot_pdgrm(pdgrm, model, samples['weights'])
    fig1.savefig('periodogram_fit.png')
    fig2 = plot_trace(mcmc)
    fig2.savefig('traceplot.png')
    print('Demo complete: periodogram_fit.png, traceplot.png')

Documentation
-------------

- `docs/` contains Sphinx sources for detailed API documentation.
- `examples/` holds Jupyter notebooks illustrating various workflows.

To build HTML documentation::

    cd docs
    make html

License
-------

MIT License. See `LICENSE`.

Author
------

Avi Vajpeyi – University of Auckland

Acknowledgements
----------------

Part of the NZ-Gravity and International LISA Consortium efforts on gravitational-wave data analysis.
