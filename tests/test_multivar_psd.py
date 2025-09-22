"""
Simple test for multivariate NUTS sampler with VARMAData.
"""

import pytest
import numpy as np
import tempfile

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.datatypes import MultivarFFT
from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines
from log_psplines.samplers.multivar.multivar_nuts import MultivarNUTSSampler, MultivarNUTSConfig
import os




def test_multivar_analysis(outdir):
    """Test basic multivariate PSD analysis with VARMA data."""
    outdir = f"{outdir}/test_multivar_analysis"
    os.makedirs(outdir, exist_ok=True)


    # Generate test data
    np.random.seed(42)
    varma = VARMAData(n_samples=256)
    x = varma.data
    n_dim = varma.dim

    print(f"VARMA data shape: {x.shape}, dim={n_dim}")

    # Convert to FFT
    fft_data = MultivarFFT.compute_fft(x, fs=1.0)
    print(f"FFT shapes: y_re={fft_data.y_re.shape}, Z_re={fft_data.Z_re.shape}")

    # Create multivariate P-splines model
    mv_model = MultivariateLogPSplines.from_multivar_fft(
        fft_data,
        n_knots=5,  # Small for fast testing
        degree=3,
        diffMatrixOrder=2
    )
    print(f"Created model: {mv_model}")

    # Set up sampler with temporary output directory
    config = MultivarNUTSConfig(
        verbose=True,
        outdir=outdir,
        target_accept_prob=0.8
    )
    sampler = MultivarNUTSSampler(fft_data, mv_model, config)

    # Run short sampling
    idata = sampler.sample(n_samples=50, n_warmup=50)

    # Basic checks
    assert idata is not None
    assert "posterior" in idata.groups()
    assert idata.posterior.sizes["draw"] == 50

    # Check key parameters exist
    assert "log_likelihood" in idata.sample_stats.data_vars

    # Check diagonal parameters
    for j in range(n_dim):
        assert f"delta_{j}" in idata.posterior.data_vars
        assert f"phi_delta_{j}" in idata.posterior.data_vars
        assert f"weights_delta_{j}" in idata.posterior.data_vars

    # Print some results
    ll_samples = idata.sample_stats["log_likelihood"].values.flatten()
    print(f"Log likelihood range: {ll_samples.min():.2f} to {ll_samples.max():.2f}")

    # Test PSD reconstruction if we have the required samples
    if all(key in idata.sample_stats.data_vars for key in ["log_delta_sq", "theta_re", "theta_im"]):
        psd_reconstructed = mv_model.reconstruct_psd_matrix(
            idata.sample_stats["log_delta_sq"].values,
            idata.sample_stats["theta_re"].values,
            idata.sample_stats["theta_im"].values,
            n_samples_max=4
        )
        print(f"Reconstructed PSD shape: {psd_reconstructed.shape}")

        # Basic sanity checks
        assert psd_reconstructed.shape[1] == fft_data.n_freq
        assert psd_reconstructed.shape[2] == n_dim
        assert psd_reconstructed.shape[3] == n_dim

    print("✓ Multivariate analysis test passed!")


