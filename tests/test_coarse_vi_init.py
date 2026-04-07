import jax
import numpy as np

from log_psplines.datatypes.univar import Timeseries
from log_psplines.example_datasets.ar_data import ARData
from log_psplines.mcmc import (
    DiagnosticsConfig,
    ModelConfig,
    MultivariateTimeseries,
    RunMCMCConfig,
    VIConfig,
    run_mcmc,
)
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig
from log_psplines.preprocessing.preprocessing import _preprocess_data
from log_psplines.samplers.vi_init.mixin import VIInitialisationArtifacts


def _simulate_var2_3d(n: int, *, seed: int) -> np.ndarray:
    """Return (n, 3) samples from a stable VAR(2) process."""
    a1 = np.diag([0.4, 0.3, 0.2])
    a2 = np.array(
        [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]],
        dtype=np.float64,
    )
    sigma = np.array(
        [[0.25, 0.0, 0.08], [0.0, 0.25, 0.08], [0.08, 0.08, 0.25]],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    burn = 128
    noise = rng.multivariate_normal(np.zeros(3), sigma, size=n + burn)
    x = np.zeros((n + burn, 3), dtype=np.float64)
    for t in range(2, n + burn):
        x[t] = noise[t] + a1 @ x[t - 1] + a2 @ x[t - 2]
    return x[burn:]


def test_preprocess_builds_explicit_coarse_vi_context():
    data = ARData(order=2, duration=2.0, fs=128, sigma=0.5, seed=0).ts
    cfg = RunMCMCConfig(
        model=ModelConfig(n_knots=5),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            coarse_grain_config_vi={"enabled": True, "Nh": 4, "Nc": None},
        ),
    )

    preproc = _preprocess_data(data, config=cfg)

    assert preproc.coarse_vi_context is not None
    assert preproc.coarse_vi_context.metadata["coarse_vi_mode"] == "config"
    assert preproc.coarse_vi_context.metadata["coarse_vi_nfreq"] < len(
        preproc.processed_data.freqs
    )
    assert len(preproc.processed_data.freqs) == len(
        _preprocess_data(data, config=RunMCMCConfig()).processed_data.freqs
    )


def test_preprocess_adjusts_explicit_coarse_vi_nc_to_divisor():
    data = ARData(order=2, duration=2.0, fs=128, sigma=0.5, seed=10).ts
    cfg = RunMCMCConfig(
        model=ModelConfig(n_knots=5),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            coarse_grain_config_vi=CoarseGrainConfig(
                enabled=True,
                Nc=80,
                Nh=None,
            ),
        ),
    )

    preproc = _preprocess_data(data, config=cfg)

    assert preproc.coarse_vi_context is not None
    metadata = preproc.coarse_vi_context.metadata
    assert metadata["coarse_vi_mode"] == "config"
    full_nfreq = int(metadata["coarse_vi_full_nfreq"])
    coarse_nfreq = int(metadata["coarse_vi_nfreq"])
    assert full_nfreq % coarse_nfreq == 0
    assert coarse_nfreq <= 80
    assert metadata["coarse_vi_requested_Nc"] == 80
    assert metadata["coarse_vi_adjusted_Nc"] == coarse_nfreq


def test_preprocess_builds_auto_coarse_vi_context():
    data = ARData(order=2, duration=8.0, fs=256, sigma=0.5, seed=1).ts
    cfg = RunMCMCConfig(
        model=ModelConfig(n_knots=5),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            auto_coarse_vi=True,
            auto_coarse_vi_min_full_nfreq=128,
        ),
    )

    preproc = _preprocess_data(data, config=cfg)

    assert preproc.coarse_vi_context is not None
    assert preproc.coarse_vi_context.metadata["coarse_vi_mode"] == "auto"
    full_nfreq = len(preproc.processed_data.freqs)
    assert (
        preproc.coarse_vi_context.metadata["coarse_vi_full_nfreq"]
        == full_nfreq
    )
    target = preproc.coarse_vi_context.metadata["coarse_vi_target_nfreq"]
    assert target <= cfg.vi.auto_coarse_vi_target_nfreq
    coarse_nfreq = preproc.coarse_vi_context.metadata["coarse_vi_nfreq"]
    assert coarse_nfreq < full_nfreq
    assert coarse_nfreq <= target + 1  # rounding tolerance


def test_preprocess_skips_auto_coarse_vi_below_20k_threshold():
    data = ARData(order=2, duration=1.0, fs=256, sigma=0.5, seed=4).ts
    cfg = RunMCMCConfig(
        model=ModelConfig(n_knots=10, degree=3),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            auto_coarse_vi=True,
            auto_coarse_vi_min_full_nfreq=1,
        ),
    )

    preproc = _preprocess_data(data, config=cfg)

    assert preproc.coarse_vi_context is None


def test_univariate_coarse_vi_warm_start_keeps_full_grid():
    ar_data = ARData(order=2, duration=1.0, fs=128, sigma=0.5, seed=2)
    full_preproc = _preprocess_data(
        ar_data.ts,
        config=RunMCMCConfig(
            model=ModelConfig(n_knots=5),
            diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        ),
    )
    cfg = RunMCMCConfig(
        n_samples=3,
        n_warmup=3,
        num_chains=1,
        model=ModelConfig(n_knots=5),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            vi_steps=20,
            vi_lr=5e-2,
            vi_posterior_draws=8,
            coarse_grain_config_vi=CoarseGrainConfig(
                enabled=True, Nh=4, Nc=None
            ),
        ),
    )

    idata = run_mcmc(ar_data.ts, config=cfg)

    assert int(idata.attrs.get("coarse_vi_attempted", 0)) == 1
    assert int(idata.attrs.get("coarse_vi_success", 0)) == 1
    assert idata.posterior_psd.sizes["freq"] == len(
        full_preproc.processed_data.freqs
    )
    assert np.all(np.isfinite(idata.posterior_psd["psd"].values))


def test_univariate_coarse_vi_fallback_to_default_init(monkeypatch):
    import log_psplines.samplers.univar.nuts as nuts_mod

    def _bad_coarse_vi(*args, **kwargs):
        return VIInitialisationArtifacts(
            init_strategy=None,
            rng_key=jax.random.PRNGKey(123),
            diagnostics={
                "coarse_vi_attempted": 1,
                "coarse_vi_success": 0,
                "coarse_vi_mode": "config",
            },
        )

    monkeypatch.setattr(
        nuts_mod,
        "compute_coarse_vi_artifacts_univar",
        _bad_coarse_vi,
    )

    ts = ARData(order=2, duration=0.5, fs=64, sigma=0.5, seed=3).ts
    cfg = RunMCMCConfig(
        n_samples=2,
        n_warmup=2,
        num_chains=1,
        model=ModelConfig(n_knots=4),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            coarse_grain_config_vi=CoarseGrainConfig(
                enabled=True, Nh=2, Nc=None
            ),
        ),
    )

    idata = run_mcmc(ts, config=cfg)

    groups = (
        idata.groups()
        if callable(getattr(idata, "groups", None))
        else idata.groups
    )
    assert "/posterior" in groups
    assert int(idata.attrs.get("coarse_vi_attempted", 0)) == 1
    assert int(idata.attrs.get("coarse_vi_success", 0)) == 0


def test_multivariate_var2_3d_coarse_vi_warm_start_invariants():
    n = 128
    ts = MultivariateTimeseries(
        t=np.arange(n, dtype=np.float64),
        y=_simulate_var2_3d(n, seed=11),
    )
    full_preproc = _preprocess_data(
        ts,
        config=RunMCMCConfig(
            Nb=2,
            model=ModelConfig(n_knots=5),
            diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        ),
    )
    cfg = RunMCMCConfig(
        n_samples=3,
        n_warmup=3,
        num_chains=1,
        Nb=2,
        model=ModelConfig(n_knots=5),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            vi_steps=20,
            vi_lr=1e-2,
            vi_posterior_draws=8,
            coarse_grain_config_vi=CoarseGrainConfig(
                enabled=True, Nh=4, Nc=None
            ),
        ),
    )

    idata = run_mcmc(ts, config=cfg)

    assert int(idata.attrs.get("coarse_vi_attempted", 0)) == 1
    assert int(idata.attrs.get("coarse_vi_success", 0)) == 1
    assert idata.posterior_psd.sizes["freq"] == full_preproc.processed_data.N

    psd_real = np.asarray(
        idata.posterior_psd["psd_matrix_real"]
        .sel(percentile=50, method="nearest")
        .values
    )
    coherence = np.asarray(
        idata.posterior_psd["coherence"]
        .sel(percentile=50, method="nearest")
        .values
    )

    assert np.all(np.isfinite(psd_real))
    assert np.allclose(psd_real, np.swapaxes(psd_real, 1, 2), atol=1e-6)
    diag = np.diagonal(psd_real, axis1=1, axis2=2)
    assert np.all(diag > 0.0)
    assert np.nanmin(coherence) >= -1e-8
    assert np.nanmax(coherence) <= 1.0 + 1e-8


def test_multivariate_auto_coarse_vi_uses_max_component_basis_size():
    n = 1024
    ts = MultivariateTimeseries(
        t=np.arange(n, dtype=np.float64),
        y=_simulate_var2_3d(n, seed=21),
    )
    cfg = RunMCMCConfig(
        Nb=2,
        model=ModelConfig(
            n_knots={
                "delta": 5,
                "theta_re": 5,
                "theta_im": 9,
            },
            degree=3,
        ),
        diagnostics=DiagnosticsConfig(verbose=False, compute_lnz=False),
        vi=VIConfig(
            auto_coarse_vi=True,
            auto_coarse_vi_min_full_nfreq=1,
        ),
    )

    preproc = _preprocess_data(ts, config=cfg)

    assert preproc.coarse_vi_context is not None
    metadata = preproc.coarse_vi_context.metadata
    assert metadata["coarse_vi_mode"] == "auto"
    assert metadata["coarse_vi_basis_target_floor"] == 110
