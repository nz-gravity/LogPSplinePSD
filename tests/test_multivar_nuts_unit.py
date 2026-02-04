from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

import log_psplines.samplers.multivar.multivar_nuts as multivar_nuts
from log_psplines.samplers.multivar.multivar_nuts import (
    MultivarNUTSConfig,
    MultivarNUTSSampler,
    multivariate_psplines_model,
)


def _make_sampler(num_chains=2):
    sampler = MultivarNUTSSampler.__new__(MultivarNUTSSampler)
    sampler.config = MultivarNUTSConfig(num_chains=num_chains, verbose=True)
    sampler.config.outdir = None
    sampler.rng_key = jax.random.PRNGKey(0)
    sampler.chain_method = "sequential"
    sampler.device = "cpu"
    sampler.n_channels = 2
    sampler._logpost_fn = lambda params: jnp.array(0.0)
    sampler.u_re = jnp.zeros((2, 2, 2))
    sampler.u_im = jnp.zeros((2, 2, 2))
    sampler.nu = 2
    sampler.all_bases = (jnp.ones((2, 1)),)
    sampler.all_penalties = (jnp.eye(1),)
    sampler.freq_weights = jnp.ones((2,))
    sampler.freq_bin_counts = jnp.ones((2,))
    sampler._default_init_strategy = lambda: "init"
    sampler._compute_empirical_psd = lambda: None
    sampler._save_vi_diagnostics = lambda empirical_psd=None: None
    return sampler


def test_multivariate_model_emits_sites_univariate_and_multivar(monkeypatch):
    def fake_block(*args, penalty_matrix=None, **kwargs):
        k = penalty_matrix.shape[0]
        return {"weights": jnp.ones((k,))}

    monkeypatch.setattr(multivar_nuts, "sample_pspline_block", fake_block)

    def _run_and_assert(n_dim, n_theta):
        n_freq = 4
        u_re = jnp.zeros((n_freq, n_dim, n_dim))
        u_im = jnp.zeros((n_freq, n_dim, n_dim))
        all_bases = [
            jnp.ones((n_freq, 3)) for _ in range(n_dim + 2 * (n_theta > 0))
        ]
        all_penalties = [jnp.eye(3) for _ in all_bases]
        freq_weights = jnp.ones((n_freq,))
        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(
                multivariate_psplines_model, jax.random.PRNGKey(0)
            )
        ).get_trace(
            u_re=u_re,
            u_im=u_im,
            nu=2,
            all_bases=all_bases,
            all_penalties=all_penalties,
            freq_weights=freq_weights,
            freq_bin_counts=freq_weights,
        )

        assert "likelihood" in trace
        assert "log_delta_sq" in trace
        assert "theta_re" in trace
        assert "theta_im" in trace
        assert "log_likelihood" in trace
        assert trace["log_delta_sq"]["value"].shape == (n_freq, n_dim)
        assert trace["theta_re"]["value"].shape == (n_freq, n_theta)
        assert trace["theta_im"]["value"].shape == (n_freq, n_theta)

    _run_and_assert(n_dim=1, n_theta=0)
    _run_and_assert(n_dim=2, n_theta=1)


def test_default_init_strategy_calls_default_init_values(monkeypatch):
    sampler = MultivarNUTSSampler.__new__(MultivarNUTSSampler)
    sampler.config = MultivarNUTSConfig(num_chains=1, verbose=True)
    sampler.spline_model = object()
    called = {}

    def fake_default_init_values_multivar(
        spline_model, alpha_phi, beta_phi, alpha_delta, beta_delta
    ):
        called["args"] = (
            spline_model,
            alpha_phi,
            beta_phi,
            alpha_delta,
            beta_delta,
        )
        return {"weights_delta_0": jnp.zeros(2)}

    monkeypatch.setattr(
        multivar_nuts,
        "default_init_values_multivar",
        fake_default_init_values_multivar,
    )
    init_strategy = sampler._default_init_strategy()
    assert callable(init_strategy)
    assert called["args"][0] is sampler.spline_model


def test_compile_model_success_does_not_raise(monkeypatch):
    sampler = _make_sampler(num_chains=1)
    sampler.spline_model = object()

    def fake_init(*args, **kwargs):
        return {}

    monkeypatch.setattr(
        multivar_nuts, "default_init_values_multivar", lambda *a, **k: {}
    )
    monkeypatch.setattr("numpyro.infer.util.initialize_model", fake_init)
    sampler._compile_model()


def test_compile_model_failure_does_not_raise(monkeypatch):
    sampler = _make_sampler(num_chains=1)
    sampler.spline_model = object()

    def fake_init(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        multivar_nuts, "default_init_values_multivar", lambda *a, **k: {}
    )
    monkeypatch.setattr("numpyro.infer.util.initialize_model", fake_init)
    sampler._compile_model()


def test_sample_real_path_moves_deterministics_and_reshapes_lp(monkeypatch):
    sampler = _make_sampler(num_chains=2)
    captured = {}

    def fake_to_arviz(samples, stats):
        captured["samples"] = samples
        captured["stats"] = stats
        return {"samples": samples, "stats": stats}

    sampler.to_arviz = fake_to_arviz

    class DummyArtifacts:
        def __init__(self):
            self.rng_key = jax.random.PRNGKey(0)
            self.init_strategy = None
            self.diagnostics = {"losses": np.array([1.0, 0.5]), "guide": "foo"}
            self.posterior_draws = None
            self.means = None

    monkeypatch.setattr(
        multivar_nuts,
        "compute_vi_artifacts_multivar",
        lambda *a, **k: DummyArtifacts(),
    )

    class DummyNUTS:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class DummyMCMC:
        last_extra_fields = None

        def __init__(
            self, kernel, num_warmup, num_samples, num_chains, **kwargs
        ):
            self.num_samples = num_samples
            self.num_chains = num_chains

        def run(self, *args, **kwargs):
            DummyMCMC.last_extra_fields = kwargs.get("extra_fields", ())

        def get_samples(self, group_by_chain=True):
            n_chains = self.num_chains
            n_draws = self.num_samples
            return {
                "weights_delta_0": jnp.ones((n_chains, n_draws, 2)),
                "phi_delta_0": jnp.ones((n_chains, n_draws)),
                "delta_0": jnp.ones((n_chains, n_draws)),
                "log_delta_sq": jnp.zeros((n_chains, n_draws, 2, 2)),
                "theta_re": jnp.zeros((n_chains, n_draws, 2, 1)),
                "theta_im": jnp.zeros((n_chains, n_draws, 2, 1)),
                "log_likelihood": jnp.zeros((n_chains, n_draws)),
                "lp": jnp.zeros((n_chains, n_draws)),
            }

        def get_extra_fields(self, group_by_chain=True):
            return {
                "accept_prob": jnp.full(
                    (self.num_chains, self.num_samples), 0.8
                )
            }

    monkeypatch.setattr(multivar_nuts, "NUTS", DummyNUTS)
    monkeypatch.setattr(multivar_nuts, "MCMC", DummyMCMC)
    monkeypatch.setattr(
        multivar_nuts,
        "evaluate_log_density_batch",
        lambda *a, **k: np.arange(6, dtype=float),
    )

    result = sampler.sample(n_samples=3, n_warmup=2, only_vi=False)
    assert isinstance(result, dict)
    samples = captured["samples"]
    stats = captured["stats"]

    assert "log_delta_sq" not in samples
    assert "theta_re" not in samples
    assert "theta_im" not in samples
    assert "log_likelihood" not in samples
    assert "log_delta_sq" in stats
    assert "theta_re" in stats
    assert "theta_im" in stats
    assert "log_likelihood" in stats
    assert stats["lp"].shape == (2, 3)
    assert np.allclose(
        np.asarray(samples["phi_delta_0"]), np.exp(np.ones((2, 3)))
    )


def test_sample_real_path_without_extra_fields(monkeypatch):
    sampler = _make_sampler(num_chains=1)
    sampler.config.save_nuts_diagnostics = False

    sampler.to_arviz = lambda samples, stats: {
        "samples": samples,
        "stats": stats,
    }

    class DummyArtifacts:
        def __init__(self):
            self.rng_key = jax.random.PRNGKey(0)
            self.init_strategy = None
            self.diagnostics = {}
            self.posterior_draws = None
            self.means = None

    monkeypatch.setattr(
        multivar_nuts,
        "compute_vi_artifacts_multivar",
        lambda *a, **k: DummyArtifacts(),
    )

    class DummyNUTS:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class DummyMCMC:
        last_extra_fields = None

        def __init__(
            self, kernel, num_warmup, num_samples, num_chains, **kwargs
        ):
            self.num_samples = num_samples
            self.num_chains = num_chains

        def run(self, *args, **kwargs):
            DummyMCMC.last_extra_fields = kwargs.get("extra_fields", ())

        def get_samples(self, group_by_chain=True):
            return {"weights_delta_0": jnp.ones((1, 1, 2))}

        def get_extra_fields(self, group_by_chain=True):
            return {}

    monkeypatch.setattr(multivar_nuts, "NUTS", DummyNUTS)
    monkeypatch.setattr(multivar_nuts, "MCMC", DummyMCMC)
    monkeypatch.setattr(
        multivar_nuts,
        "evaluate_log_density_batch",
        lambda *a, **k: np.array([0.0]),
    )

    sampler.sample(n_samples=1, n_warmup=1, only_vi=False)
    assert DummyMCMC.last_extra_fields == ()


def test_prepare_logpost_params_flattens_inputs():
    sampler = _make_sampler(num_chains=1)
    samples = {
        "weights_delta_0": jnp.ones((2, 3, 4)),
        "phi_delta_0": jnp.ones((2, 3)),
        "delta_0": jnp.array(1.0),
        "other": jnp.zeros((2,)),
    }
    flat = sampler._prepare_logpost_params(samples)
    assert flat["weights_delta_0"].shape == (6, 4)
    assert flat["phi_delta_0"].shape == (6,)
    assert flat["delta_0"].shape == (1,)
    assert "other" not in flat


def test_compute_log_posterior_logs_phi():
    sampler = _make_sampler(num_chains=1)

    def fake_logpost(params):
        assert np.allclose(
            np.asarray(params["phi_delta_0"]),
            np.log(np.array([2.0, 3.0])),
        )
        assert np.allclose(
            np.asarray(params["weights_delta_0"]), np.array([1.0, 2.0])
        )
        return jnp.array(0.0)

    sampler._logpost_fn = fake_logpost
    params = {
        "phi_delta_0": jnp.array([2.0, 3.0]),
        "weights_delta_0": jnp.array([1.0, 2.0]),
    }
    assert sampler._compute_log_posterior(params) == 0.0


def test_vi_only_inference_data_with_draws(monkeypatch):
    sampler = _make_sampler(num_chains=1)
    captured = {}

    def fake_create(samples, sample_stats, diagnostics):
        captured["samples"] = samples
        captured["stats"] = sample_stats
        captured["diagnostics"] = diagnostics
        return {"samples": samples, "stats": sample_stats}

    sampler._create_vi_inference_data = fake_create

    monkeypatch.setattr(
        multivar_nuts,
        "evaluate_log_density_batch",
        lambda *a, **k: np.array([1.0, 2.0]),
    )
    sampler._logpost_fn = lambda params: jnp.array(0.0)

    artifacts = SimpleNamespace(
        diagnostics={},
        posterior_draws={
            "weights_delta_0": jnp.ones((2, 2)),
            "phi_delta_0": jnp.ones((2,)),
            "delta_0": jnp.ones((2,)),
        },
        means=None,
    )

    result = sampler._vi_only_inference_data(artifacts)
    assert result["stats"]["lp"].shape == (1, 2)
    assert np.allclose(
        np.asarray(captured["samples"]["phi_delta_0"]), np.exp(np.ones((2,)))
    )
    assert sampler.runtime == 0.0


def test_vi_only_inference_data_means_fallback_and_lp_failure(monkeypatch):
    sampler = _make_sampler(num_chains=1)
    captured = {}

    sampler._create_vi_inference_data = (
        lambda samples, sample_stats, diagnostics: captured.update(
            {"samples": samples, "stats": sample_stats}
        )
        or {"samples": samples, "stats": sample_stats}
    )

    def fake_eval(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(multivar_nuts, "evaluate_log_density_batch", fake_eval)
    sampler._logpost_fn = lambda params: jnp.array(0.0)

    artifacts = SimpleNamespace(
        diagnostics={},
        posterior_draws=None,
        means={
            "weights_delta_0": jnp.ones((2,)),
            "phi_delta_0": jnp.ones((2,)),
            "delta_0": jnp.ones((2,)),
        },
    )

    result = sampler._vi_only_inference_data(artifacts)
    assert result["stats"] == {}
    assert captured["samples"]["weights_delta_0"].shape[0] == 1


def test_vi_only_inference_data_requires_draws_or_means():
    sampler = _make_sampler(num_chains=1)
    artifacts = SimpleNamespace(diagnostics={}, posterior_draws=None, means={})
    with pytest.raises(ValueError):
        sampler._vi_only_inference_data(artifacts)
