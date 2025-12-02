import arviz as az
import numpy as np

from log_psplines.arviz_utils.rhat import extract_rhat_values
from log_psplines.samplers.base_sampler import BaseSampler, SamplerConfig


class _DummySampler(BaseSampler):
    """Minimal BaseSampler subclass to exercise Rhat handling."""

    def _setup_data(self) -> None:  # pragma: no cover - nothing to initialise
        return

    def _get_lnz(self, samples, sample_stats):
        return np.nan, np.nan

    @property
    def sampler_type(self) -> str:
        return "dummy"

    @property
    def data_type(self) -> str:
        return "dummy"

    def sample(
        self, n_samples: int, n_warmup: int = 1000, *, only_vi=False, **kwargs
    ):
        raise NotImplementedError("Not used in tests")

    def _save_plots(self, idata: az.InferenceData) -> None:
        return

    def _create_inference_data(
        self,
        samples,
        sample_stats,
        lnz: float,
        lnz_err: float,
    ) -> az.InferenceData:
        attrs = dict(
            device="cpu",
            runtime=0.0,
            lnz=lnz,
            lnz_err=lnz_err,
            sampler_type=self.sampler_type,
            data_type=self.data_type,
            ess=np.array([10.0, 20.0]),
        )
        return az.from_dict(
            posterior=samples,
            sample_stats=sample_stats,
            attrs=attrs,
        )


def test_extract_rhat_values_handles_disjoint_dims():
    rng = np.random.default_rng(0)
    idata = az.from_dict(
        posterior={
            "a": rng.normal(size=(2, 6, 3)),
            "b": rng.normal(size=(2, 6, 4)),
        }
    )

    rhat_vals = extract_rhat_values(idata)

    assert rhat_vals.size == 7  # 3 entries for a, 4 for b
    assert np.all(np.isfinite(rhat_vals))


def test_base_sampler_attaches_rhat_for_multiple_chains():
    rng = np.random.default_rng(1)
    samples = {
        "weights_a": rng.normal(size=(2, 5, 3)),
        "weights_b": rng.normal(size=(2, 5, 2)),
    }
    sample_stats = {"log_likelihood": rng.normal(size=(2, 5))}

    sampler = _DummySampler(
        None, None, SamplerConfig(num_chains=2, verbose=False)
    )
    idata = sampler.to_arviz(samples, sample_stats)

    assert "rhat" in idata.attrs
    rhat_vals = idata.attrs["rhat"]
    assert isinstance(rhat_vals, np.ndarray)
    assert rhat_vals.size == 5  # matches total parameter entries
    assert np.all(np.isfinite(rhat_vals))
