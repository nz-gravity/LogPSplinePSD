"""Frequency reuse preserves initialization and the full posterior target."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.dw_wdm_small import benchmark as bench
from experiments.dw_wdm_small.gap_smoke import dw_data, masks
from numpyro.infer.util import potential_energy

from log_psplines import PowerConfig, PowerData
from log_psplines.inference.power import prepare_power_model

ORIGINAL = (
    Path(__file__).parent / "results/reuse_rerun/original_power_source.py"
)


def test_frequency_reuse_preserves_the_frequency_attached_to_every_time():
    x = bench.simulate(6101)
    master = bench.master_model()
    weights = np.random.default_rng(4441).normal(0, 0.15, master.weights.shape)
    cases = [bench.prepare(x, "dw")]
    cases.extend(dw_data(x, observed) for observed in masks().values())
    for data in cases:
        frequency, first, inverse = np.unique(
            data.frequency, return_index=True, return_inverse=True
        )
        np.testing.assert_array_equal(frequency[inverse], data.frequency)
        bt = master.time.design_at(data.time)
        bf = master.frequency.design_at(data.frequency)
        np.testing.assert_array_equal(bf[first][inverse], bf)
        original = np.einsum("pi,ij,pj->p", bt, weights, bf)
        reused = np.sum(bt * (bf[first] @ weights.T)[inverse], axis=1)
        np.testing.assert_allclose(original, reused, rtol=1e-12, atol=1e-12)


def original_preparer():
    spec = importlib.util.spec_from_file_location("original_power", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_power_model


@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize(
    "geometry", ["dw", "contiguous", "fragmented", "unique", "wdm"]
)
def test_posterior_value_gradient_and_initialization_are_preserved(
    centered, geometry
):
    x = bench.simulate(6101)
    if geometry in masks():
        data = dw_data(x, masks()[geometry])
    else:
        data = bench.prepare(x, "wdm" if geometry == "wdm" else "dw")
    if geometry == "unique":
        data = PowerData(
            data.power[:97],
            data.counts[:97],
            np.linspace(0.025, 0.475, 97),
            data.time[:97],
            data.units,
        )
    master = bench.master_model()
    model = (
        bench.on_grid(master, data.time, data.frequency)
        if data.is_grid
        else master
    )
    config = PowerConfig(centered=centered)
    old, old_init, old_pair = original_preparer()(data, model, config)
    new, new_init, new_pair = prepare_power_model(data, model, config)
    for key in old_init:
        np.testing.assert_array_equal(old_init[key], new_init[key])
    for key in old_pair:
        np.testing.assert_array_equal(old_pair[key], new_pair[key])

    start = np.concatenate(
        [
            old_init["s"],
            np.log([old_init["sigma_time"], old_init["sigma_freq"]]),
        ]
    )

    def gradient_of(target):
        def objective(q):
            return potential_energy(
                target,
                (),
                {},
                {"s": q[:-2], "sigma_time": q[-2], "sigma_freq": q[-1]},
            )

        return jax.jit(jax.value_and_grad(objective))

    old_vg, new_vg = gradient_of(old), gradient_of(new)
    rng = np.random.default_rng(4401)
    for multiplier in (0, 1, 5):
        perturbation = rng.normal(size=start.size) * 0.0003 * multiplier
        perturbation[-2:] *= 100
        a, b = (
            old_vg(jnp.asarray(start + perturbation)),
            new_vg(jnp.asarray(start + perturbation)),
        )
        for original, reused in zip(a, b, strict=True):
            np.testing.assert_allclose(original, reused, rtol=2e-12, atol=1e-8)
