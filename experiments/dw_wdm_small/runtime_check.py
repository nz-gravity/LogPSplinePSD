"""Isolate PSD-evaluation cost; do not run another sampler."""

import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from experiments.dw_wdm_small.benchmark import master_model, prepare, simulate

from log_psplines import PSDResult
from log_psplines.likelihoods.whittle import power_whittle_log_likelihood

ROOT = Path(__file__).resolve().parent / "results" / "reuse_rerun"


def timing(data, *, grid: bool, frequency_reuse: bool = False) -> dict:
    model = master_model()
    if grid:
        bt = jnp.asarray(model.time.design_at(data.time))
        bf = jnp.asarray(model.frequency.design_at(data.frequency))

        def evaluate(w):
            return jnp.einsum("ti,ij,fj->tf", bt, w, bf, optimize="optimal")
    else:
        t = (
            np.repeat(data.time, len(data.frequency))
            if data.is_grid
            else data.time
        )
        f = (
            np.tile(data.frequency, len(data.time))
            if data.is_grid
            else data.frequency
        )
        bt = jnp.asarray(model.time.design_at(t))
        bf = jnp.asarray(model.frequency.design_at(f))

        if frequency_reuse:
            _, first, inverse = np.unique(
                f, return_index=True, return_inverse=True
            )
            unique_basis = bf[first]
            index = jnp.asarray(inverse)

            def evaluate(w):
                return jnp.sum(bt * (unique_basis @ w.T)[index], axis=1)
        else:

            def evaluate(w):
                return jnp.einsum("pi,ij,pj->p", bt, w, bf, optimize="optimal")

    shape = data.power.shape if grid else (-1,)
    power, counts = (
        jnp.asarray(data.power.reshape(shape)),
        jnp.asarray(data.counts.reshape(shape)),
    )

    def objective(w):
        return power_whittle_log_likelihood(power, counts, evaluate(w))

    value_grad = jax.jit(jax.value_and_grad(objective))
    w = jnp.zeros(model.weights.shape)
    value, grad = value_grad(w)
    grad.block_until_ready()

    @jax.jit
    def repeat(start):
        # Dependence on the previous weight prevents loop-invariant hoisting.
        return jax.lax.fori_loop(
            0, 2000, lambda _, a: a + 1e-9 * value_grad(a)[1], start
        )

    repeat(w).block_until_ready()
    measurements = []
    for _ in range(7):
        started = perf_counter()
        repeat(w).block_until_ready()
        measurements.append((perf_counter() - started) / 2000)
    return {
        "observations": int(np.count_nonzero(data.counts)),
        "real_components": float(data.counts.sum()),
        "value_and_gradient_us": float(1e6 * np.median(measurements)),
        "value": float(value),
        "gradient": np.asarray(grad).tolist(),
        "repeats": 2000,
        "rounds": 7,
    }


def main():
    x = simulate(6101)
    wdm, dw = prepare(x, "wdm"), prepare(x, "dw")
    results = {
        "wdm_grid": timing(wdm, grid=True),
        "wdm_scattered": timing(wdm, grid=False),
        "dw_scattered": timing(dw, grid=False),
        "dw_frequency_reuse": timing(dw, grid=False, frequency_reuse=True),
    }
    np.testing.assert_allclose(
        results["wdm_grid"]["value"],
        results["wdm_scattered"]["value"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        results["wdm_grid"]["gradient"],
        results["wdm_scattered"]["gradient"],
        rtol=1e-12,
        atol=1e-10,
    )
    for key in ("value", "gradient"):
        np.testing.assert_allclose(
            results["dw_scattered"][key],
            results["dw_frequency_reuse"][key],
            rtol=1e-12,
            atol=1e-10,
        )
    for values in results.values():
        del values["gradient"]
    for method in ("dw", "wdm"):
        steps = []
        acceptance = []
        for seed in range(6101, 6111):
            result = PSDResult.from_netcdf(
                ROOT / f"seed{seed}_{method}" / "posterior.nc"
            )
            steps.append(float(result.sample_stats.n_steps.mean()))
            acceptance.append(
                float(result.sample_stats.acceptance_rate.mean())
            )
        results[method + "_sampler"] = {
            "mean_retained_draw_leapfrog_steps": float(np.mean(steps)),
            "mean_acceptance": float(np.mean(acceptance)),
            "scope": "retained draws only; excludes warmup and compilation",
        }
    (ROOT / "runtime_check.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
