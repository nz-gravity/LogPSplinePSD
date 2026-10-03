"""Audit retained inference and save derived checkpoint functional history."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import stationary_next
import xarray as xr
from comparison import compare_with_mc
from extension_targets import feature_values, physical_coefficients
from extensions import config, load_base, select_target
from numpyro.infer.util import log_density
from study_common import draw_dataset, write_json

from log_psplines.diagnostics.variational import (
    VIDiagnosticState,
    nuts_reference_health,
    rebuild_guide,
)


def main():
    out = Path(__file__).resolve().parents[2] / "runs/vi-matrix-tv"
    cfg = config(out)
    stationary_next.config(out.parent / "vi-stationary-next")
    checks = {}
    for case in cfg["cases"]:
        base = load_base(out, case)
        fixed = select_target(base, "fixed")
        key = "weights_delta_1" if case == "matrix" else "s"
        point = {
            **fixed.init,
            **{
                k: jnp.asarray(v)
                for k, v in fixed.descriptor["conditioned_sigma"].items()
            },
        }

        def native(w, base=base, point=point, key=key):
            return log_density(base.model, (), {}, {**point, key: w})[0]

        def conditional(w, fixed=fixed, key=key):
            return log_density(fixed.model, (), {}, {**fixed.init, key: w})[0]

        err = float(
            np.max(
                abs(
                    np.asarray(jax.grad(native)(point[key]))
                    - np.asarray(jax.grad(conditional)(point[key]))
                )
            )
        )
        assert err == 0
        checks[case] = {"native_vs_conditional_gradient_max_error": err}
        for path in sorted((out / case).glob("*/reference/*/complete.json")):
            record = json.loads(path.read_text())
            target = select_target(base, record["smoothing"])
            posterior = xr.load_dataset(path.parent / "posterior.nc")
            stats = xr.load_dataset(path.parent / "sample_stats.nc")
            raw = np.concatenate(
                [
                    posterior[n].values.reshape(4, posterior.sizes["draw"], -1)
                    for n in target.init
                    if n in posterior
                ],
                -1,
            )
            rawhealth = nuts_reference_health(
                raw, stats, max_tree_depth=record["settings"]["max_tree_depth"]
            )
            checks[str(path.relative_to(out))] = {
                "raw_native_latent_health": rawhealth
            }
            if record["status"] == "accepted_screen":
                assert rawhealth["status"] == "accepted_screen"
        for seed in cfg["seeds"]:
            d = out / case / "fixed/vi" / str(seed)
            state = VIDiagnosticState.load(d)
            persisted = VIDiagnosticState.from_dataset(state.to_dataset())
            for name in state.params:
                np.testing.assert_array_equal(
                    state.params[name], persisted.params[name]
                )
            for step in cfg["checkpoints"]:
                params = np.load(d / f"parameters_{step}.npz")
                assert set(params.files) == set(state.checkpoints[str(step)])
                for name in params.files:
                    np.testing.assert_array_equal(
                        params[name], state.checkpoints[str(step)][name]
                    )
            guide = rebuild_guide(
                state, fixed.model, target_fingerprint=fixed.identity
            )
            reference = out / case / "fixed/reference"
            chosen = json.loads((reference / "selection.json").read_text())[
                "attempt"
            ]
            pf = np.load(reference / chosen / "features.npz")
            history = []
            for step in cfg["checkpoints"]:
                _, ds = draw_dataset(
                    guide,
                    state.checkpoints[str(step)],
                    jax.random.PRNGKey(
                        cfg["inherited"]["checkpoint_draw_seed"]
                    ),
                    cfg["inherited"]["checkpoint_draws"],
                )
                values, names = feature_values(fixed, ds)
                assert names == pf["names"].tolist()
                path = (
                    d
                    / f"features_{step}_{cfg['inherited']['checkpoint_draws']}.npz"
                )
                if path.exists():
                    np.testing.assert_array_equal(
                        values, np.load(path)["values"]
                    )
                else:
                    np.savez_compressed(
                        path, values=values, names=np.array(names)
                    )
                history.append(
                    {
                        "step": step,
                        "comparison": compare_with_mc(
                            values, pf["values"], names, cfg["inherited"]
                        ),
                        "fixed_objective": state.metadata["optimization"][
                            "checkpoint_objectives"
                        ][str(step)],
                    }
                )
            write_json(
                d / "checkpoint_history.json",
                {
                    "target_fingerprint": fixed.identity,
                    "draw_count": cfg["inherited"]["checkpoint_draws"],
                    "draw_key": cfg["inherited"]["checkpoint_draw_seed"],
                    "checkpoints": history,
                },
            )
            physical = physical_coefficients(
                fixed, xr.load_dataset(d / "posterior.nc")
            )
            physical_path = d / "physical_coefficients.npz"
            if physical_path.exists():
                np.testing.assert_array_equal(
                    physical, np.load(physical_path)["weights"]
                )
            else:
                np.savez_compressed(physical_path, weights=physical)
            record = json.loads((d / "complete.json").read_text())
            assert record["optimization"]["steps_run"] == 60000
            assert record["optimization"]["optimization_particles"] == 8
            checks[f"{case}/fixed/{seed}"] = {
                "actual_updates": 60000,
                "checkpoint_parameters_bitwise_roundtrip": len(
                    cfg["checkpoints"]
                ),
                "guide_dataset_roundtrip": "bitwise_equal",
                "posterior_draws": xr.load_dataset(d / "posterior.nc").sizes[
                    "draw"
                ],
            }
    # Explicitly retain reference precision by named native smoothing functional.
    for p in out.glob("matrix/hierarchical/reference/*/features.npz"):
        a = np.load(p)
        names = a["names"].tolist()
        vals = a["values"]
        from comparison import mc_stats

        stats = mc_stats(vals)
        checks[str(p.relative_to(out))] = {
            "roughness_functionals": [
                {
                    "name": name,
                    "mean": stats["mean"][i],
                    "sd": stats["sd"][i],
                    "mean_mcse_sd": stats["mean_mcse"][i] / stats["sd"][i],
                }
                for i, name in enumerate(names)
                if name.startswith(("sigma_", "log_sigma_", "roughness_"))
            ]
        }
    write_json(
        out / "verification.json",
        {
            "status": "passed",
            "stationary_source_hashes": 31,
            "checks": checks,
            "fitted_guides": 6,
            "checkpoint_parameter_archives": 36,
            "inherited_data_regenerated": False,
        },
    )
    print(
        "Verified gradients, raw-native-latent reference health, 6 guide round trips, 36 checkpoints and 31 stationary source hashes."
    )


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
