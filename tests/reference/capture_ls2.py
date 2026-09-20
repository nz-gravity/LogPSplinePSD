"""Capture an LS2/WDM reference from a supplied source checkout.

Run with this repo's .venv and the optional WDM dependency installed:
    .venv/bin/python tests/reference/capture_ls2.py /path/to/wdm_psd
Extract source functions verbatim so unrelated optional frontend imports are
not needed. Records source hashes, observed powers, operators and NUTS draws.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import log_density
from scipy import interpolate
from wdm_transform import TimeSeries

jax.config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("--warmup", type=int, default=24)
parser.add_argument("--draws", type=int, default=16)
parser.add_argument("--max-tree-depth", type=int, default=6)
parser.add_argument("--output", type=Path, default=Path(__file__).parent)
options = parser.parse_args()
root = options.source / "tv_pspline_psd"
ns = dict(
    np=np,
    jnp=jnp,
    numpyro=numpyro,
    dist=dist,
    PSplineConfig=SimpleNamespace,
    interpolate=interpolate,
    TimeSeries=TimeSeries,
)
hashes = {}
for filename, names in {
    "datasets/ls2.py": ["simulate_ls2", "true_psd_ls2"],
    "splines.py": [
        "create_bspline_basis",
        "_evaluate_basis",
        "create_bspline_roughness_penalty",
    ],
    "inference.py": [
        "_wdm_coeffs_2d",
        "wdm_analysis_coefficients",
        "_trimmed_wdm_analysis_grid",
    ],
    "model.py": [
        "tensor_product_surface",
        "power_floor",
        "whiten_penalty_pair",
        "_sample_log_gamma",
        "_sample_smoothing_precision",
        "eigen_prior_scale",
        "sample_eigen_coefficients",
        "sample_tensor_eigen_coefficients",
        "power_whittle_log_likelihood",
        "pspline_surface_model",
        "initialize_with_penalized_least_squares",
        "whitened_init_values",
    ],
}.items():
    source = (root / filename).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.get_source_segment(source, node)
            hashes[f"{filename}:{node.name}"] = hashlib.sha256(
                code.encode()
            ).hexdigest()
            exec(code, ns)
config = SimpleNamespace(
    alpha_phi=2.0,
    beta_phi=1.0,
    phi_log_base_scale=1.0,
    smoothing_prior="gamma",
    null_precision=1e-4,
    ridge_eps=1e-6,
    init_penalty_time=0.05,
    init_penalty_freq=0.05,
    centered=False,
    trim_time_bins=1,
    trim_low_freq_channels=1,
    trim_high_freq_channels=1,
)
x = ns["simulate_ls2"](512, rng=np.random.default_rng(0))
c, t, f = ns["wdm_analysis_coefficients"](x, 0.1, 32, config)
Bt, kt = ns["create_bspline_basis"](t, 4)
Bf, kf = ns["create_bspline_basis"](f / f[-1], 4)
Pt = ns["create_bspline_roughness_penalty"](kt, degree=3)
Pf = ns["create_bspline_roughness_penalty"](kf, degree=3)
pair = ns["whiten_penalty_pair"](Pt, Pf)
args = tuple(
    jnp.asarray(a)
    for a in (
        c**2,
        np.ones_like(c),
        Bt @ pair["U_time"],
        Bf @ pair["U_freq"],
        pair["lam_time"],
        pair["lam_freq"],
        pair["joint_null"],
    )
)
arrays = dict(
    data=x,
    power=c**2,
    time=t,
    frequency=f,
    basis_time=Bt,
    basis_freq=Bf,
    penalty_time=Pt,
    penalty_freq=Pf,
    knots_time=kt,
    knots_freq=kf,
    truth=ns["true_psd_ls2"](t, f, 0.1),
)
for centered in (False, True):
    config.centered = centered
    prefix = "centered_" if centered else "noncentered_"
    pls = ns["initialize_with_penalized_least_squares"](
        c**2, Bt, Bf, Pt, Pf, config
    )
    init = ns["whitened_init_values"](pls, pair, config)

    def target(sites):
        return log_density(
            ns["pspline_surface_model"], (*args, config, False), {}, sites
        )[0]

    value, gradient = jax.value_and_grad(target)(init)
    arrays[prefix + "log_density"] = value
    for key in init:
        arrays[prefix + "init_" + key] = init[key]
        arrays[prefix + "gradient_" + key] = gradient[key]
    sampler = MCMC(
        NUTS(
            ns["pspline_surface_model"],
            init_strategy=init_to_value(values=init),
            max_tree_depth=options.max_tree_depth,
            target_accept_prob=0.85,
        ),
        num_warmup=options.warmup,
        num_samples=options.draws,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
    )
    sampler.run(
        jax.random.PRNGKey(7),
        *args,
        config,
        False,
        extra_fields=(
            "diverging",
            "num_steps",
            "accept_prob",
            "potential_energy",
            "energy",
        ),
    )
    for key, value in sampler.get_samples(group_by_chain=True).items():
        arrays[prefix + "sample_" + key] = value
    for key, value in sampler.get_extra_fields(group_by_chain=True).items():
        arrays[prefix + "stat_" + key] = value
    source_samples = sampler.get_samples()
    source_psd = []
    for draw in range(options.draws):
        sites = {
            key: source_samples[key][draw]
            for key in ("s", "phi_time", "phi_freq")
        }
        traced = numpyro.handlers.trace(
            numpyro.handlers.seed(
                numpyro.handlers.substitute(
                    ns["pspline_surface_model"], data=sites
                ),
                0,
            )
        ).get_trace(*args, config, True)
        source_psd.append(np.exp(np.asarray(traced["log_psd"]["value"])))
    arrays[prefix + "psd"] = np.asarray(source_psd)[None]
    print(prefix, "complete", flush=True)
path = options.output
path.mkdir(parents=True, exist_ok=True)
np.savez_compressed(path / "ls2_wdm.npz", **arrays)
(path / "ls2_wdm.json").write_text(
    json.dumps(
        dict(
            source=str(root),
            function_hashes=hashes,
            n=512,
            dt=0.1,
            nt=32,
            interior_knots=4,
            warmup=options.warmup,
            draws=options.draws,
            seed=7,
            max_tree_depth=options.max_tree_depth,
            jax=jax.__version__,
            numpyro=numpyro.__version__,
            scope="Numerical parity check, not a calibrated recovery study",
        ),
        indent=2,
    )
    + "\n"
)
