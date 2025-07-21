"""Convert inference results to ArviZ InferenceData format."""
import arviz as az
import numpy as np
from typing import Dict, Any
from xarray import DataArray, Dataset
from dataclasses import asdict


def results_to_arviz(
        samples: dict,
        sample_stats: dict,
        config: "BaseSampler",
        periodogram: "Periodogram",
        spline_model: "LogPSplines",
        attributes: Dict[str, Any],
) -> az.InferenceData:
    # Ensure all arrays have chain dimension
    def add_chain_dim(data_dict):
        return {k: np.array(v)[None, ...] if np.array(v).ndim <= 2 else np.array(v)
                for k, v in data_dict.items()}

    samples = add_chain_dim(samples)
    sample_stats = add_chain_dim(sample_stats)

    # Extract dimensions
    n_chains, n_draws, n_weights = samples["weights"].shape

    # Create posterior predictive samples
    weights_chain0 = samples['weights'][0]  # First chain
    n_pp = min(500, n_draws)
    pp_idx = np.random.choice(n_draws, n_pp, replace=False) if n_draws > n_pp else slice(None)

    pp_samples = np.array([spline_model(w) for w in weights_chain0[pp_idx]])
    pp_samples = np.exp(pp_samples)

    # Coordinates
    coords = {
        "chain": range(n_chains),
        "draw": range(n_draws),
        "pp_draw": range(n_pp),
        "weight_dim": range(n_weights),
        "freq": periodogram.freqs,
    }

    # Dimensions for each data group
    dims = {
        # Posterior
        "phi": ["chain", "draw"],
        "delta": ["chain", "draw"],
        "weights": ["chain", "draw", "weight_dim"],
        # Sample stats
        **{k: ["chain", "draw"] for k in sample_stats.keys()},
        # Observed data
        "periodogram": ["freq"],
        # Posterior predictive
        "psd": ["chain", "pp_draw", "freq"],
    }

    # Add log posterior if both likelihood and prior exist
    if {"log_likelihood", "log_prior"}.issubset(sample_stats.keys()):
        sample_stats["lp"] = sample_stats["log_likelihood"] + sample_stats["log_prior"]

    # Convert config to attributes (handle booleans)
    config_attrs = {k: int(v) if isinstance(v, bool) else v
                    for k, v in asdict(config).items()}
    attributes.update(config_attrs)
    attributes.update(dict(ess=az.ess(samples).to_array().values.flatten()))

    # Create InferenceData with custom posterior_psd group
    idata = az.from_dict(
        posterior=samples,
        sample_stats=sample_stats,
        observed_data={"periodogram": periodogram.power},
        dims={k: v for k, v in dims.items() if k != "psd"},
        coords={k: v for k, v in coords.items() if k != "pp_draw"},
        attrs=attributes,
    )

    # Add posterior predictive samples
    idata.add_groups(posterior_psd=Dataset(
        {"psd": DataArray(pp_samples, dims=["pp_draw", "freq"]),
         },
        coords={
            "pp_draw": coords["pp_draw"],
            "freq": coords["freq"],
        }))

    # Add spline model info
    idata.add_groups(spline_model=_pack_spline_model(spline_model))
    return idata


def _pack_spline_model(spline_model) -> Dataset:
    """Pack spline model parameters into xarray Dataset."""
    data = {
        "knots": (["knots_dim"], np.array(spline_model.knots)),
        "degree": spline_model.degree,
        "diffMatrixOrder": spline_model.diffMatrixOrder,
        "n": spline_model.n,
        "basis": (["freq", "weights_dim"], np.array(spline_model.basis)),
        "penalty_matrix": (["weights_dim_row", "weights_dim_col"], np.array(spline_model.penalty_matrix)),
        "parametric_model": (["freq"], np.array(spline_model.parametric_model)),
    }

    coords = {
        "knots_dim": np.arange(len(spline_model.knots)),
        "weights_dim": np.arange(spline_model.basis.shape[1]),
        "weights_dim_row": np.arange(spline_model.penalty_matrix.shape[0]),
        "weights_dim_col": np.arange(spline_model.penalty_matrix.shape[1]),
        "freq": np.arange(spline_model.basis.shape[0]),
    }

    return Dataset({
        k: DataArray(v[1], dims=v[0]) if isinstance(v, tuple) else DataArray(v)
        for k, v in data.items()
    }, coords=coords)
