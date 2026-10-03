"""Inherited posterior MC uncertainty and stability screens.

Numerical definitions are shared by stationary and extension studies. Feature
roles are explicit; this module never modifies a posterior or runs inference.
"""

import numpy as np
from arviz_stats.base import array_stats


def screen_interval(value, mcse, low, high):
    if not np.isfinite(value) or not np.isfinite(mcse) or mcse < 0:
        return "unavailable"
    left, right = value - 2 * mcse, value + 2 * mcse
    if left >= low and right <= high:
        return "within_screen"
    if right < low or left > high:
        return "outside_screen"
    return "mc_precision_limited"


def primary_indices(names):
    return [
        i
        for i, name in enumerate(names)
        if name.startswith(
            (
                "log_S",
                "log_band",
                "log_sigma_",
                "sigma_",
                "csd_re_",
                "csd_im_",
                "coherence_",
                "temporal_contrast_",
            )
        )
        or name in ("log_sigma", "sigma")
    ]


def mc_stats(values, iid=False):
    values = np.asarray(values)
    flat = values.reshape(-1, values.shape[-1])
    mean, sd = flat.mean(0), flat.std(0, ddof=1)
    mean_se = (
        sd / np.sqrt(len(flat))
        if iid
        else np.asarray(
            array_stats.mcse(values, chain_axis=0, draw_axis=1, method="mean")
        )
    )
    if iid:
        fourth = np.mean((flat - mean) ** 4, axis=0)
        sd_se = np.sqrt(
            np.maximum(fourth - sd**4, 0) / (4 * len(flat) * sd**2)
        )
    else:
        sd_se = np.asarray(
            array_stats.mcse(values, chain_axis=0, draw_axis=1, method="sd")
        )
    result = {
        "mean": mean,
        "sd": sd,
        "mean_mcse": mean_se,
        "sd_mcse": sd_se,
        "sample_count": len(flat),
    }
    for level in (0.5, 0.9, 0.95):
        probs = [(1 - level) / 2, (1 + level) / 2]
        q = np.quantile(flat, probs, axis=0)
        se = [
            np.asarray(
                array_stats.mcse(
                    values,
                    chain_axis=0,
                    draw_axis=1,
                    method="quantile",
                    prob=p,
                )
            )
            for p in probs
        ]
        result[f"width_{int(100 * level)}"] = q[1] - q[0]
        result[f"width_mcse_upper_bound_{int(100 * level)}"] = se[0] + se[1]
    return result


def compare_with_mc(q, p, names, cfg):
    a, b = mc_stats(q, iid=True), mc_stats(p)
    rows = []
    for i, name in enumerate(names):
        delta = (a["mean"][i] - b["mean"][i]) / b["sd"][i]
        ratio = a["sd"][i] / b["sd"][i]
        mean_se = (
            np.sqrt(
                a["mean_mcse"][i] ** 2
                + b["mean_mcse"][i] ** 2
                + (delta * b["sd_mcse"][i]) ** 2
            )
            / b["sd"][i]
        )
        sd_se = ratio * np.sqrt(
            (a["sd_mcse"][i] / a["sd"][i]) ** 2
            + (b["sd_mcse"][i] / b["sd"][i]) ** 2
        )
        rows.append(
            {
                "name": name,
                "standardized_mean_difference": float(delta),
                "mean_difference_mcse": float(mean_se),
                "sd_ratio": float(ratio),
                "sd_ratio_mcse": float(sd_se),
                "mean_status": screen_interval(
                    delta,
                    mean_se,
                    -cfg["mean_tolerance_sd"],
                    cfg["mean_tolerance_sd"],
                ),
                "sd_status": screen_interval(
                    ratio, sd_se, *cfg["sd_ratio_limits"]
                ),
                "point_within_mean_screen": bool(
                    abs(delta) <= cfg["mean_tolerance_sd"]
                ),
                "point_within_sd_screen": bool(
                    cfg["sd_ratio_limits"][0]
                    <= ratio
                    <= cfg["sd_ratio_limits"][1]
                ),
                "vi_mean_mcse": float(a["mean_mcse"][i]),
                "reference_mean_mcse": float(b["mean_mcse"][i]),
                "vi_sd_mcse": float(a["sd_mcse"][i]),
                "reference_sd_mcse": float(b["sd_mcse"][i]),
                "interval_ratio_mcse_upper_bounds": {
                    str(level): float(
                        np.sqrt(
                            (
                                a[f"width_mcse_upper_bound_{level}"][i]
                                / b[f"width_{level}"][i]
                            )
                            ** 2
                            + (
                                a[f"width_{level}"][i]
                                * b[f"width_mcse_upper_bound_{level}"][i]
                                / b[f"width_{level}"][i] ** 2
                            )
                            ** 2
                        )
                    )
                    for level in (50, 90, 95)
                },
            }
        )
    return rows


def paired_stability(a, b, ref, names, first_objective, second_objective, cfg):
    """Paired draw influence estimates; reference SD retains its MC uncertainty."""
    a, b = (np.asarray(x).reshape(-1, len(names)) for x in (a, b))
    stats = mc_stats(ref)
    changes = b - a
    delta = changes.mean(0) / stats["sd"]
    mean_se = (
        np.sqrt(
            changes.var(0, ddof=1) / len(a) + (delta * stats["sd_mcse"]) ** 2
        )
        / stats["sd"]
    )
    sa, sb = a.std(0, ddof=1), b.std(0, ddof=1)
    drift = 2 * (sb - sa) / (sa + sb)
    ia = ((a - a.mean(0)) ** 2 - sa**2) / (2 * sa)
    ib = ((b - b.mean(0)) ** 2 - sb**2) / (2 * sb)
    influence = -4 * sb / (sa + sb) ** 2 * ia + 4 * sa / (sa + sb) ** 2 * ib
    sd_se = influence.std(0, ddof=1) / np.sqrt(len(a))
    obj = np.asarray(second_objective["values"]) - np.asarray(
        first_objective["values"]
    )
    obj_mean, obj_se = (
        float(obj.mean()),
        float(obj.std(ddof=1) / np.sqrt(len(obj))),
    )
    mean_tol, sd_tol = (
        cfg["stability_mean_tolerance_sd"],
        cfg["stability_sd_relative_tolerance"],
    )
    rows = [
        {
            "name": name,
            "mean_change_reference_sd": float(delta[i]),
            "paired_mean_mcse": float(mean_se[i]),
            "relative_sd_change": float(drift[i]),
            "paired_sd_change_mcse": float(sd_se[i]),
            "mean_status": screen_interval(
                delta[i], mean_se[i], -mean_tol, mean_tol
            ),
            "sd_status": screen_interval(drift[i], sd_se[i], -sd_tol, sd_tol),
        }
        for i, name in enumerate(names)
    ]
    relevant = [
        i
        for i, name in enumerate(names)
        if i in primary_indices(names) or name.startswith("coefficient_")
    ]
    point_pass = bool(
        np.max(abs(delta[relevant])) <= mean_tol
        and np.max(abs(drift[relevant])) <= sd_tol
        and abs(obj_mean) <= cfg["objective_change_tolerance"] + 2 * obj_se
    )
    resolved_outside = (
        any(
            rows[i][key] == "outside_screen"
            for i in relevant
            for key in ("mean_status", "sd_status")
        )
        or abs(obj_mean) > cfg["objective_change_tolerance"] + 2 * obj_se
    )
    return {
        "features": rows,
        "max_mean_change_reference_sd": float(np.max(abs(delta[relevant]))),
        "max_relative_sd_change": float(np.max(abs(drift[relevant]))),
        "paired_objective_change": obj_mean,
        "paired_objective_mcse": obj_se,
        "point_within_screen": point_pass,
        "status": "outside_screen"
        if resolved_outside
        else "within_screen"
        if all(
            rows[i][key] == "within_screen"
            for i in relevant
            for key in ("mean_status", "sd_status")
        )
        and point_pass
        else "mc_precision_limited",
        "coupling": "same draw key within one guide family; paired influence-function SD MCSE; no cross-family coupling claim",
    }
