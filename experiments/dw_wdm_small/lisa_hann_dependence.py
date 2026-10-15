"""Gaussian covariance probes for the selected physical Hann ordinates.

The local physical PSD is frozen within the window, as in the mean preflight.
This is a probe of the diagonal approximation, not full TDI calibration.
"""

from __future__ import annotations

import numpy as np
from experiments.dw_wdm_small import lisa_hann_one_case as case


def transform(delta: np.ndarray) -> np.ndarray:
    z = case.DT * delta

    def d(value):
        return (
            case.LENGTH
            * np.exp(-1j * np.pi * (case.LENGTH - 1) * value)
            * np.sinc(case.LENGTH * value)
            / np.sinc(value)
        )

    return 0.5 * d(z) - 0.25 * (
        d(z + 1 / (case.LENGTH - 1)) + d(z - 1 / (case.LENGTH - 1))
    )


def main() -> None:
    response, t0, orbit, galaxy = case.inputs()
    bins = np.array([6, 9, 12, 30, 33, 36, 90, 93, 96, 3066, 3069, 3072])
    frequency = bins / (case.LENGTH * case.DT)
    nfft = 8 * case.LENGTH
    df = 1 / (nfft * case.DT)
    fq = (np.arange(nfft // 2) + 0.5) * df
    epoch = t0 + 14 * 86400
    noise = response.analytic_channel_noise_psd(
        "A", orbit, np.array([epoch]), fq, frequency_chunk=384
    )[0]
    gal = galaxy(np.column_stack((np.full(len(fq), epoch), np.log(fq))))
    energy = 3 * (case.LENGTH - 1) / 8
    phase = np.exp(2j * np.pi * fq[None, :] * (bins[:, None] - 1) * case.DT)
    plus = (
        phase * transform(frequency[:, None] - fq[None, :]) / np.sqrt(energy)
    )
    minus = (
        phase.conj()
        * transform(frequency[:, None] + fq[None, :])
        / np.sqrt(energy)
    )
    results = {}
    for name, spectrum in (
        ("white", np.full(len(fq), 2 * case.DT)),
        ("physical_total", noise + gal),
    ):
        weights = spectrum * df / 2
        c = (plus * weights) @ plus.conj().T + (
            minus * weights
        ) @ minus.conj().T
        p = (plus * weights) @ minus.T + (minus * weights) @ plus.T
        cov = abs(c) ** 2 + abs(p) ** 2
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        np.fill_diagonal(corr, 0)
        if name == "white":
            np.testing.assert_allclose(np.diag(c).real, 1, atol=1e-10)
        results[name] = {
            "max_power_correlation": float(corr.max()),
            "max_nonproperness": float(
                np.max(abs(np.diag(p)) / np.diag(c).real)
            ),
        }
    case.write_json(
        case.OUT / "covariance_probes.json",
        {
            "scope": "12 local ordinates, stationary Gaussian approximation at day 14; no full-transform covariance or repeated coverage claim",
            "rungs": bins.tolist(),
            "results": results,
        },
    )
    print(results)


if __name__ == "__main__":
    main()
