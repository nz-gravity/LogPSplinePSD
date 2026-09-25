"""Manim explainer for a Bayesian P-spline fit.

Visual story
------------
Top panel:
    observed data + posterior 95% credible band + posterior median.

Bottom panel:
    colour-matched weighted B-spline contributions are added one by one.

Each basis coefficient is switched on from 0 -> its posterior draws. Therefore
both the posterior median and its credible interval build up progressively.

Optional real-data input
------------------------
Place a file called ``pspline_posterior.npz`` beside this script with:

    x          : (N,) grid
    y          : (N_data,) observed data values
    x_data     : (N_data,) x locations of observations
                 optional, defaults to x
    basis      : (K, N) B-spline basis matrix
    w_draws    : (S, K) posterior draws of spline coefficients

If the file is absent, a synthetic demo posterior is generated.

Render:
    manim -pqh make_c2_pspline_posterior_explainer.py PSplinePosteriorExplainer
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Polygon,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    VMobject,
    WHITE,
    always_redraw,
    config,
    smooth,
)


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "pspline_posterior.npz"

# ============================================================
# Style
# ============================================================

COLORS = [
    "#007FAF",  # blue
    "#FF6B00",  # orange
    "#DFA900",  # mustard
    "#8F2DA6",  # purple
    "#4E9A06",  # green
    "#159FBE",  # cyan
    "#C8102E",  # red
]

INK = "#20252B"
GREY = "#7A7A7A"
LIGHT_GREY = "#C7C7C7"
BAND = "#AAB4BE"
DATA_GREY = "#555A60"

config.background_color = WHITE
config.frame_rate = 30


# ============================================================
# Basis utilities
# ============================================================

def cardinal_cubic(u: np.ndarray) -> np.ndarray:
    """Centred cardinal cubic B-spline, with B(0)=2/3."""
    a = np.abs(np.asarray(u, dtype=float))

    out = np.zeros_like(a)

    inner = a < 1.0
    shoulder = (a >= 1.0) & (a < 2.0)

    out[inner] = (
        4.0
        - 6.0 * a[inner] ** 2
        + 3.0 * a[inner] ** 3
    ) / 6.0

    out[shoulder] = (
        2.0 - a[shoulder]
    ) ** 3 / 6.0

    return out


def line_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    width: float,
) -> VMobject:
    """Create a Manim polyline from x/y arrays."""
    points = np.column_stack(
        [x, y, np.zeros_like(x)]
    )

    curve = VMobject(
        color=color,
        stroke_width=width,
    )

    curve.set_points_as_corners(points)

    return curve


def band_polygon(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    color: str = BAND,
    opacity: float = 0.34,
) -> Polygon:
    """Posterior credible-region polygon."""

    upper = np.column_stack(
        [x, hi, np.zeros_like(x)]
    )

    lower = np.column_stack(
        [x[::-1], lo[::-1], np.zeros_like(x)]
    )

    points = np.vstack([upper, lower])

    return Polygon(
        *points,
        stroke_width=0,
        fill_color=color,
        fill_opacity=opacity,
    )


# ============================================================
# Synthetic demo
# ============================================================

def make_demo_data():
    """Generate a reproducible synthetic example."""

    rng = np.random.default_rng(17)

    # --------------------------------------------------------
    # Grid + B-spline basis
    # --------------------------------------------------------
    x = np.linspace(0.0, 10.0, 601)

    n_basis = 7

    centres = np.linspace(
        2.0,
        8.0,
        n_basis,
    )

    spacing = centres[1] - centres[0]

    basis = np.array(
        [
            cardinal_cubic(
                (x - centre) / spacing
            )
            for centre in centres
        ]
    )

    # --------------------------------------------------------
    # Synthetic posterior on weights
    # --------------------------------------------------------
    w_mean = np.array([
        0.34,
        0.70,
        1.08,
        0.76,
        1.24,
        0.72,
        0.38,
    ])

    w_sd = np.array([
        0.08,
        0.09,
        0.10,
        0.09,
        0.11,
        0.09,
        0.07,
    ])

    # Mild correlation between neighbouring coefficients
    rho = 0.38

    indices = np.arange(n_basis)

    corr = rho ** np.abs(
        np.subtract.outer(indices, indices)
    )

    cov = (
        np.outer(w_sd, w_sd)
        * corr
    )

    w_draws = rng.multivariate_normal(
        mean=w_mean,
        cov=cov,
        size=1200,
    )

    # --------------------------------------------------------
    # Synthetic noisy data
    # --------------------------------------------------------
    truth_w = np.array([
        0.31,
        0.75,
        1.03,
        0.80,
        1.18,
        0.78,
        0.35,
    ])

    truth = truth_w @ basis

    x_data = np.linspace(
        0.55,
        9.45,
        58,
    )

    y_true = np.interp(
        x_data,
        x,
        truth,
    )

    y = (
        y_true
        + rng.normal(
            0.0,
            0.055,
            size=len(x_data),
        )
    )

    return (
        x,
        x_data,
        y,
        basis,
        w_draws,
    )


# ============================================================
# Load real posterior or demo
# ============================================================

def load_inputs():

    if not DATA_FILE.exists():
        print(
            f"{DATA_FILE.name} not found. "
            "Using synthetic demo posterior."
        )

        return make_demo_data()

    print(
        f"Loading posterior from {DATA_FILE}"
    )

    data = np.load(DATA_FILE)

    required = {
        "x",
        "y",
        "basis",
        "w_draws",
    }

    missing = required.difference(
        data.files
    )

    if missing:
        raise ValueError(
            f"{DATA_FILE.name} is missing keys: "
            f"{sorted(missing)}"
        )

    x = np.asarray(
        data["x"],
        dtype=float,
    )

    y = np.asarray(
        data["y"],
        dtype=float,
    )

    basis = np.asarray(
        data["basis"],
        dtype=float,
    )

    w_draws = np.asarray(
        data["w_draws"],
        dtype=float,
    )

    if "x_data" in data.files:
        x_data = np.asarray(
            data["x_data"],
            dtype=float,
        )
    else:
        x_data = x

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    if basis.ndim != 2:
        raise ValueError(
            "basis must have shape (K, N)"
        )

    if w_draws.ndim != 2:
        raise ValueError(
            "w_draws must have shape (S, K)"
        )

    if basis.shape[0] != w_draws.shape[1]:
        raise ValueError(
            "basis.shape[0] must equal "
            "w_draws.shape[1]"
        )

    if basis.shape[1] != len(x):
        raise ValueError(
            "basis.shape[1] must equal len(x)"
        )

    if len(x_data) != len(y):
        raise ValueError(
            "x_data and y must have the same length"
        )

    return (
        x,
        x_data,
        y,
        basis,
        w_draws,
    )


# ============================================================
# Manim scene
# ============================================================

class PSplinePosteriorExplainer(Scene):
    """Build a posterior spline fit one basis function at a time."""

    def construct(self):

        (
            x,
            x_data,
            y_data,
            basis,
            w_draws,
        ) = load_inputs()

        n_basis = basis.shape[0]

        if n_basis > len(COLORS):
            raise ValueError(
                "Current palette supports "
                f"{len(COLORS)} basis functions, "
                f"but received {n_basis}."
            )

        # ----------------------------------------------------
        # Posterior summaries of coefficients
        # ----------------------------------------------------
        w_median = np.median(
            w_draws,
            axis=0,
        )

        # ====================================================
        # Layout
        # ====================================================

        left = -6.25
        right = 6.25

        # Top fit panel
        top_bottom = -0.05
        top_top = 2.72

        # Lower contribution panel
        basis_bottom = -2.62
        basis_top = -1.02

        # ----------------------------------------------------
        # Full posterior curves
        # ----------------------------------------------------
        full_curves = (
            w_draws @ basis
        )

        y_lo_full = np.percentile(
            full_curves,
            2.5,
            axis=0,
        )

        y_hi_full = np.percentile(
            full_curves,
            97.5,
            axis=0,
        )

        # ----------------------------------------------------
        # Vertical range of top panel
        # ----------------------------------------------------
        y_min = min(
            np.min(y_data),
            np.min(y_lo_full),
            0.0,
        )

        y_max = max(
            np.max(y_data),
            np.max(y_hi_full),
        )

        pad = (
            0.10
            * max(
                1e-8,
                y_max - y_min,
            )
        )

        y_min -= pad
        y_max += pad

        # ====================================================
        # Coordinate transforms
        # ====================================================

        def sx(values):
            values = np.asarray(values)

            return (
                left
                + (
                    (values - x.min())
                    / (x.max() - x.min())
                )
                * (right - left)
            )

        def sy(values):
            values = np.asarray(values)

            return (
                top_bottom
                + (
                    (values - y_min)
                    / (y_max - y_min)
                )
                * (top_top - top_bottom)
            )

        # Separate scale for the bottom contributions
        contribution_scale = max(
            1e-8,
            np.max(
                np.abs(
                    w_median[:, None]
                    * basis
                )
            ),
        )

        def by(values):
            values = np.asarray(values)

            return (
                basis_bottom
                + (
                    values / contribution_scale
                )
                * (basis_top - basis_bottom)
            )

        x_scene = np.asarray(
            sx(x)
        )

        # ====================================================
        # Labels
        # ====================================================

        equation = Text(
            "s(f) = sum_m  B_m(f) w_m",
            color=INK,
            font="Arial",
            font_size=34,
        )

        equation.move_to([
            -3.45,
            3.28,
            0.0,
        ])

        title = Text(
            "Posterior P-spline fit",
            color=GREY,
            font="Arial",
            font_size=23,
        )

        title.move_to([
            4.65,
            3.28,
            0.0,
        ])

        data_label = Text(
            "data",
            color=DATA_GREY,
            font="Arial",
            font_size=18,
        )

        data_label.move_to([
            5.70,
            2.50,
            0.0,
        ])

        band_label = Text(
            "95% credible interval",
            color=GREY,
            font="Arial",
            font_size=17,
        )

        band_label.move_to([
            4.88,
            2.18,
            0.0,
        ])

        median_label = Text(
            "posterior median",
            color=INK,
            font="Arial",
            font_size=17,
        )

        median_label.move_to([
            5.02,
            1.91,
            0.0,
        ])

        basis_label = Text(
            "weighted B-spline contributions",
            color=GREY,
            font="Arial",
            font_size=19,
        )

        basis_label.move_to([
            -4.55,
            -0.69,
            0.0,
        ])

        # ====================================================
        # Data points
        # ====================================================

        data_points = VGroup(
            *[
                Dot(
                    [
                        float(sx(xi)),
                        float(sy(yi)),
                        0.0,
                    ],
                    radius=0.037,
                    color=DATA_GREY,
                    fill_opacity=0.70,
                    stroke_width=0,
                )
                for xi, yi
                in zip(
                    x_data,
                    y_data,
                )
            ]
        )

        # ====================================================
        # Trackers
        # ====================================================

        trackers = [
            ValueTracker(0.0)
            for _ in range(n_basis)
        ]

        def alpha():
            """Current activation of each basis coefficient."""
            return np.array(
                [
                    tracker.get_value()
                    for tracker in trackers
                ]
            )

        def active_draws():
            """
            Progressively turn on each posterior coefficient.

            Each coefficient's entire posterior distribution is
            multiplied by its activation parameter.
            """
            return (
                w_draws
                * alpha()[None, :]
            )

        def posterior_summaries():
            curves = (
                active_draws()
                @ basis
            )

            lo = np.percentile(
                curves,
                2.5,
                axis=0,
            )

            med = np.median(
                curves,
                axis=0,
            )

            hi = np.percentile(
                curves,
                97.5,
                axis=0,
            )

            return lo, med, hi

        # ====================================================
        # Posterior credible band
        # ====================================================

        posterior_band = always_redraw(
            lambda: band_polygon(
                x_scene,
                np.asarray(
                    sy(
                        posterior_summaries()[0]
                    )
                ),
                np.asarray(
                    sy(
                        posterior_summaries()[2]
                    )
                ),
                color=BAND,
                opacity=0.36,
            )
        )

        # ====================================================
        # Posterior median
        # ====================================================

        median_curve = always_redraw(
            lambda: line_from_xy(
                x_scene,
                np.asarray(
                    sy(
                        posterior_summaries()[1]
                    )
                ),
                color=INK,
                width=7.0,
            )
        )

        # ====================================================
        # Bottom panel
        # ====================================================

        zero_y = float(
            by(0.0)
        )

        basis_axis = Line(
            [
                left,
                zero_y,
                0.0,
            ],
            [
                right,
                zero_y,
                0.0,
            ],
            color=LIGHT_GREY,
            stroke_width=2.0,
        )

        # ----------------------------------------------------
        # Weighted basis contributions
        # ----------------------------------------------------

        contributions = VGroup()

        for j in range(n_basis):

            contributions.add(
                always_redraw(
                    lambda jj=j:
                    line_from_xy(
                        x_scene,
                        np.asarray(
                            by(
                                trackers[jj]
                                .get_value()
                                * w_median[jj]
                                * basis[jj]
                            )
                        ),
                        color=COLORS[jj],
                        width=6.0,
                    )
                )
            )

        # ====================================================
        # Weight labels
        # ====================================================

        centres = (
            np.sum(
                basis * x[None, :],
                axis=1,
            )
            / np.maximum(
                np.sum(
                    basis,
                    axis=1,
                ),
                1e-12,
            )
        )

        weight_labels = VGroup(
            *[
                Text(
                    f"w{j + 1}",
                    color=COLORS[j],
                    font="Arial",
                    font_size=19,
                ).move_to(
                    [
                        float(
                            sx(
                                centres[j]
                            )
                        ),
                        -3.02,
                        0.0,
                    ]
                )
                for j in range(n_basis)
            ]
        )

        # ====================================================
        # Status message
        # ====================================================

        status = Text(
            "Start with all coefficients at zero",
            color=GREY,
            font="Arial",
            font_size=20,
        )

        status.move_to([
            0.0,
            -3.55,
            0.0,
        ])

        # ====================================================
        # Initial frame
        # ====================================================

        self.add(
            equation,
            title,

            data_points,
            data_label,

            posterior_band,
            median_curve,

            band_label,
            median_label,

            basis_label,
            basis_axis,
            contributions,

            weight_labels,
            status,
        )

        self.wait(0.8)

        # ====================================================
        # Add each basis contribution
        # ====================================================

        for j, tracker in enumerate(trackers):

            new_status = Text(
                f"Add basis function {j + 1}",
                color=COLORS[j],
                font="Arial",
                font_size=20,
            )

            new_status.move_to(
                status.get_center()
            )

            self.play(
                FadeOut(
                    status,
                    run_time=0.12,
                ),
                FadeIn(
                    new_status,
                    run_time=0.12,
                ),
            )

            status = new_status

            self.play(
                tracker.animate.set_value(
                    1.0
                ),
                run_time=0.72,
                rate_func=smooth,
            )

            self.wait(0.10)

        # ====================================================
        # Final message
        # ====================================================

        final_status = Text(
            "Posterior fit = sum of uncertain basis contributions",
            color=INK,
            font="Arial",
            font_size=20,
        )

        final_status.move_to(
            status.get_center()
        )

        self.play(
            FadeOut(
                status,
                run_time=0.18,
            ),
            FadeIn(
                final_status,
                run_time=0.18,
            ),
        )

        self.wait(1.5)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        "Render with:\n\n"
        "manim -pqh "
        "make_c2_pspline_posterior_explainer.py "
        "PSplinePosteriorExplainer"
    )