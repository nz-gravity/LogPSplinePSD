"""
Render the LogPSplinePSD homepage animation with Manim.

Prerequisite:
    python make_animation_data.py

Preview:
    manim -pql make_animation.py LogPSplineHero

High quality:
    manim -pqh make_animation.py LogPSplineHero
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from matplotlib import colormaps
from matplotlib.colors import to_hex

from manim import *
from manimpango import list_fonts


# ============================================================
# Manim setup
# ============================================================

config.background_color = WHITE

CACHE_DIR = Path("animation_cache")
MANIFEST_FILE = CACHE_DIR / "manifest.json"


# ============================================================
# Font
# ============================================================

available_fonts = set(list_fonts())

for candidate in (
    "Avenir Next",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
):
    if candidate in available_fonts:
        FONT = candidate
        break
else:
    FONT = "sans-serif"

print(f"Using font: {FONT}")


# ============================================================
# Styling
# ============================================================

TEXT_COLOR = ManimColor("#222222")
SUBTLE_TEXT = ManimColor("#777777")
AXIS_COLOR = ManimColor("#AAAAAA")

POSTERIOR_COLOR = ManimColor("#65AD7B")
PERIODOGRAM_COLOR = ManimColor("#AFAFAF")

STATE_HOLD = 0.40
TRANSITION_TIME = 0.90

MAX_BASIS_POINTS = 220
MAX_PSD_POINTS = 420


# ============================================================
# Layout
# ============================================================

LEFT_X = -4.35
LEFT_WIDTH = 3.65

RIGHT_X = 2.45
RIGHT_WIDTH = 7.55

BASIS_Y = 1.45
BASIS_HEIGHT = 2.05

WEIGHTS_Y = -1.15
WEIGHTS_HEIGHT = 1.15

PSD_Y = 0.0
PSD_HEIGHT = 5.35


# ============================================================
# Load states
# ============================================================

def load_states():
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    states = []

    for info in manifest["states"]:
        data = np.load(
            CACHE_DIR / info["file"]
        )

        states.append(
            {
                key: np.asarray(data[key])
                for key in data.files
            }
        )

    return states


STATES = load_states()


# ============================================================
# Global ranges
# ============================================================

FMIN = min(
    float(np.min(s["psd_frequency"]))
    for s in STATES
)

FMAX = max(
    float(np.max(s["psd_frequency"]))
    for s in STATES
)

WEIGHT_MAX = (
    1.08
    * max(
        float(np.max(s["weight_magnitude"]))
        for s in STATES
    )
)

all_psd = np.concatenate(
    [
        *[s["psd_lo"] for s in STATES],
        *[s["psd_hi"] for s in STATES],
        STATES[0]["periodogram"],
    ]
)

positive_psd = all_psd[
    np.isfinite(all_psd)
    & (all_psd > 0)
]

LOG_Y_MIN = float(
    np.floor(
        np.log10(
            np.percentile(
                positive_psd,
                0.5,
            )
        )
    )
)

LOG_Y_MAX = float(
    np.ceil(
        np.log10(
            np.percentile(
                positive_psd,
                99.5,
            )
        )
    )
)


# ============================================================
# Colors
# ============================================================

def basis_colors(
    n_basis: int,
):
    cmap = colormaps["viridis"]

    samples = np.linspace(
        0.18,
        0.82,
        n_basis,
    )

    return [
        ManimColor(
            to_hex(cmap(x))
        )
        for x in samples
    ]


# ============================================================
# Generic curve
# ============================================================

def curve_from_xy(
    axes,
    x,
    y,
    *,
    color,
    stroke_width=2.0,
    opacity=1.0,
    max_points=MAX_PSD_POINTS,
):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) > max_points:
        idx = np.linspace(
            0,
            len(x) - 1,
            max_points,
            dtype=int,
        )
        x = x[idx]
        y = y[idx]

    points = [
        axes.c2p(
            float(xi),
            float(yi),
        )
        for xi, yi in zip(
            x,
            y,
            strict=True,
        )
    ]

    curve = VMobject()

    curve.set_points_as_corners(
        points
    )

    curve.set_stroke(
        color=color,
        width=stroke_width,
        opacity=opacity,
    )

    return curve


# ============================================================
# Basis curves
# ============================================================

def make_basis_group(
    axes,
    state,
):
    x = np.asarray(
        state["basis_frequency"]
    )

    basis = np.asarray(
        state["basis"]
    )

    n_basis = int(
        state["n_basis"]
    )

    if len(x) > MAX_BASIS_POINTS:
        idx = np.linspace(
            0,
            len(x) - 1,
            MAX_BASIS_POINTS,
            dtype=int,
        )

        x = x[idx]
        basis = basis[idx, :]

    colors = basis_colors(
        n_basis
    )

    linewidth = max(
        1.0,
        2.8 - 0.035 * n_basis,
    )

    curves = VGroup()

    for k in range(n_basis):
        curves.add(
            curve_from_xy(
                axes,
                x,
                basis[:, k],
                color=colors[k],
                stroke_width=linewidth,
                max_points=MAX_BASIS_POINTS,
            )
        )

    return curves


# ============================================================
# Basis matching
# ============================================================

def basis_centres_from_state(
    state,
):
    return np.asarray(
        state["basis_centres"],
        dtype=float,
    )


def nearest_basis_matching(
    old_state,
    new_state,
):
    """
    Match old basis functions to nearest new basis centers.

    Returns
    -------
    matches : list[(old_index, new_index)]
    new_only : list[new_index]
    old_only : list[old_index]
    """

    old_centres = basis_centres_from_state(
        old_state
    )

    new_centres = basis_centres_from_state(
        new_state
    )

    available_new = set(
        range(len(new_centres))
    )

    matches = []

    for old_index, old_centre in enumerate(
        old_centres
    ):
        if not available_new:
            break

        new_index = min(
            available_new,
            key=lambda j: abs(
                new_centres[j]
                - old_centre
            ),
        )

        matches.append(
            (
                old_index,
                new_index,
            )
        )

        available_new.remove(
            new_index
        )

    matched_old = {
        old_i
        for old_i, _
        in matches
    }

    matched_new = {
        new_i
        for _, new_i
        in matches
    }

    old_only = [
        i
        for i in range(
            len(old_centres)
        )
        if i not in matched_old
    ]

    new_only = [
        i
        for i in range(
            len(new_centres)
        )
        if i not in matched_new
    ]

    return (
        matches,
        new_only,
        old_only,
    )


def flattened_curve(
    curve,
    axes,
):
    """
    Collapse a curve onto the y=0 baseline.
    """

    flat = curve.copy()

    points = flat.get_points().copy()

    baseline_y = axes.c2p(
        FMIN,
        0,
    )[1]

    points[:, 1] = baseline_y

    flat.set_points(
        points
    )

    return flat


# ============================================================
# Weight bars
# ============================================================

def make_weight_group(
    axes,
    state,
):
    centres = np.asarray(
        state["basis_centres"]
    )

    weights = np.asarray(
        state["weight_magnitude"]
    )

    colors = basis_colors(
        len(weights)
    )

    scene_x = np.array(
        [
            axes.c2p(
                float(x),
                0,
            )[0]
            for x in centres
        ]
    )

    if len(scene_x) > 1:
        spacing = np.median(
            np.diff(scene_x)
        )

        bar_width = (
            0.65
            * abs(spacing)
        )
    else:
        bar_width = 0.20

    bars = VGroup()

    for centre, value, color in zip(
        centres,
        weights,
        colors,
        strict=True,
    ):
        bottom = axes.c2p(
            float(centre),
            0.0,
        )

        top = axes.c2p(
            float(centre),
            float(value),
        )

        height = max(
            float(
                top[1] - bottom[1]
            ),
            0.008,
        )

        bar = Rectangle(
            width=bar_width,
            height=height,
            fill_color=color,
            fill_opacity=0.92,
            stroke_width=0,
        )

        bar.move_to(
            [
                bottom[0],
                bottom[1]
                + height / 2,
                0,
            ]
        )

        bars.add(bar)

    return bars


# ============================================================
# Posterior band
# ============================================================

def make_posterior_band(
    axes,
    state,
):
    x = np.asarray(
        state["psd_frequency"]
    )

    lo = np.asarray(
        state["psd_lo"]
    )

    hi = np.asarray(
        state["psd_hi"]
    )

    if len(x) > MAX_PSD_POINTS:
        idx = np.linspace(
            0,
            len(x) - 1,
            MAX_PSD_POINTS,
            dtype=int,
        )

        x = x[idx]
        lo = lo[idx]
        hi = hi[idx]

    lo = np.log10(
        np.maximum(
            lo,
            1e-30,
        )
    )

    hi = np.log10(
        np.maximum(
            hi,
            1e-30,
        )
    )

    upper = [
        axes.c2p(
            float(xi),
            float(yi),
        )
        for xi, yi in zip(
            x,
            hi,
            strict=True,
        )
    ]

    lower = [
        axes.c2p(
            float(xi),
            float(yi),
        )
        for xi, yi in zip(
            x[::-1],
            lo[::-1],
            strict=True,
        )
    ]

    return Polygon(
        *(upper + lower),
        fill_color=POSTERIOR_COLOR,
        fill_opacity=0.32,
        stroke_width=0,
    )


# ============================================================
# Periodogram
# ============================================================

def make_periodogram(
    axes,
    state,
):
    x = np.asarray(
        state["periodogram_frequency"]
    )

    y = np.log10(
        np.maximum(
            state["periodogram"],
            1e-30,
        )
    )

    dots = VGroup()

    for xi, yi in zip(
        x,
        y,
        strict=True,
    ):
        if not (
            LOG_Y_MIN
            <= yi
            <= LOG_Y_MAX
        ):
            continue

        dots.add(
            Dot(
                axes.c2p(
                    float(xi),
                    float(yi),
                ),
                radius=0.010,
                color=PERIODOGRAM_COLOR,
                fill_opacity=0.20,
                stroke_width=0,
            )
        )

    return dots


# ============================================================
# Axis labels
# ============================================================

SUPERSCRIPT = str.maketrans(
    "-0123456789",
    "⁻⁰¹²³⁴⁵⁶⁷⁸⁹",
)


def exponent_label(
    n,
):
    return (
        "10"
        + str(n).translate(
            SUPERSCRIPT
        )
    )


def make_psd_tick_labels(
    axes,
):
    labels = VGroup()

    for tick in np.arange(
        5,
        FMAX,
        5,
    ):
        label = Text(
            str(int(tick)),
            font=FONT,
            font_size=17,
            color=SUBTLE_TEXT,
        )

        label.next_to(
            axes.c2p(
                tick,
                LOG_Y_MIN,
            ),
            DOWN,
            buff=0.08,
        )

        labels.add(label)

    for exp in range(
        int(LOG_Y_MIN),
        int(LOG_Y_MAX) + 1,
    ):
        label = Text(
            exponent_label(exp),
            font=FONT,
            font_size=17,
            color=SUBTLE_TEXT,
        )

        label.next_to(
            axes.c2p(
                FMIN,
                exp,
            ),
            LEFT,
            buff=0.10,
        )

        labels.add(label)

    return labels


# ============================================================
# Main scene
# ============================================================

class LogPSplineHero(
    Scene
):

    def construct(
        self,
    ):
        first = STATES[0]

        # ====================================================
        # Basis panel
        # ====================================================

        basis_axes = Axes(
            x_range=[
                FMIN,
                FMAX,
                5,
            ],
            y_range=[
                0,
                1.05,
                0.5,
            ],
            x_length=LEFT_WIDTH,
            y_length=BASIS_HEIGHT,
            tips=False,
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 0.8,
                "include_numbers": False,
            },
        )

        basis_axes.move_to(
            [
                LEFT_X,
                BASIS_Y,
                0,
            ]
        )

        basis_axes.y_axis.set_opacity(
            0
        )

        basis_axes.x_axis.set_opacity(
            0.25
        )

        basis_title = Text(
            "B-spline basis",
            font=FONT,
            font_size=27,
            color=TEXT_COLOR,
        )

        basis_title.next_to(
            basis_axes,
            UP,
            buff=0.16,
        )

        basis_title.align_to(
            basis_axes,
            LEFT,
        )

        current_k = Text(
            f"K = {int(first['n_basis'])}",
            font=FONT,
            font_size=20,
            color=SUBTLE_TEXT,
        )

        current_k.next_to(
            basis_axes,
            UP,
            buff=0.19,
        )

        current_k.align_to(
            basis_axes,
            RIGHT,
        )

        # ====================================================
        # Weight panel
        # ====================================================

        weight_axes = Axes(
            x_range=[
                FMIN,
                FMAX,
                5,
            ],
            y_range=[
                0,
                WEIGHT_MAX,
                max(
                    WEIGHT_MAX / 2,
                    1e-3,
                ),
            ],
            x_length=LEFT_WIDTH,
            y_length=WEIGHTS_HEIGHT,
            tips=False,
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 0.8,
                "include_numbers": False,
            },
        )

        weight_axes.move_to(
            [
                LEFT_X,
                WEIGHTS_Y,
                0,
            ]
        )

        weight_axes.y_axis.set_opacity(
            0
        )

        weight_axes.x_axis.set_opacity(
            0.25
        )

        weight_title = Text(
            "basis weights",
            font=FONT,
            font_size=18,
            color=SUBTLE_TEXT,
        )

        weight_title.next_to(
            weight_axes,
            DOWN,
            buff=0.12,
        )

        # ====================================================
        # PSD panel
        # ====================================================

        psd_axes = Axes(
            x_range=[
                FMIN,
                FMAX,
                5,
            ],
            y_range=[
                LOG_Y_MIN,
                LOG_Y_MAX,
                1,
            ],
            x_length=RIGHT_WIDTH,
            y_length=PSD_HEIGHT,
            tips=False,
            axis_config={
                "color": AXIS_COLOR,
                "stroke_width": 0.85,
                "include_numbers": False,
            },
        )

        psd_axes.move_to(
            [
                RIGHT_X,
                PSD_Y,
                0,
            ]
        )

        psd_title = Text(
            "PSD inference",
            font=FONT,
            font_size=30,
            color=TEXT_COLOR,
        )

        psd_title.next_to(
            psd_axes,
            UP,
            buff=0.15,
        )

        psd_title.align_to(
            psd_axes,
            LEFT,
        )

        x_label = Text(
            "Frequency [Hz]",
            font=FONT,
            font_size=19,
            color=SUBTLE_TEXT,
        )

        x_label.next_to(
            psd_axes,
            DOWN,
            buff=0.34,
        )

        y_label = Text(
            "Power spectral density",
            font=FONT,
            font_size=18,
            color=SUBTLE_TEXT,
        )

        y_label.rotate(
            PI / 2
        )

        y_label.next_to(
            psd_axes,
            LEFT,
            buff=0.57,
        )

        tick_labels = (
            make_psd_tick_labels(
                psd_axes
            )
        )

        # ====================================================
        # Static data
        # ====================================================

        periodogram = (
            make_periodogram(
                psd_axes,
                first,
            )
        )

        periodogram.set_z_index(
            1
        )

        # ====================================================
        # Initial dynamic objects
        # ====================================================

        current_basis = (
            make_basis_group(
                basis_axes,
                first,
            )
        )

        current_weights = (
            make_weight_group(
                weight_axes,
                first,
            )
        )

        current_band = (
            make_posterior_band(
                psd_axes,
                first,
            )
        )

        current_band.set_z_index(
            2
        )

        current_state = first

        # ====================================================
        # Add scene
        # ====================================================

        self.add(
            basis_axes,
            basis_title,
            current_k,
            current_basis,

            weight_axes,
            current_weights,
            weight_title,

            psd_axes,
            tick_labels,
            psd_title,
            x_label,
            y_label,

            periodogram,
            current_band,
        )

        self.wait(
            STATE_HOLD
        )

        # ====================================================
        # Transition helper
        # ====================================================

        def transition_to(
            new_state,
        ):
            nonlocal current_basis
            nonlocal current_weights
            nonlocal current_band
            nonlocal current_k
            nonlocal current_state

            new_basis = (
                make_basis_group(
                    basis_axes,
                    new_state,
                )
            )

            new_weights = (
                make_weight_group(
                    weight_axes,
                    new_state,
                )
            )

            new_band = (
                make_posterior_band(
                    psd_axes,
                    new_state,
                )
            )

            new_band.set_z_index(
                2
            )

            (
                matches,
                new_only,
                old_only,
            ) = nearest_basis_matching(
                current_state,
                new_state,
            )

            animations = []

            # ------------------------------------------------
            # Existing basis curves morph
            # ------------------------------------------------

            used_new_basis = set()

            for old_i, new_i in matches:
                animations.append(
                    Transform(
                        current_basis[old_i],
                        new_basis[new_i],
                    )
                )

                used_new_basis.add(
                    new_i
                )

            # ------------------------------------------------
            # New basis curves grow from baseline
            # ------------------------------------------------

            new_curve_objects = []

            for new_i in new_only:
                target = (
                    new_basis[new_i]
                )

                flat = flattened_curve(
                    target,
                    basis_axes,
                )

                self.add(flat)

                animations.append(
                    Transform(
                        flat,
                        target,
                    )
                )

                new_curve_objects.append(
                    (
                        new_i,
                        flat,
                    )
                )

            # ------------------------------------------------
            # Old curves disappearing when K decreases
            # ------------------------------------------------

            for old_i in old_only:
                animations.append(
                    FadeOut(
                        current_basis[old_i]
                    )
                )

            # ------------------------------------------------
            # Existing weight bars morph
            #
            # Use same centre matching as basis curves.
            # ------------------------------------------------

            for old_i, new_i in matches:
                if (
                    old_i
                    < len(current_weights)
                    and new_i
                    < len(new_weights)
                ):
                    animations.append(
                        Transform(
                            current_weights[old_i],
                            new_weights[new_i],
                        )
                    )

            # ------------------------------------------------
            # New bars grow upward
            # ------------------------------------------------

            new_bar_objects = []

            for new_i in new_only:
                if new_i >= len(new_weights):
                    continue

                bar = (
                    new_weights[new_i]
                    .copy()
                )

                self.add(bar)

                animations.append(
                    GrowFromEdge(
                        bar,
                        DOWN,
                    )
                )

                new_bar_objects.append(
                    (
                        new_i,
                        bar,
                    )
                )

            # ------------------------------------------------
            # Remove old bars on downward transitions
            # ------------------------------------------------

            for old_i in old_only:
                if old_i < len(current_weights):
                    animations.append(
                        FadeOut(
                            current_weights[
                                old_i
                            ]
                        )
                    )

            # ------------------------------------------------
            # Posterior morph
            # ------------------------------------------------

            animations.append(
                Transform(
                    current_band,
                    new_band,
                )
            )

            # ------------------------------------------------
            # Play scientific animations only
            # ------------------------------------------------

            self.play(
                *animations,
                run_time=TRANSITION_TIME,
                rate_func=smooth,
            )

            # =================================================
            # Rebuild canonical groups after transition.
            #
            # This prevents accumulated transformed object
            # indexing from becoming confusing on later steps.
            # =================================================

            self.remove(
                current_basis,
                current_weights,
            )

            for _, obj in (
                new_curve_objects
            ):
                self.remove(obj)

            for _, obj in (
                new_bar_objects
            ):
                self.remove(obj)

            current_basis = (
                make_basis_group(
                    basis_axes,
                    new_state,
                )
            )

            current_weights = (
                make_weight_group(
                    weight_axes,
                    new_state,
                )
            )

            self.add(
                current_basis,
                current_weights,
            )

            # ------------------------------------------------
            # Update K instantly.
            # No animation needed.
            # ------------------------------------------------

            new_k = Text(
                f"K = {int(new_state['n_basis'])}",
                font=FONT,
                font_size=20,
                color=SUBTLE_TEXT,
            )

            new_k.move_to(
                current_k.get_center()
            )

            current_k.become(
                new_k
            )

            current_state = (
                new_state
            )

            self.wait(
                STATE_HOLD
            )

        # ====================================================
        # Increase K
        # ====================================================

        for state in STATES[1:]:
            transition_to(
                state
            )

        # ====================================================
        # Decrease K for clean loop
        # ====================================================

        for state in STATES[-2::-1]:
            transition_to(
                state
            )

        self.wait(0.2)