import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------
# Generate synthetic signal
# ----------------------------------------------------------
fs = 500
T = 2.0
t = np.arange(0, T, 1 / fs)

freqs = [3, 7, 15]
amps = [1.0, 0.7, 0.4]
colors = ["#b026ff", "#d864ff", "#5a2ecc"]

components = [
    amps[i] * np.sin(2 * np.pi * freqs[i] * t) for i in range(len(freqs))
]
signal = sum(components)

# Frequency-domain PSD
freqs_fft = np.fft.rfftfreq(len(t), 1 / fs)
psd = np.abs(np.fft.rfft(signal)) ** 2

# ----------------------------------------------------------
# 3D figure setup
# ----------------------------------------------------------
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=20, azim=-55)


# ----------------------------------------------------------
# Helper to draw tilted panels ("cards")
# ----------------------------------------------------------
def draw_panel(ax, x0, y0, width, height, tilt_x, tilt_y, facecolor="white"):
    """Draw a tilted rectangular panel and return a transform for local coords."""
    origin = np.array([x0, y0, 0.0])
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(tilt_x), -np.sin(tilt_x)],
            [0, np.sin(tilt_x), np.cos(tilt_x)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(tilt_y), 0, np.sin(tilt_y)],
            [0, 1, 0],
            [-np.sin(tilt_y), 0, np.cos(tilt_y)],
        ]
    )
    rot = Ry @ Rx

    local_corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ]
    )
    world_corners = (rot @ local_corners.T).T + origin

    panel = Poly3DCollection(
        [world_corners],
        facecolor=facecolor,
        edgecolor="#d0d0d0",
        linewidth=0.8,
        alpha=1.0,
    )
    ax.add_collection3d(panel)

    return {"rot": rot, "origin": origin, "width": width, "height": height}


def project_to_panel(panel, u_vals, v_vals):
    """Map local panel coordinates (u, v) to 3D space."""
    u = np.asarray(u_vals)
    v = np.asarray(v_vals)
    local = np.vstack([u, v, np.zeros_like(u, dtype=float)])
    pts = panel["rot"] @ local
    pts += panel["origin"][:, None]
    return pts[0], pts[1], pts[2]


# ----------------------------------------------------------
# Draw left panel (time-domain)
# ----------------------------------------------------------
left_panel = draw_panel(
    ax,
    x0=0,
    y0=0,
    width=3.0,
    height=2.0,
    tilt_x=0.3,
    tilt_y=0.6,
    facecolor="#f6f6f6",
)

# Project the time-series onto the panel
u_time = np.linspace(
    0.2,
    left_panel["width"] - 0.2,
    len(signal),
)
amp_scale = left_panel["height"] * 0.35
v_time = left_panel["height"] / 2 + amp_scale * (
    signal / np.max(np.abs(signal))
)
ts_x, ts_y, ts_z = project_to_panel(left_panel, u_time, v_time)

ax.plot(ts_x, ts_y, ts_z, color="crimson", lw=2)

# ----------------------------------------------------------
# Draw right panel (frequency-domain)
# ----------------------------------------------------------
right_panel = draw_panel(
    ax,
    x0=4.0,
    y0=0,
    width=3.0,
    height=2.0,
    tilt_x=0.3,
    tilt_y=-0.6,
    facecolor="#f6f6f6",
)

# Project PSD onto right panel
freq_norm = freqs_fft / freqs_fft.max()
u_psd = 0.2 + (right_panel["width"] - 0.4) * freq_norm
psd_norm = psd / psd.max()
v_psd = 0.25 + (right_panel["height"] - 0.35) * psd_norm
psd_x, psd_y, psd_z = project_to_panel(right_panel, u_psd, v_psd)

ax.plot(psd_x, psd_y, psd_z, color="#2196f3", lw=2)
baseline_v = np.full_like(u_psd, 0.25)
base_x, base_y, base_z = project_to_panel(right_panel, u_psd, baseline_v)
ax.plot(base_x, base_y, base_z, color="#2196f3", lw=1, alpha=0.35)

# ----------------------------------------------------------
# Draw floating sinusoids between the panels
# ----------------------------------------------------------
mid_x = np.linspace(2.0, 4.0, len(signal))
for comp, col in zip(components, colors):
    mid_y = 1.2 + 0.1 * np.sin(2 * np.pi * t / T)
    mid_z = 0.3 * comp + 0.4
    ax.plot(mid_x, mid_y, mid_z, color=col, lw=2)

# ----------------------------------------------------------
# Style / cleanup
# ----------------------------------------------------------
ax.set_axis_off()
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 2)

plt.tight_layout()
OUT = "out_fft_demo"
os.makedirs(OUT, exist_ok=True)
# plt.savefig(os.path.join(OUT, "fft_demo_3d.png"), dpi=150)
plt.show()
