from pathlib import Path

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.io import imread, imsave

# ============================================================
# Settings
# ============================================================

HERE = Path(__file__).parent
OUTDIR = HERE / "out_denoise"
OUTDIR.mkdir(exist_ok=True)

# Path to input image (change this to your file if needed)
IMAGE_PATH = OUTDIR / "input.png"  # put your image here

# SNR levels in dB (descending for storytelling)
SNR_LEVELS = [40, 20, 10, 5, 2, 1, 0]

# When True, we reverse the noise creation order to make the panel look like
# progressively better "denoised" outputs.
DISPLAY_AS_DENOISE = True

# Include the pristine reference frame as the last panel.
APPEND_CLEAN_REFERENCE = True

# Make the lowest-SNR stage almost pure static noise (but still faint image).
STATIC_LOWEST_STAGE = True
STATIC_LOWEST_STAGE_IMAGE_WEIGHT = 0.05

# Animation settings (set to False to skip GIF creation).
CREATE_ANIMATION = True
ANIMATION_PATH = OUTDIR / "comparison_animation.gif"
FRAME_DURATION = 10  # seconds per frame

# Fraction of image height at which to take the scanline for PSD
SCANLINE_FRACTION = 0.4  # e.g. 0.4 = 40% down from top


# ============================================================
# Noise helpers
# ============================================================


def add_noise_to_reach_snr(image, snr_db, *, force_static=False):
    """
    Add per-channel Gaussian noise so that the *theoretical* SNR is snr_db.
    SNR (dB) = 10 * log10(P_signal / P_noise).

    For very low SNR (<= 0 dB), we slightly boost the noise so the image
    becomes visually almost unreadable.
    """
    image = img_as_float(image)
    if image.ndim == 2:  # grayscale safety, but we expect RGB
        image = np.stack([image] * 3, axis=-1)
    if force_static:
        rng = np.random.default_rng()
        uniform_noise = rng.uniform(0.0, 1.0, size=image.shape)
        weight = STATIC_LOWEST_STAGE_IMAGE_WEIGHT
        return np.clip(
            weight * image + (1.0 - weight) * uniform_noise, 0.0, 1.0
        )

    noisy = np.empty_like(image)
    snr_linear = 10 ** (snr_db / 10)

    for ch in range(image.shape[2]):
        channel = image[..., ch]
        P_signal = np.mean(channel**2)

        if P_signal == 0:
            noisy[..., ch] = channel
            continue

        P_noise = P_signal / snr_linear
        sigma = np.sqrt(P_noise)

        # Make 0 dB extra brutal so you can't see the image at all
        if snr_db <= 0:
            sigma *= 2.0

        noise = np.random.normal(0.0, sigma, size=channel.shape)
        noisy[..., ch] = channel + noise

    return np.clip(noisy, 0.0, 1.0)


# ============================================================
# IO helpers
# ============================================================


def save_png(path: Path, image: np.ndarray) -> None:
    """Ensure Pillow/imageio can handle the dtype when writing PNG files."""
    clipped = np.clip(image, 0.0, 1.0)
    as_uint8 = (clipped * 255).round().astype(np.uint8)
    imsave(path, as_uint8)


# ============================================================
# PSD helper
# ============================================================


def compute_scanline_psd(image, frac=0.5):
    """
    Convert RGB image to grayscale, take one horizontal scanline and
    compute its 1D PSD via FFT.

    Returns:
        freqs : array of frequency bins (unit: 1/pixels)
        psd   : power spectral density (arbitrary units)
    """
    gray = rgb2gray(image)
    h, w = gray.shape
    row_idx = int(h * frac)
    row = gray[row_idx, :]

    # Remove mean and apply window to reduce edge effects
    row = row - np.mean(row)
    window = np.hanning(len(row))
    row_win = row * window

    fft_vals = np.fft.rfft(row_win)
    psd = (np.abs(fft_vals) ** 2) / len(row_win)

    freqs = np.fft.rfftfreq(len(row_win), d=1.0)  # "cycles per pixel"
    return freqs, psd


# ============================================================
# Main: load, generate noisy/denoised, make figure
# ============================================================


def main():
    # ----- Load image -----
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {IMAGE_PATH}. "
            f"Put your image there or change IMAGE_PATH."
        )
    img = img_as_float(imread(IMAGE_PATH))

    # Ensure RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]

    # Save original for reference
    save_png(OUTDIR / "clean.png", img)

    # ----- Generate noisy variants -----
    lowest_snr = min(SNR_LEVELS)
    noisy_images = {}
    for snr in SNR_LEVELS:
        force_static = STATIC_LOWEST_STAGE and snr == lowest_snr
        noisy = add_noise_to_reach_snr(img, snr, force_static=force_static)

        noisy_images[snr] = (noisy, force_static)
        save_png(OUTDIR / f"noisy_{snr}dB.png", noisy)

    # ----- Compute PSDs -----
    # "True" PSD from the clean image
    freqs_true, psd_true = compute_scanline_psd(img, frac=SCANLINE_FRACTION)

    # Determine the order that will be displayed.
    if DISPLAY_AS_DENOISE:
        display_snrs = sorted(SNR_LEVELS)
    else:
        display_snrs = sorted(SNR_LEVELS, reverse=True)

    stage_data = []
    for snr in display_snrs:
        noisy_img, is_static = noisy_images[snr]
        if is_static:
            label = "Static noise"
        else:
            label = f"SNR {snr:g} dB"
        stage_data.append(
            {"image": noisy_img, "label": label, "snr": float(snr)}
        )

    if APPEND_CLEAN_REFERENCE:
        stage_data.append(
            {"image": img, "label": "Clean reference", "snr": np.inf}
        )

    stage_freqs_psds = [
        compute_scanline_psd(stage["image"], frac=SCANLINE_FRACTION)
        for stage in stage_data
    ]

    # ----- Save per-stage comparison figures ([image | PSD]) -----
    saved_figures = []
    for i, stage in enumerate(stage_data):
        freqs_stage, psd_stage = stage_freqs_psds[i]
        snr = stage["snr"]
        if np.isfinite(snr):
            snr_label = f"{snr:.0f} dB"
        else:
            snr_label = "Clean"

        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(10, 4),
            gridspec_kw={"width_ratios": [1.2, 1.0]},
            constrained_layout=True,
        )

        ax_img, ax_psd = axes
        ax_img.imshow(stage["image"])
        ax_img.set_title(f"Signal/noise ≈ {snr_label}")
        ax_img.axis("off")

        ax_psd.loglog(
            freqs_true[1:], psd_true[1:], color="red", label="SIGNAL"
        )
        ax_psd.loglog(
            freqs_stage[1:],
            psd_stage[1:],
            color="blue",
            alpha=0.8,
            label="NOISY DATA",
        )
        ax_psd.set_ylim(10**-7, 1)
        ax_psd.set_xlim(1e-3, freqs_stage.max())

        ax_psd.set_xlabel("Spatial frequency (cycles / pixel)", fontsize=12)
        ax_psd.set_ylabel("Power", fontsize=12)
        ax_psd.tick_params(labelsize=10)
        ax_psd.legend(loc="lower left", fontsize=10, frameon=False)

        stage_slug = (
            stage["label"]
            .lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace(".", "")
        )
        fig_path = OUTDIR / f"comparison_{i+1:02d}_{stage_slug}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        saved_figures.append(fig_path)

    if CREATE_ANIMATION and saved_figures:
        frames = [iio.imread(path) for path in saved_figures]
        # pause at the last frame repeatedly
        frames = frames + [frames[-1]] * 20
        # now go back in reverse order to create a looping effect
        frames = frames + list(reversed(frames[1:-1]))
        frames = frames + [frames[-1]] * 20

        durations = [FRAME_DURATION] * len(frames)
        iio.mimsave(
            ANIMATION_PATH,
            frames,
            duration=durations,
            loop=0,
        )

    print(f"Saved outputs to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
