import pandas as pd
import jax.numpy as jnp
import numpy as np
import jax
from log_psplines.datasets import Periodogram, Timeseries

URL = "https://raw.githubusercontent.com/bilby-dev/bilby/main/bilby/gw/detector/noise_curves/aLIGO_O4_high_asd.txt"


def load_lvk_psd() -> Periodogram:
    df = pd.read_csv(URL, comment='#', sep='\s+', header=None)
    freq, asd = df[0].values, df[1].values
    # rescale ASD so numbers are not too small
    asd = asd / asd.min()
    return Periodogram(freq, asd)


def create_white_noise(sampling_frequency, duration):
    number_of_samples = duration * sampling_frequency
    number_of_samples = int(np.round(number_of_samples))

    number_of_samples = int(np.round(duration * sampling_frequency))
    number_of_frequencies = int(np.round(number_of_samples / 2) + 1)

    frequencies = np.linspace(start=0,
                              stop=sampling_frequency / 2,
                              num=number_of_frequencies)

    norm1 = 0.5 * duration ** 0.5
    re1, im1 = np.random.normal(0, norm1, (2, len(frequencies)))
    white_noise = re1 + 1j * im1

    # set DC and Nyquist = 0
    white_noise[0] = 0
    # no Nyquist frequency when N=odd
    if np.mod(number_of_samples, 2) == 0:
        white_noise[-1] = 0

    # python: transpose for use with infft
    white_noise = np.transpose(white_noise)
    frequencies = np.transpose(frequencies)

    return white_noise, frequencies


def get_lvk_noise_realisation(sampling_frequency=4096.0, duration=4.0):
    psd = load_lvk_psd()
    white_noise, frequencies = create_white_noise(sampling_frequency, duration)
    with np.errstate(invalid="ignore"):
        # setup iterp1d over PSD
        colored_noise = np.interp(frequencies, psd.freqs, psd.power) ** 0.5 * white_noise

    return Periodogram(frequencies, np.abs(colored_noise) ** 2)
