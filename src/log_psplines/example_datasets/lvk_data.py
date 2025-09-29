import matplotlib.pyplot as plt
import numpy as np
from gwosc import datasets as gwosc_datasets
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


class LVKData:
    def __init__(self, strain: np.ndarray, psd: np.ndarray, freqs: np.ndarray):
        self.strain = strain
        self.psd = psd
        self.freqs = freqs

    @classmethod
    def from_simulation(cls, duration: int = 4, fs: int = 1024) -> "LVKData":
        from io import StringIO

        import numpy as np
        import pycbc.noise
        import requests
        from pycbc.types import FrequencySeries as PycbcFrequencySeries

        # --- Download the ASD file from LIGO DCC ---
        url = "https://dcc.ligo.org/public/0165/T2000012/002/aligo_O3actual_H1.txt"
        response = requests.get(url)
        response.raise_for_status()  # ensure it downloaded properly

        # --- Parse into numpy arrays ---
        data = np.loadtxt(StringIO(response.text))
        freq, asd = data[:, 0], data[:, 1]

        # --- Convert ASD → PSD ---
        psd_vals = asd**2

        # --- Wrap into PyCBC FrequencySeries ---
        delta_f = freq[1] - freq[0]
        psd = PycbcFrequencySeries(psd_vals, delta_f=delta_f, copy=True)

        # ---- simulate noise timeseries -----
        delta_t = 1.0 / fs
        tsamples = int(duration / delta_t)
        ts = pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=127)
        strain = ts.numpy()
        return cls(strain=strain, psd=psd_vals, freqs=freq)

    @classmethod
    def download_data(
        cls,
        detector: str = "H1",
        gps_start: int = 1126259462,
        duration: int = 1024,
        fmin: float = 20,
        fmax: float = 512,
    ) -> "LVKData":
        gps_end = gps_start + duration
        print(f"Downloading {detector} data [{gps_start} - {gps_end}]")
        strain = TimeSeries.fetch_open_data(detector, gps_start, gps_end)
        strain = (strain - strain.mean()) / strain.std()
        psd = strain.psd()
        psd = psd.crop(fmin, fmax)
        return cls(
            strain=strain.value, psd=psd.value, freqs=psd.frequencies.value
        )

    @classmethod
    def from_event(
        cls,
        event_name: str,
        detector: str = "H1",
        event_duration: int = 4,
        psd_duration: int = 4,
        fmin: float = 20,
        fmax: float = 2048,
    ) -> "LVKData":
        try:
            event_gps = gwosc_datasets.event_gps(event_name)
        except ValueError:
            avail_events = gwosc_datasets.find_datasets()
            raise ValueError(
                f"Event {event_name} not found in GWOSC datasets. Avail datasets: {avail_events}"
            )

        gps_start = event_gps - event_duration - psd_duration
        return cls.download_data(
            detector=detector,
            gps_start=gps_start,
            duration=psd_duration,
            fmin=fmin,
            fmax=fmax,
        )

    def plot_psd(self, fname: str = None) -> None:
        freq = FrequencySeries(self.psd, frequencies=self.freqs)
        fig, ax = plt.subplots()
        ax.loglog(freq, color="black")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power/Frequency [1/Hz]")
        ax.set_title("Power Spectral Density")
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        return fig, ax
