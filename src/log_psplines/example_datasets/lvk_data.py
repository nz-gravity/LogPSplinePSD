import numpy as np
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import matplotlib.pyplot as plt


class LVKData:
    def __init__(self, strain: np.ndarray, psd: np.ndarray, freqs: np.ndarray):
        self.strain = strain
        self.psd = psd
        self.freqs = freqs

    @classmethod
    def download_data(cls, detector: str = "H1", gps_start: int = 1126259462,
                      duration: int = 1024, fmin: float = 20, fmax: float = 512

                      ) -> "LVKData":
        gps_end = gps_start + duration
        print(f"Downloading {detector} data [{gps_start} - {gps_end}]")
        strain = TimeSeries.fetch_open_data(detector, gps_start, gps_end)
        strain = (strain - strain.mean()) / strain.std()
        psd = strain.psd()
        psd = psd.crop(fmin, fmax)
        return cls(strain=strain.value,
                   psd=psd.value,
                   freqs=psd.frequencies.value)

    def plot_psd(self, fname: str = None) -> None:
        freq = FrequencySeries(self.psd, frequencies=self.freqs)
        fig, ax = plt.subplots()
        ax.loglog(freq, color='black')
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power/Frequency [1/Hz]")
        ax.set_title("Power Spectral Density")
        if fname:
            plt.savefig(fname, bbox_inches='tight')
        return fig, ax