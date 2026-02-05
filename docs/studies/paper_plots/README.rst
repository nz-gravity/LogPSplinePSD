Paper plot jobs
===============

This directory contains small, reproducible “job scripts” used to generate the
plots for the paper (VAR(3) simulation + LISA XYZ).

Why here?
---------
These scripts are analysis/figure generation workflows (not library code), so
they live under ``docs/studies/`` alongside the other study runners.

What you run
------------

- ``var3_paper_job.py``: VAR(3) simulation study (short/long N, optional coarse
  graining).
- ``generate_lisa_paper_data.py``: generate paper-sized, low-cadence LISA XYZ
  synthetic datasets (avoids aliasing from naive downsampling).
- ``lisa_paper_job.py``: LISA XYZ study (short/long N, optional coarse graining,
  selectable frequency band).
- ``run_all.sh``: one-click runner that executes the 9 paper jobs with a
  consistent naming scheme and output structure.

Outputs
-------
Each job writes to its own output directory (passed via ``--outdir``), and the
core sampler writes:

- ``inference_data.nc``
- ``psd_matrix.png`` (and other diagnostics/plots)
- ``summary_statistics.csv`` (unless disabled)

All outputs are gitignored by default in this repo.

Notes
-----
- VAR(3) coarse graining “size 512” is interpreted as ``n_bins=512`` (i.e. 512
  coarse frequency bins).
- LISA coarse graining uses ``n_freqs_per_bin=5`` by default (coarse-grain every
  5 fine-grid frequencies) since ``n_freqs_per_bin`` must be odd.
- The “restricted” LISA band is ``[1e-4, 1e-2]`` Hz. The “full” band defaults to
  ``[1e-4, 1e-1]`` Hz (adjust in ``run_all.sh`` if you want a different upper
  bound).
- ``run_all.sh`` generates paper-sized synthetic datasets under
  ``docs/studies/paper_plots/data/`` using lisatools. Set cadence via
  ``LISA_DELTA_T`` (default 5 s, which supports Nyquist = 0.1 Hz), and set the
  Wishart block length via ``LISA_BLOCK_SIZE`` (default 5000 samples).
- Default LISA durations are 4 weeks (short) and 12 weeks (long); override via
  ``LISA_SHORT_WEEKS`` / ``LISA_LONG_WEEKS``.
