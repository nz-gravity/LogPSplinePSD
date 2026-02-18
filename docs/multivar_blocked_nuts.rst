Multivariate Blocked NUTS: Math ↔ Code
========================================

This page documents the *implemented* multivariate likelihood and
parameterisation used by the blocked sampler
:class:`log_psplines.samplers.multivar.multivar_blocked_nuts.MultivarBlockedNUTSSampler`.

It is intended as a durable bridge between the draft derivation in our paper
and the code under ``src/log_psplines/``.

Overview
--------

The multivariate pipeline is:

1. Build frequency-domain sufficient statistics from a multichannel time series
   using block-averaged FFTs.
   Frequency truncation is controlled by ``model.fmin``/``model.fmax``.
2. (Optional) coarse-grain the frequency axis by summing Wishart statistics
   across bins (coarse config controls ``Nc``/``Nh`` only).
3. Fit a Cholesky-parameterised spectral density matrix with P-splines,
   sampling each Cholesky row as an *independent* NumPyro/NUTS problem.

The implementation is designed around the Whittle/Wishart sufficient statistic
at each retained Fourier frequency.

Code pointers (clickable)
-------------------------

The links below point to the current `main` branch on GitHub (line numbers may
drift over time):

- `Wishart FFT construction (MultivarFFT.compute_wishart) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/datatypes/multivar.py#L165-L298>`_
- `Blocked NumPyro likelihood (_blocked_channel_model) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/multivar/multivar_blocked_nuts.py#L104-L260>`_
- `Coarse graining (apply_coarse_grain_multivar_fft) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/coarse_grain/multivar.py#L16-L138>`_
- `PSD reconstruction (reconstruct_psd_matrix) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/psplines/multivar_psplines.py#L387-L438>`_
- `Wishart→PSD conversion (Y_to_S) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/spectrum_utils.py#L157-L183>`_
- `U→Y conversion (U_to_Y) <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/spectrum_utils.py#L137-L145>`_

.. math::

   \Y(f_k)=\sum_{i=1}^{N_b} \underbrace{\d^{(i)}(f_k)\d^{(i)}(f_k)^*}_{\I^{(i)}(f_k)}

with degrees of freedom :math:`N_b` equal to the number of non-overlapping
blocks (called ``Nb`` in code).

Data → Wishart statistics
-------------------------

The sufficient statistics are computed by
:func:`log_psplines.datatypes.multivar.MultivarFFT.compute_wishart`.

Given time-domain data ``x`` with shape ``(n, p)``, the code:

- splits ``x`` into ``Nb`` contiguous non-overlapping blocks,
- detrends each block (constant detrend),
- applies a taper/window (default: Hann),
- uses ``np.fft.rfft`` to work on the positive-frequency grid,
- applies a one-sided PSD normalisation factor (Nyquist bin handled specially),
- forms the per-frequency matrix

  .. math::

     Y(f_k) = \sum_{b=1}^{N_b} D_b(f_k)\, D_b(f_k)^H,

  where :math:`D_b(f_k)` is the tapered/normalised FFT vector for block :math:`b`.

The code then eigendecomposes :math:`Y(f_k)` and stores

.. math::

   Y(f_k) = U(f_k) U(f_k)^H,

where the columns of :math:`U(f_k)` are :math:`u_\ell^{(k)} = \sqrt{\lambda_\ell^{(k)}}\, v_\ell^{(k)}`.

Those eigenvector-weighted components :math:`U(f_k)` are the arrays
``u_re``/``u_im`` exposed on :class:`log_psplines.datatypes.multivar.MultivarFFT`.

Cholesky parameterisation
-------------------------

The blocked sampler follows the Rosen/Orihara-style inverse parameterisation:

.. math::

   S(f_k)^{-1} = T(f_k)^H\, D(f_k)^{-1}\, T(f_k),

where

- :math:`D(f_k)` is diagonal with entries :math:`\delta_j(f_k)^2`,
- :math:`T(f_k)` is unit lower-triangular with strictly-lower entries
  :math:`-\theta_{jl}(f_k)`.

In code, the diagonal field is represented as

.. math::

   \log \delta_j(f_k)^2 = \sum_m B_m(f_k)\, w^{(\delta)}_{j,m}

and each off-diagonal coefficient has its own real/imag spline field

.. math::

   \Re\,\theta_{jl}(f_k) = \sum_m B_m(f_k)\, w^{(\Re)}_{jl,m},
   \qquad
   \Im\,\theta_{jl}(f_k) = \sum_m B_m(f_k)\, w^{(\Im)}_{jl,m}.

The blocked sampler actually samples distinct weight vectors per
:math:`\theta_{jl}` (within a fixed row :math:`j`).

Per-row factorisation of the likelihood
---------------------------------------

With the parameterisation above, the complex-Wishart/Whittle log-likelihood can
be written as a sum of row-wise terms (one term per Cholesky row).

Let :math:`u_j(f_k)` denote the :math:`j`-th row of :math:`U(f_k)`, i.e. the
vector of replicate components for channel :math:`j` at frequency :math:`f_k`.

Define the row residual

.. math::

   r_j(f_k) = u_j(f_k) - \sum_{l<j} \theta_{jl}(f_k)\, u_l(f_k).

Exact per-row likelihood (from ``overleaf``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The derivation expresses each factor as (verbatim LaTeX):

.. math::

   \mathcal{L}_j(\u_j,\u_{<j}|\btheta_j,\bdelta_j) \propto \nonumber \\
   \prod_{k=1}^{N} \delta_{jk}^{-2N_b} \exp \left(\frac{-\sum_{N_b=1}^p\left|u_{jN_b}^{(k)}-\sum_{l=1}^{j-1}\theta_{jl}^{(k)}u_{lN_b}^{(k)} \right|^2}{T\delta_{jk}^2} \right)

The blocked model implements (up to constants)

.. math::

   \log \mathcal{L}_j
   \;\propto\;
   -N_b\,N_h\,\sum_k \log\big(\delta_j(f_k)^2\big)
   \; - \sum_k \frac{\|r_j(f_k)\|_2^2}{T\,\delta_j(f_k)^2}.

Key points:

- :math:`N_b` is the number of averaged blocks (``fft_data.Nb``).
- :math:`T` is the per-block observation duration (``fft_data.duration``).
- :math:`N_h` is the constant coarse-bin size (``fft_data.Nh``). It scales the
  log-determinant term when coarse graining is enabled.
- The quadratic term uses the *summed sufficient statistics* directly, so it
  does not introduce an additional :math:`N_h` factor.

This is implemented in the internal NumPyro model
``_blocked_channel_model`` inside
:mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`.

Coarse-graining compatibility
-----------------------------

Coarse graining for multivariate FFT/Wishart statistics is performed by
:func:`log_psplines.preprocessing.coarse_grain.multivar.apply_coarse_grain_multivar_fft`.

Within each coarse bin :math:`J_h`, it sums

.. math::

   \bar Y_h = \sum_{f\in J_h} Y(f)

and recomputes :math:`\bar U_h` so that :math:`\bar Y_h = \bar U_h\bar U_h^H`.

The returned scalar ``Nh`` equals the bin member count for equal bins and is
stored as ``fft_data.Nh``. With this choice, each bin behaves like a Wishart
statistic with effective degrees of freedom :math:`N_b Nh`.

Exact coarse-grained likelihood form (from ``overleaf``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The coarse-grained approximation is written as (verbatim LaTeX):

.. math::

   \prod_{h=1}^{Nc} \left|\S(\bar{f}_h)\right|^{-N_b*Nh} \exp\left(- \tr\left[ \S(\bar{f}_h)^{-1} \bar{\Y}_h\right]\right)

Symbol ↔ code mapping
---------------------

The table below lists the most important objects and where they appear.

- :math:`p` (number of channels) → ``fft_data.p`` / ``self.p``
- :math:`f_k` (frequency grid) → ``fft_data.freq`` / ``self.freq``
- :math:`N_b` (block count / Wishart DOF) → ``fft_data.Nb`` / ``self.Nb``
- :math:`U(f_k)` (eigen replicates) → ``fft_data.u_re`` + i ``fft_data.u_im``
- :math:`\log \delta_j(f_k)^2` → deterministic nodes ``log_delta_sq_{j}``
- :math:`\theta_{jl}(f_k)` → deterministic nodes ``theta_re_{j}``, ``theta_im_{j}``
- coarse-bin member counts :math:`Nh` → ``fft_data.Nh`` / ``self.Nh``

PSD reconstruction
------------------

Posterior samples of ``log_delta_sq`` and ``theta_re``/``theta_im`` are
converted into a spectral density matrix by
:meth:`log_psplines.psplines.multivar_psplines.MultivariateLogPSplines.reconstruct_psd_matrix`.

For each frequency, it builds :math:`T(f_k)` with strictly-lower entries
:math:`-\theta_{jl}(f_k)`, constructs :math:`D(f_k)=\mathrm{diag}(\delta_j(f_k)^2)`,
and returns

.. math::

   S(f_k) = T(f_k)^{-1}\, D(f_k)\, T(f_k)^{-H}.

Notes on scaling conventions
----------------------------

The derivation in ``overleaf`` writes the Whittle likelihood with an explicit
observation-time factor :math:`T`. The exact LaTeX from ``overleaf`` is:

.. math::

   \mathcal{L}(\d|\S) \propto  \prod_{k=1}^{N} \det(\S(f_k))^{-1} \times
   \exp\left(-\frac{1}{T}\d(f_k)^* \S(f_k)^{-1} \d(f_k)\right),

The implementation keeps the :math:`1/T` factor explicit in the NumPyro
likelihood. Concretely:

- :func:`~log_psplines.datatypes.multivar.MultivarFFT.compute_wishart` records
  the per-block observation duration ``fft_data.duration`` (seconds), and scales
  the stored frequency-domain components so the quadratic term can be written as
  :math:`(1/T)\,d(f_k)^* S(f_k)^{-1} d(f_k)`.
- PSD conversions divide by ``duration`` via
  :func:`~log_psplines.spectrum_utils.Y_to_S(duration=...)`, so
  the resulting PSD matrices remain in the same one-sided “per Hz” convention
  used elsewhere in the codebase.

This is algebraically equivalent to absorbing :math:`T` into a deterministic
rescaling of the Fourier coefficients; we keep it explicit to match the paper
notation and to make the dependence on observation time unambiguous.

Exact Cholesky factorisation (from ``overleaf``)
-----------------------------------------------------

The derivation in ``overleaf`` uses the following exact Cholesky
parameterisation:

.. math::

   \mathbf{S}(f_k)^{-1} = \mathbf{T}_k^* \, \mathbf{D}_k^{-1} \, \mathbf{T}_k,

with

.. math::

   \bold{T}_k = \begin{pmatrix}
   1 & 0 & 0 & \cdots & 0 \\
   -\theta_{21}^{(k)} & 1 & 0 & \cdots & 0 \\
   -\theta_{31}^{(k)} & -\theta_{32}^{(k)} & 1 & \ddots & \vdots \\
   \vdots & \vdots & \ddots & \ddots & 0 \\
   -\theta_{p1}^{(k)} & -\theta_{p2}^{(k)}& \cdots & -\theta_{p,p-1}^{(k)} & 1
   \end{pmatrix}.

The blocked sampler’s internal likelihood corresponds to the per-row likelihood
factors :math:`\mathcal{L}_j` (see Eq. (likelihoodj) in ``overleaf``),
modulo the implementation’s FFT/PSD normalisation conventions.
