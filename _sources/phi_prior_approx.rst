Prior on \phi: TeX vs implementation
======================================

This page explains a deliberate difference between the draft math in
``overleaf`` and the sampler implementation.

Background: hierarchical P-spline prior
---------------------------------------

The draft writes the spline-weight prior as (exact LaTeX from ``overleaf``)

.. math::

   \bold{w}_j | \phi_j \sim \mathcal{N}(\bold{0}, (\phi_j \bold{P}_j)^{-1})

with hyperpriors (again copied verbatim):

.. math::

   \phi_j | \delta_j \sim \text{Gamma}(\alpha_\phi, \delta_j \beta_\phi)
   \\
   \delta_j \sim \text{Gamma}(\alpha_\delta, \beta_\delta)

In the implementation (NumPyro), we interpret these Gamma distributions using
the *rate* parameterisation.

What the code actually samples
------------------------------

The sampler utility :func:`log_psplines.samplers.utils.sample_pspline_block`
keeps the same *intended* hierarchical structure but changes the sampling
parameterisation for :math:`\phi`:

- it samples :math:`\delta \sim \mathrm{Gamma}(\alpha_\delta, \beta_\delta)` as
  written;
- it does **not** sample :math:`\phi\mid\delta` from a Gamma directly;
- instead it samples :math:`\log \phi` from a Normal distribution chosen to
  *moment-match* the Gamma.

Concretely, for

.. math::

   \phi\mid\delta \sim \mathrm{Gamma}(\alpha_\phi, \mathrm{rate}=\beta_\phi\,\delta),

the mean and variance are

.. math::

   \mathbb{E}[\phi\mid\delta] = \frac{\alpha_\phi}{\beta_\phi\,\delta},
   \qquad
   \mathrm{Var}[\phi\mid\delta] = \frac{\alpha_\phi}{(\beta_\phi\,\delta)^2}.

The code constructs a log-normal approximation

.. math::

   \log \phi\mid\delta \sim \mathcal{N}(\mu, \sigma^2)

with

.. math::

   \sigma^2 = \log\Big(1 + \frac{1}{\alpha_\phi}\Big),
   \qquad
   \mu = \log\Big(\frac{\alpha_\phi}{\beta_\phi\,\delta}\Big) - \frac{\sigma^2}{2}.

This exactly matches the formulas in
:func:`log_psplines.samplers.utils.sample_pspline_block`.

Why this change exists
----------------------

Sampling :math:`\phi` directly under the Gamma prior frequently produces a
“funnel” geometry once :math:`\phi` interacts with the Gaussian spline weights.
That geometry can make NUTS initialization and adaptation brittle.

Sampling :math:`\log\phi` under a moment-matched Normal has two practical
benefits:

- it reduces extreme curvature in the posterior geometry,
- it tends to improve VI-based initialization and NUTS warmup stability.

Importantly, this is **not** the same distribution as the Gamma prior. The two
match the first two moments of :math:`\phi\mid\delta`, but tails differ.

Where to look in the code
-------------------------

- Implementation: :func:`log_psplines.samplers.utils.sample_pspline_block`.
   Source: `src/log_psplines/samplers/utils.py#sample_pspline_block <https://github.com/nz-gravity/LogPSplinePSD/blob/main/src/log_psplines/samplers/utils.py#L46-L103>`_
- Usage:

  - :mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`
  - :mod:`log_psplines.samplers.multivar.multivar_blocked_nuts`

If you want the docs to reflect the implementation, you can interpret the
written Gamma prior for :math:`\phi\mid\delta` as an *idealised target*, and the
code as using a numerically friendlier, moment-matched approximation.
