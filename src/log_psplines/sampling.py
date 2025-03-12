import dataclasses

import jax
import jax.numpy as jnp
from jax import jit, random

from .distributions import gamma_logpdf, sample_gamma


@dataclasses.dataclass
class SamplingArgs:
    phi: jnp.ndarray  # shape (3,) x, alpha, beta
    delta: jnp.ndarray  # shape (3,) x, alpha, beta
    weights: jnp.ndarray  # shape (n_basis,)
    log_pdgrm: "Periodogram"  # LOGGED Periodogram instance
    spline_model: "LogPSplines"  # LogPSplines instance
    key: random.PRNGKey  # random key

    @property
    def P(self) -> jnp.ndarray:
        return self.spline_model.penalty_matrix

    @property
    def k(self) -> int:
        return self.spline_model.n_basis


MIN = jnp.finfo(jnp.float32).min


@jax.jit
def lnlikelihood(lndata: jnp.ndarray, lnspline: jnp.ndarray) -> float:
    """Compute the Whittle log likelihood.

    Args:
        lndata_log: Log power spectral density data.
        log_psplines: Instance of LogPSplines.
        weights: Spline weights.

    Returns:
        The computed log likelihood.
    """
    integrand = lnspline + jnp.exp(lndata - lnspline - jnp.log(2 * jnp.pi))
    lnlike = -jnp.sum(integrand) / 2

    # If lnlike is not finite, return a very large negative value.
    lnlike = jnp.where(jnp.isfinite(lnlike), lnlike, MIN)
    return lnlike


@jit
def _xPx(w: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(w, jnp.dot(P, w))


def lprior(args: SamplingArgs):
    phi, delta = args.phi, args.delta
    P, k, w = args.P, args.k, args.weights

    log_prior = k * 0.5 * jnp.log(phi[0]) - 0.5 * phi[0] * _xPx(w, P)
    log_prior += gamma_logpdf(phi[0], alpha=phi[1], beta=delta[0] * phi[2])
    log_prior += gamma_logpdf(delta[0], alpha=delta[1], beta=delta[2])
    return log_prior


def sample_phi(args: SamplingArgs):
    wTPw = _xPx(args.weights, args.P)
    shape = 0.5 * args.k + args.phi[1]  # 0.5 * k + phialpha
    rate = (
        args.phi[2] * args.delta[0] + 0.5 * wTPw
    )  # phibeta * delta + 0.5 * wTPw
    return sample_gamma(shape, rate, shape=(1,), key=args.key)


def sample_delta(args: SamplingArgs):
    shape = args.phi[1] + args.delta[1]  # phialpha + deltaalpha
    rate = (
        args.phi[2] * args.phi[1] + args.delta[2]
    )  # phibeta * phialpha + deltabeta
    return sample_gamma(shape, rate=rate, shape=(1,), key=args.key)


def lpost(args: SamplingArgs):
    logprior = lprior(args)
    loglike = lnlikelihood(args.log_pdgrm, args.spline_model(args.weights))
    logpost = logprior + loglike
    if not jnp.isfinite(logpost):
        raise ValueError(
            f"logpost is not finite:\nlnpri: {logprior},\nlnlike: {loglike},\nlnpost: {logpost}"
        )
    return logpost


def mcmc_step(args: SamplingArgs):
    # update weights by proposing new weight from normal distribution centered at current weight with std sigma (only keep if LnPosterior better)
    weights = update_weights(args)
    # update phi by sampling from gamma distribution
    phi = sample_phi(args)
    # update delta by sampling from gamma distribution
    delta = sample_delta(args)
