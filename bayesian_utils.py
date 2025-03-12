import jax.numpy as jnp
from jax import random, jit
from jax.scipy.stats import gamma
from jax.scipy.linalg import solve
from jax.scipy.special import gammaln
from jax import grad, value_and_grad
from jax import lax
from jax.random import PRNGKey
from jax import random




@jit
def _xPx(w, P):
    return jnp.dot(w, jnp.dot(P, w))

@jit
def lprior(args):
    phialpha, phibeta, deltaalpha, deltabeta = args.phialpha, args.phibeta, args.deltaalpha, args.deltabeta
    phi, delta = args.phi, args.delta
    P = args.spline_model.penalty_matrix
    k = args.spline_model.n_basis
    w = args.w

    log_prior = k * 0.5 * jnp.log(phi) - 0.5 * phi * _xPx(w, P)
    log_prior += gamma.logpdf(phi, a=phialpha, scale=1 / (delta * phibeta))
    log_prior += gamma.logpdf(delta, a=deltaalpha, scale=1 / deltabeta)
    return log_prior

@jit
def phi_prior(k, w, P, phialpha, phibeta, delta):
    wTPw = _xPx(w, P)
    shape = 0.5 * k + phialpha
    rate = phibeta * delta + 0.5 * wTPw
    return random.gamma(shape, scale=1 / rate)

@jit
def delta_prior(phi, phialpha, phibeta, deltaalpha, deltabeta):
    shape = phialpha + deltaalpha
    rate = phibeta * phi + deltabeta
    return gamma(shape, scale=1 / rate)

@jit
def sample_phidelta(args, key):
    w, phialpha, phibeta, deltaalpha, deltabeta, delta, spline_model = args.w, args.phialpha, args.phibeta, args.deltaalpha, args.deltabeta, args.delta, args.spline_model
    k = spline_model.n_basis
    key, subkey = random.split(key)
    phi = phi_prior(k, w, spline_model.penalty_matrix, phialpha, phibeta, delta).rvs(random_key=subkey)
    key, subkey = random.split(key)
    delta = delta_prior(phi, phialpha, phibeta, deltaalpha, deltabeta).rvs(random_key=subkey)
    return phi, delta

@jit
def lpost(args):
    w, data, spline_model = args.w, args.data, args.spline_model
    logprior = lprior(args)
    loglike = spline_model.lnlikelihood(data=data, weights=w)
    logpost = logprior + loglike
    if not jnp.isfinite(logpost):
        raise ValueError(f"logpost is not finite:\nlnpri: {logprior},\nlnlike: {loglike},\nlnpost: {logpost}")
    return logpost