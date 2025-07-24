"""SVI for Log P-splines on Periodograms

I thought SVI would be great for log P-splines
The weights are all normal distributed.

However, the phi and delta parameters are tricky...


Im not sure what is wrong with this, it doesnt work well at all.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.distributions import Normal, TransformedDistribution
from numpyro.distributions.transforms import ExpTransform
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_value
from numpyro.infer.svi import SVIState

from log_psplines.psplines import LogPSplines, Periodogram, build_spline


@jax.jit
def log_likelihood(
    weights: jnp.ndarray,
    log_pdgrm: jnp.ndarray,
    basis_matrix: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> jnp.ndarray:
    ln_model = build_spline(basis_matrix, weights, log_parametric)
    integrand = ln_model + jnp.exp(log_pdgrm - ln_model)
    return -0.5 * jnp.sum(integrand)


def bayesian_model(
    log_pdgrm: jnp.ndarray,
    lnspline_basis: jnp.ndarray,
    penalty_matrix: jnp.ndarray,
    ln_parametric: jnp.ndarray,
    alpha_phi,
    beta_phi,
    alpha_delta,
    beta_delta,
):
    # delta ~ Gamma(...)  → log_delta ~ TransformedDistribution
    log_delta = numpyro.sample(
        "log_delta",
        TransformedDistribution(
            dist.Gamma(concentration=alpha_delta, rate=beta_delta),
            ExpTransform(),
        ),
    )
    delta = jnp.exp(log_delta)

    # phi ~ Gamma(...), rate = beta_phi * delta
    log_phi = numpyro.sample(
        "log_phi",
        TransformedDistribution(
            dist.Gamma(concentration=alpha_phi, rate=delta * beta_phi),
            ExpTransform(),
        ),
    )
    phi = jnp.exp(log_phi)

    # 3. Sample weights (unchanged)
    k = penalty_matrix.shape[0]
    w = numpyro.sample("weights", dist.Normal(0, 1).expand([k]).to_event(1))

    # 5. Prior factor for weights ~ MVN(0, (phi * P)^-1)
    wPw = jnp.dot(w, jnp.dot(penalty_matrix, w))
    log_prior_v = 0.5 * k * jnp.log(phi) - 0.5 * phi * wPw
    numpyro.factor("weights_prior", log_prior_v)

    # 5. Likelihood
    numpyro.factor(
        "ln_likelihood",
        log_likelihood(w, log_pdgrm, lnspline_basis, ln_parametric),
    )


def run_svi(
    pdgrm: Periodogram,
    parametric_model: jnp.ndarray = None,
    alpha_phi=1.0,
    beta_phi=1.0,
    alpha_delta=1e-4,
    beta_delta=1e-4,
    num_steps=10000,
    learning_rate=1e-4,
    rng_key=0,
    **spline_kwgs,
):
    """
    Run SVI optimization for log P-splines
    """
    # Setup
    rng_key = jax.random.PRNGKey(rng_key)
    log_pdgrm = jnp.log(pdgrm.power)

    # Build spline model
    spline_model = LogPSplines.from_periodogram(
        pdgrm,
        n_knots=spline_kwgs.get("n_knots", 10),
        degree=spline_kwgs.get("degree", 3),
        diffMatrixOrder=spline_kwgs.get("diffMatrixOrder", 2),
        parametric_model=parametric_model,
    )

    print(f"Spline model: {spline_model}")

    # Compute initial values
    delta_0 = alpha_delta / beta_delta
    phi_0 = alpha_phi / (beta_phi * delta_0)
    weights_0 = spline_model.weights

    ll = log_likelihood(
        spline_model.weights,
        log_pdgrm,
        spline_model.basis,
        spline_model.log_parametric_model,
    )
    print(f"Log-likelihood at initialization: {ll:.4f}")

    print("init log_phi:", jnp.log(phi_0))
    print("init phi:", phi_0)

    print(
        "init weights dot P weights:",
        float(
            jnp.dot(weights_0, jnp.dot(spline_model.penalty_matrix, weights_0))
        ),
    )
    print(
        "phi * wPw:",
        float(
            phi_0
            * jnp.dot(
                weights_0, jnp.dot(spline_model.penalty_matrix, weights_0)
            )
        ),
    )

    guide = AutoDiagonalNormal(
        bayesian_model,
        init_loc_fn=init_to_value(
            values={
                # "log_delta": jnp.log(delta_0),
                # "log_phi": jnp.log(phi_0),
                "weights": weights_0,
            }
        ),
    )

    # Setup SVI
    optimizer = optax.adam(learning_rate)
    svi = SVI(bayesian_model, guide, optimizer, loss=Trace_ELBO())

    # Initialize SVI state
    init_rng_key, rng_key = jax.random.split(rng_key)
    svi_state = svi.init(
        init_rng_key,
        log_pdgrm,
        spline_model.basis,
        spline_model.penalty_matrix,
        spline_model.log_parametric_model,
        alpha_phi,
        beta_phi,
        alpha_delta,
        beta_delta,
    )

    samples = guide.sample_posterior(
        rng_key=jax.random.PRNGKey(0),
        params=svi.get_params(svi_state),
        sample_shape=(100,),
    )

    print("phi samples:", jnp.exp(samples["log_phi"]))
    print("delta samples:", jnp.exp(samples["log_delta"]))

    # Run SVI optimization
    losses = []

    @jax.jit
    def svi_step(svi_state):
        svi_state, loss = svi.update(
            svi_state,
            log_pdgrm,
            spline_model.basis,
            spline_model.penalty_matrix,
            spline_model.log_parametric_model,
            alpha_phi,
            beta_phi,
            alpha_delta,
            beta_delta,
        )
        return svi_state, loss

    for step in range(num_steps):
        svi_state, loss = svi_step(svi_state)
        losses.append(loss)

        if step % 1000 == 0:
            print(f"Step {step:5d}, Loss: {loss:.4f}")

    print(f"100 first losses: {losses[:100]}")

    return {
        "svi_state": svi_state,
        "guide": guide,
        "svi": svi,
        "losses": jnp.array(losses),
        "spline_model": spline_model,
    }


def get_posterior_predictions(results, n_samples=1000):
    """
    Generate posterior predictive samples for the PSD
    """
    guide = results["guide"]
    svi_state = results["svi_state"]
    spline_model = results["spline_model"]
    svi = results["svi"]

    # Sample from posterior
    rng_key = jax.random.PRNGKey(42)
    posterior_samples = guide.sample_posterior(
        rng_key, svi.get_params(svi_state), sample_shape=(n_samples,)
    )

    # Transform back from log-space
    transformed_samples = {
        "delta": jnp.exp(posterior_samples["log_delta"]),
        "phi": jnp.exp(posterior_samples["log_phi"]),
        "weights": posterior_samples["weights"],
    }

    # Extract weight samples
    weights_samples = transformed_samples["weights"]

    # Compute PSD predictions
    ln_splines = jnp.array([spline_model(w) for w in weights_samples])
    psd_samples = jnp.exp(ln_splines)

    return {
        "psd_samples": psd_samples,
        "weights_samples": weights_samples,
        "posterior_samples": transformed_samples,  # Return in original space
        "raw_posterior_samples": posterior_samples,  # Keep transformed space too
    }
