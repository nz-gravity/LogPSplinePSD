from log_psplines.preprocessing.configs import (
    RunMCMCConfig,
    SamplerFactoryConfig,
)
from log_psplines.preprocessing.sampler_factory import (
    _build_common_sampler_kwargs,
    _validate_extra_kwargs,
)
from log_psplines.samplers import MultivarBlockedNUTSConfig, NUTSConfig


def test_univar_extra_kwargs_can_override_common_sampler_fields():
    run_config = RunMCMCConfig(
        extra_kwargs={"compute_psis": False, "target_accept_prob": 0.91}
    )
    factory_config = SamplerFactoryConfig(
        sampler_type="nuts",
        run_config=run_config,
        scaling_factor=1.0,
        true_psd=None,
        channel_stds=None,
    )

    common_kwargs = _build_common_sampler_kwargs(factory_config)
    extra_kwargs = _validate_extra_kwargs(NUTSConfig, run_config.extra_kwargs)
    config_kwargs = {
        **common_kwargs,
        "target_accept_prob": run_config.nuts.target_accept_prob,
        "max_tree_depth": run_config.nuts.max_tree_depth,
        "dense_mass": run_config.nuts.dense_mass,
        "init_from_vi": run_config.vi.init_from_vi,
        "vi_steps": run_config.vi.vi_steps,
        "vi_lr": run_config.vi.vi_lr,
        "vi_guide": run_config.vi.vi_guide,
        "vi_posterior_draws": run_config.vi.vi_posterior_draws,
        "vi_progress_bar": run_config.vi.vi_progress_bar,
        **extra_kwargs,
    }

    config = NUTSConfig(**config_kwargs)

    assert config.compute_psis is False
    assert config.target_accept_prob == 0.91


def test_multivar_extra_kwargs_can_override_common_sampler_fields():
    run_config = RunMCMCConfig(
        extra_kwargs={
            "compute_psis": False,
            "target_accept_prob": 0.93,
            "eta": "auto",
            "eta_c": 2.5,
        }
    )
    factory_config = SamplerFactoryConfig(
        sampler_type="multivar_blocked_nuts",
        run_config=run_config,
        scaling_factor=1.0,
        true_psd=None,
        channel_stds=None,
    )

    common_kwargs = _build_common_sampler_kwargs(factory_config)
    extra_kwargs = _validate_extra_kwargs(
        MultivarBlockedNUTSConfig, run_config.extra_kwargs
    )
    config_kwargs = {
        **common_kwargs,
        "target_accept_prob": run_config.nuts.target_accept_prob,
        "target_accept_prob_by_channel": run_config.nuts.target_accept_prob_by_channel,
        "max_tree_depth": run_config.nuts.max_tree_depth,
        "max_tree_depth_by_channel": run_config.nuts.max_tree_depth_by_channel,
        "dense_mass": run_config.nuts.dense_mass,
        "init_from_vi": run_config.vi.init_from_vi,
        "vi_steps": run_config.vi.vi_steps,
        "vi_lr": run_config.vi.vi_lr,
        "vi_guide": run_config.vi.vi_guide,
        "vi_posterior_draws": run_config.vi.vi_posterior_draws,
        "vi_progress_bar": run_config.vi.vi_progress_bar,
        "alpha_phi_theta": run_config.nuts.alpha_phi_theta,
        "beta_phi_theta": run_config.nuts.beta_phi_theta,
        "design_from_vi": run_config.nuts.design_from_vi,
        "design_from_vi_tau": run_config.nuts.design_from_vi_tau,
        **extra_kwargs,
    }

    config = MultivarBlockedNUTSConfig(**config_kwargs)

    assert config.compute_psis is False
    assert config.target_accept_prob == 0.93
    assert config.eta == "auto"
    assert config.eta_c == 2.5
