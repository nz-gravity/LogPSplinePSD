import warnings

import numpy as np
import pytest

import log_psplines.datatypes.multivar_utils as multivar_utils
import log_psplines.mcmc as mcmc
import log_psplines.preprocessing.coarse_grain as coarse_grain


def _typecheck_stack_available() -> bool:
    try:
        import beartype  # noqa: F401
        import jaxtyping  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _typecheck_stack_available(),
    reason="jaxtyping+beartype not installed",
)


def test_runtime_typecheck_rejects_complex_univar_input(monkeypatch) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "1")

    freqs = np.linspace(0.1, 1.0, 8)
    spec = coarse_grain.compute_binning_structure(freqs, Nh=2)
    power_complex = np.ones(8, dtype=np.complex128)

    with pytest.raises(Exception):
        coarse_grain.apply_coarse_graining_univar(power_complex, spec)


def test_runtime_typecheck_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "0")

    freqs = np.linspace(0.1, 1.0, 8)
    spec = coarse_grain.compute_binning_structure(freqs, Nh=2)
    power_complex = np.ones(8, dtype=np.complex128)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = coarse_grain.apply_coarse_graining_univar(power_complex, spec)

    assert out.shape == (4,)


def test_runtime_typecheck_rejects_list_frequency_targets(monkeypatch) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "1")

    psd = np.linspace(1.0, 4.0, 4)
    freq_src = np.linspace(0.1, 0.4, 4)
    freq_tgt_list = [0.1, 0.25, 0.4]

    with pytest.raises(Exception):
        mcmc._interp_psd_array(psd, freq_src, freq_tgt_list)


def test_runtime_typecheck_can_allow_list_targets_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "0")

    psd = np.linspace(1.0, 4.0, 4)
    freq_src = np.linspace(0.1, 0.4, 4)
    freq_tgt_list = [0.1, 0.25, 0.4]

    out = mcmc._interp_psd_array(psd, freq_src, freq_tgt_list)
    assert out.shape == (3,)


def test_runtime_typecheck_rejects_list_wishart_stack(monkeypatch) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "1")

    u_stack_list = [[[1.0, 0.0], [0.0, 1.0]]]
    with pytest.raises(Exception):
        multivar_utils.u_to_wishart_matrix(u_stack_list)


def test_runtime_typecheck_can_allow_list_wishart_stack_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "0")

    u_stack_list = [[[1.0, 0.0], [0.0, 1.0]]]
    out = multivar_utils.u_to_wishart_matrix(u_stack_list)
    assert out.shape == (1, 2, 2)
