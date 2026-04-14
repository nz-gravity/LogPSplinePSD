from __future__ import annotations

from types import SimpleNamespace

import pytest

from log_psplines.samplers.multivar.multivar_blocked_nuts import (
    MultivarBlockedNUTSSampler,
)


def _make_sampler(*, eta: float | str, eta_c: float, nb: int, nh: int):
    sampler = object.__new__(MultivarBlockedNUTSSampler)
    sampler.config = SimpleNamespace(eta=eta, eta_c=eta_c)
    sampler.Nb = nb
    sampler.Nh = nh
    return sampler


def test_resolve_eta_value_auto_uses_c_over_nbnh():
    sampler = _make_sampler(eta="auto", eta_c=2.0, nb=7, nh=4)

    eta = sampler._resolve_eta_value(None, channel_index=0)

    assert eta == pytest.approx(2.0 / (7 * 4))


def test_resolve_eta_value_auto_caps_at_one():
    sampler = _make_sampler(eta="auto", eta_c=8.0, nb=2, nh=2)

    eta = sampler._resolve_eta_value(None, channel_index=1)

    assert eta == pytest.approx(1.0)


def test_resolve_eta_value_manual_override_returns_float():
    sampler = _make_sampler(eta="auto", eta_c=2.0, nb=7, nh=4)

    eta = sampler._resolve_eta_value(0.3, channel_index=2)

    assert eta == pytest.approx(0.3)
