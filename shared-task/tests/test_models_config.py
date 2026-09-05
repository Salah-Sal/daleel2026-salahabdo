"""Contract tests for the recommended serving configuration (CFG-J1).

The pin must be opt-in (cache-key stability for frozen-run replay), must
inject the exact registered provider object, and must never clobber an
explicit caller-provided provider.
"""

from daleel.models import RECOMMENDED_PROVIDER_PIN, make_lm


def test_default_has_no_extra_body():
    # Cache-key stability: default construction must be byte-identical in
    # kwargs to every pre-CFG-J1 run, or frozen-cache replay breaks.
    lm = make_lm("gemma-4-31b-paid")
    assert "extra_body" not in lm.kwargs


def test_provider_pin_true_injects_recommended():
    lm = make_lm("gemma-4-31b-paid", provider_pin=True)
    assert lm.kwargs["extra_body"]["provider"] == RECOMMENDED_PROVIDER_PIN
    # and it is a copy, not the module constant itself
    assert lm.kwargs["extra_body"]["provider"] is not RECOMMENDED_PROVIDER_PIN


def test_provider_pin_custom_dict():
    pin = {"order": ["deepinfra"], "quantizations": ["fp8"]}
    lm = make_lm("gemma-4-31b-paid", provider_pin=pin)
    assert lm.kwargs["extra_body"]["provider"] == pin


def test_caller_extra_body_provider_wins():
    caller = {"provider": {"order": ["venice"]}, "reasoning": {"enabled": False}}
    lm = make_lm("gemma-4-31b-paid", provider_pin=True, extra_body=caller)
    assert lm.kwargs["extra_body"]["provider"] == {"order": ["venice"]}
    assert lm.kwargs["extra_body"]["reasoning"] == {"enabled": False}


def test_recommended_pin_shape():
    # The registered CFG-J1 winner, exactly.
    assert RECOMMENDED_PROVIDER_PIN == {
        "order": ["coreweave"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
    }
