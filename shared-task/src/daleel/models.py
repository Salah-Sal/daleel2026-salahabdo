"""Candidate model ladder and LM construction (D10/D15).

Provider reality (checked against the live OpenRouter catalog 2026-07-09):
OpenRouter is the only API we hold a key for; the funded account gives
1,000 requests/day and 20 requests/min on `:free` models. The D10 pool
entries with no OpenRouter route (ALLaM-2-7B, Qwen3-32B, Fanar, Jais) need
a Groq/Fanar key or Kaggle GPU serving and are deferred.

Ladder policy (user directive 2026-07-09): smallest first — validate the
plumbing on a cheap weak model, walk the bake-off up in size, and let a
larger model in only when the smaller one's scores justify the quota.
Closed track caps at 70B TOTAL parameters; MoE active-parameter counts do
not license exceeding it (qwen3-next-80b is open-track only).
"""

from dataclasses import dataclass

from . import runtime  # noqa: F401  — cache/env setup before dspy import

import dspy


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used in CLI args and experiment folders
    litellm_id: str
    params_b: float  # total parameters, the closed-track quantity
    closed_track: bool
    notes: str = ""
    # Closed execution accepts only an affirmative, source-checked status.
    # "unverified" is deliberately ineligible even if closed_track was set
    # optimistically when a new catalog entry was added.
    license_status: str = "verified-open"
    # extra dspy.LM kwargs that must travel with the model (e.g. pinning
    # reasoning_effort on thinking-by-default models whose thinking tokens
    # bill as output). Tuple of pairs because the dataclass is frozen.
    lm_kwargs: tuple = ()


LADDER = (
    ModelSpec(
        "llama-3.2-3b", "openrouter/meta-llama/llama-3.2-3b-instruct:free",
        3, True, "plumbing smoke tests; not a serious candidate",
    ),
    # gemma-3 has no :free route on OpenRouter — paid, but ~$0.02-0.04 per
    # val run (user-approved escalation 2026-07-09; log spend per run).
    ModelSpec(
        "gemma-3-4b", "openrouter/google/gemma-3-4b-it",
        4, True, "paid (cheap); smallest dense Gemma",
    ),
    ModelSpec(
        "nemotron-nano-9b", "openrouter/nvidia/nemotron-nano-9b-v2:free",
        9, True, "smallest free bake-off candidate",
    ),
    ModelSpec(
        "gemma-3-12b", "openrouter/google/gemma-3-12b-it",
        12, True, "paid (cheap)",
    ),
    ModelSpec(
        "gpt-oss-20b", "openrouter/openai/gpt-oss-20b:free",
        21, True, "MoE (3.6B active), reasoning-tuned",
    ),
    ModelSpec(
        "gemma-4-26b", "openrouter/google/gemma-4-26b-a4b-it:free",
        26, True, "MoE (4B active); :free often upstream-contended",
    ),
    ModelSpec(
        "gemma-4-26b-paid", "openrouter/google/gemma-4-26b-a4b-it",
        26, True, "paid fallback when the :free route is contended",
    ),
    ModelSpec(
        "gemma-3-27b", "openrouter/google/gemma-3-27b-it",
        27, True, "paid (cheap)",
    ),
    ModelSpec(
        "gemma-4-31b", "openrouter/google/gemma-4-31b-it:free",
        31, True, "dense mid-size; :free often upstream-contended",
    ),
    ModelSpec(
        "gemma-4-31b-paid", "openrouter/google/gemma-4-31b-it",
        31, True, "paid fallback when the :free route is contended",
    ),
    # ---- broad open-weight sweep (user request 2026-07-09): per-family
    # representatives, paid tier (cheap), smallest first. Models marked
    # "license TBV" postdate verifiable knowledge — confirm the weights are
    # actually open before using them in a CLOSED-track submission.
    ModelSpec(
        "llama-3.1-8b", "openrouter/meta-llama/llama-3.1-8b-instruct",
        8, True, "cheapest paid rung ($0.02/M)",
    ),
    ModelSpec(
        "qwen3-8b", "openrouter/qwen/qwen3-8b",
        8, True, "verified open-weight; hybrid-thinking generation",
    ),
    ModelSpec(
        "qwen3.5-9b", "openrouter/qwen/qwen3.5-9b",
        9, True, "latest small Qwen; license TBV for closed track", "unverified",
    ),
    ModelSpec(
        "granite-4.1-8b", "openrouter/ibm-granite/granite-4.1-8b",
        8, True, "Apache open weights; single upstream (WandB) — 429'd "
        "even paid on 2026-07-09, bake-off row missing",
    ),
    ModelSpec(
        "command-r7b", "openrouter/cohere/command-r7b-12-2024",
        8, True, "CC-BY-NC open weights (fine for non-commercial shared task)",
    ),
    ModelSpec(
        "qwen3-14b", "openrouter/qwen/qwen3-14b",
        14, True, "verified open-weight",
    ),
    ModelSpec(
        "ministral-14b", "openrouter/mistralai/ministral-14b-2512",
        14, True, "license TBV for closed track", "unverified",
    ),
    ModelSpec(
        "phi-4-14b", "openrouter/microsoft/phi-4",
        14, True, "MIT weights; 16k ctx (fits our longest prompts)",
    ),
    ModelSpec(
        "mistral-small-24b", "openrouter/mistralai/mistral-small-3.2-24b-instruct",
        24, True, "Apache open weights",
    ),
    ModelSpec(
        "qwen3.6-27b", "openrouter/qwen/qwen3.6-27b",
        27, True, "latest mid dense Qwen; license TBV for closed track", "unverified",
    ),
    ModelSpec(
        "qwen3-30b-a3b", "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
        30, True, "verified open-weight MoE (3B active)",
    ),
    ModelSpec(
        "nemotron-nano-30b", "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        30, True, "MoE (3B active); :free exists but quota-bound today",
    ),
    ModelSpec(
        "qwen3-32b", "openrouter/qwen/qwen3-32b",
        32, True, "the original D10 closed-track default, verified open",
    ),
    ModelSpec(
        "qwen3.5-35b-a3b", "openrouter/qwen/qwen3.5-35b-a3b",
        35, True, "latest Qwen MoE; license TBV for closed track", "unverified",
    ),
    ModelSpec(
        "llama-3.3-70b", "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        70, True, "at the closed-track cap",
    ),
    ModelSpec(
        "llama-3.3-70b-paid", "openrouter/meta-llama/llama-3.3-70b-instruct",
        70, True, "paid serving of the same checkpoint; :free starved the "
        "D10 bake-off and the 2026-07-14 P4 voter run",
    ),
    ModelSpec(
        "qwen3-next-80b", "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
        80, False, "80B total exceeds the 70B cap — open track only",
    ),
    ModelSpec(
        "gpt-oss-120b", "openrouter/openai/gpt-oss-120b:free",
        120, False, "open track only",
    ),
    # ---- compile-time teacher/reflection models (user key added 2026-07-09,
    # DEEPSEEK_API_KEY in repo-root .env). Size was unpublished when these
    # entries were written; the V4 report (arXiv 2606.19348, 2026) later gave
    # 1.6T total / 49B active for -pro and 284B / 13B for -flash, and released
    # both checkpoints. Either way both are far over the 70B closed-track cap,
    # so neither is ever a task model for a closed submission. Closed-track use
    # at compile time only was the D7 organizer question — assume NO until the
    # Jul 13 info session says otherwise; it was never answered.
    ModelSpec(
        "deepseek-v4-flash", "deepseek/deepseek-v4-flash",
        284, False, "teacher/reflection only; cheap tier; 13B active",
        "open-weights, over the closed-track cap",
    ),
    ModelSpec(
        "deepseek-v4-pro", "deepseek/deepseek-v4-pro",
        1600, False, "teacher/reflection only; strongest available teacher; "
        "49B active", "open-weights, over the closed-track cap",
    ),
    # ---- Gemini API (user key added 2026-07-09). Gemini-branded models are
    # proprietary -> open track only. The gemma-4 checkpoints on this API
    # are open weights (closed-legal) and served FREE per the D15 survey —
    # if their scores match the OpenRouter rows, compiles run at $0.
    # gemini-2.5-flash-lite is listed by /models but 404s on generateContent
    # ("no longer available", checked 07-09) — current lite tier only.
    ModelSpec(
        "gemini-3.1-flash-lite", "gemini/gemini-3.1-flash-lite",
        0, False, "cheapest live Gemini tier", "proprietary",
    ),
    ModelSpec(
        "gemini-2.0-flash", "gemini/gemini-2.0-flash",
        0, False, "older-gen flash, still served; 8192-token output cap", "proprietary",
    ),
    ModelSpec(
        "gemini-3-flash-preview", "gemini/gemini-3-flash-preview",
        0, False, "thinking ON by default at high level, billed as output "
        "($3/M) — reasoning_effort pinned to avoid the 4x premium",
        "proprietary",
        lm_kwargs=(("reasoning_effort", "minimal"),),
    ),
    # ---- Cohere API (user key added 2026-07-10, COHERE_API_KEY in repo-root
    # .env). command-a-03-2025 is open-weight (CC-BY-NC) but 111B total blows
    # the 70B closed cap; command-a-plus openness/size unpublished. Both are
    # open-track-only rows. c4ai-aya-expanse-32b (open, 32B, Arabic-tuned) is
    # the one closed-track-eligible Cohere model — untried.
    ModelSpec(
        "command-a-plus", "cohere_chat/command-a-plus-05-2026",
        0, False, "Cohere flagship 2026; openness TBV", "unverified",
    ),
    ModelSpec(
        "command-a", "cohere_chat/command-a-03-2025",
        111, False, "open weights but over the 70B cap; $2.5/$10 per M",
    ),
    ModelSpec(
        "gemma-4-26b-gemini", "gemini/gemma-4-26b-a4b-it",
        26, True, "same checkpoint as gemma-4-26b, free Gemini API serving",
    ),
    ModelSpec(
        "gemma-4-31b-gemini", "gemini/gemma-4-31b-it",
        31, True, "same checkpoint as gemma-4-31b, free Gemini API serving",
    ),
)

SPECS = {spec.key: spec for spec in LADDER}

# ---- Recommended serving configuration (CFG-J1 judge probe, 2026-07-29) ----
# Measured-best default for future gemma-4-31b-paid runs: pin the CoreWeave
# bf16 endpoint. Evidence (CFG-J1, registered 056a329; report in
# MODEL_CONFIG_OPTIMIZATION_DESIGN.md s11): the unpinned route hopped NINE
# upstreams across four precisions in 20 calls; the pin scored best on the
# stress set (0.7332 vs 0.7219 unpinned / 0.7178 fp4), cut the latency tail
# from 1128s worst-case to 45s, at identical cost. The rest of the
# recommended profile is already the codebase default: temperature 0.0,
# thinking OFF (never send reasoning.enabled — it starves max_tokens and
# collapses recall, CFG-J1 C4), ChatAdapter (never JSONAdapter), max_tokens
# 6000 for full runs.
#
# NOT applied by default in this cycle: extra_body changes every DSPy cache
# key, which would break free cache-replay of the frozen recipes' runs and
# their byte-identical control replays. Opt in via make_lm(provider_pin=...);
# flipping it to the default is a next-cycle (post-2026-07-31) change.
RECOMMENDED_PROVIDER_PIN = {
    "order": ["coreweave"],
    "allow_fallbacks": False,
    "quantizations": ["bf16"],
}


def make_lm(
    model: str | ModelSpec,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    provider_pin: bool | dict | None = None,
    **kwargs,
) -> dspy.LM:
    """A cached, retrying dspy.LM for a ladder model (or any litellm id).

    Temperature 0 is the D13 determinism default; self-consistency runs
    (D11) pass their own temperature. Generous num_retries because the
    free tier rate-limits at 20 req/min and litellm backs off on 429s.

    provider_pin: True pins RECOMMENDED_PROVIDER_PIN (CFG-J1 winner); a
    dict pins a custom OpenRouter provider object. Merged under
    extra_body["provider"] (an explicit caller extra_body provider wins).
    Two caveats from CFG-J1/MODEL_CONFIG_OPTIMIZATION_DESIGN: pinning
    changes DSPy cache keys, so never enable it when replaying a frozen
    run's cache; and LiteLLM has paths that silently drop extra_body, so
    pinned runs must verify the served provider from response metadata
    (lm.history entry response.provider), never trust the request side.
    """
    spec = SPECS.get(model, model) if isinstance(model, str) else model
    litellm_id = spec.litellm_id if isinstance(spec, ModelSpec) else spec
    spec_kwargs = dict(spec.lm_kwargs) if isinstance(spec, ModelSpec) else {}
    if provider_pin:
        pin = RECOMMENDED_PROVIDER_PIN if provider_pin is True else provider_pin
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("provider", dict(pin))
        kwargs["extra_body"] = extra
    return dspy.LM(
        litellm_id,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=8,
        **{**spec_kwargs, **kwargs},  # call-site kwargs win
    )
