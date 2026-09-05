"""No-network executable-bundle tests for v3 role programs."""

import dspy
import pytest

from daleel.dspy_span_roles import BalancedRoleDemoSelector, SpanRoleDecision
from daleel.provenance import ComplianceError
from daleel.role_artifacts import build_role_bundle, load_role_bundle


def _example(pid, role, genre="editorial"):
    target = role if role else "فراغ"
    text = f"سياق {pid} {target} ونهاية"
    start = text.index(target)
    return dspy.Example(
        paragraph_id=pid,
        text=text,
        genre=genre,
        start=start,
        end=start + len(target),
        adu_roles=[] if not role else [role],
    ).with_inputs("paragraph_id", "text", "genre", "start", "end")


def _demo_shape(demos):
    return [
        (demo.paragraph_context, demo.genre, demo.target, tuple(demo.adu_roles))
        for demo in demos
    ]


def _bundle(tmp_path):
    run = tmp_path / "run"
    compiled = run / "compiled"
    compiled.mkdir(parents=True)
    examples = [
        _example(1, "CO"),
        _example(2, "AS", "debate"),
        _example(3, "ST"),
        _example(4, None, "debate"),
    ]
    program = SpanRoleDecision(
        demo_selector=BalancedRoleDemoSelector(
            examples, max_demos=4, context_chars=123, same_genre_bonus=0.07
        ),
        context_chars=456,
    )
    program.classify_span.signature = program.classify_span.signature.with_instructions(
        "BUNDLED INSTRUCTIONS"
    )
    state = compiled / "role.json"
    program.save(state, save_program=False)
    bundle_path, bundle = build_role_bundle(
        run_dir=run,
        role_state=state,
        memory_examples=examples,
        settings={
            "role_cot": False,
            "demo_selector": True,
            "max_demos": 4,
            "context_chars": 456,
            "demo_context_chars": 123,
            "same_genre_bonus": 0.07,
            "granularity": "connective",
            "containment_threshold": 0.8,
        },
        proposal_sha256="a" * 64,
        fold_manifest_sha256="b" * 64,
        model="gemma-4-31b-paid",
        track="closed",
        setting="both",
    )
    return run, state, bundle_path, bundle, program, examples


def test_role_bundle_reconstructs_exact_state_and_dynamic_demos(tmp_path):
    _, state, _, bundle, original, examples = _bundle(tmp_path)
    loaded_bundle, loaded_examples = load_role_bundle(state)
    assert loaded_bundle == bundle
    assert len(loaded_examples) == len(examples)

    settings = loaded_bundle["settings"]
    restored = SpanRoleDecision(
        demo_selector=BalancedRoleDemoSelector(
            loaded_examples,
            max_demos=settings["max_demos"],
            context_chars=settings["demo_context_chars"],
            same_genre_bonus=settings["same_genre_bonus"],
        ),
        context_chars=settings["context_chars"],
    )
    restored.load(state)
    assert restored.classify_span.signature.instructions == "BUNDLED INSTRUCTIONS"

    query = _example(99, "AS", "debate")
    original_demos = original.demo_selector.select(**query.inputs())
    restored_demos = restored.demo_selector.select(**query.inputs())
    assert _demo_shape(restored_demos) == _demo_shape(original_demos)

    captured = []

    def predictor(**kwargs):
        captured.append(
            {
                "paragraph_context": kwargs["paragraph_context"],
                "genre": kwargs["genre"],
                "target": kwargs["target"],
                "demos": _demo_shape(kwargs["demos"]),
            }
        )
        return dspy.Prediction(adu_roles=["AS"])

    original.classify_span = predictor
    restored.classify_span = predictor
    original(**query.inputs())
    restored(**query.inputs())
    assert captured[0] == captured[1]


@pytest.mark.parametrize("target", ["state", "memory"])
def test_role_bundle_rejects_tampering(tmp_path, target):
    run, state, _, bundle, _, _ = _bundle(tmp_path)
    path = state if target == "state" else run / bundle["memory"]["relative_path"]
    path.write_text(path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ComplianceError, match="SHA-256"):
        load_role_bundle(state)
