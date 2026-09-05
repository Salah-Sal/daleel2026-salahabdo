"""DSPy parser for label-blind discourse forests over frozen atoms."""

from __future__ import annotations

from typing import Literal

from . import runtime  # noqa: F401 -- cache/env setup before dspy import

import dspy
import pydantic

from .discourse_forest import RELATIONS


RelationCode = Literal[
    "ELABORATION",
    "CAUSE_REASON",
    "CONTRAST",
    "ATTRIBUTION",
    "SEQUENCE",
    "META_COMMENT",
    "OTHER_COHERENCE",
]


class ProposedDiscourseEdge(pydantic.BaseModel):
    parent: str = pydantic.Field(description="parent atom ID such as A003")
    child: str = pydantic.Field(description="child atom ID such as A004")
    relation: RelationCode = pydantic.Field(description="the discourse relation")


_RELATION_GUIDANCE = "\n".join(
    (
        "- ELABORATION: child specifies, explains, or illustrates parent.",
        "- CAUSE_REASON: child gives a cause, reason, consequence, or justification for parent.",
        "- CONTRAST: child contrasts, concedes, corrects, or opposes parent.",
        "- ATTRIBUTION: child is content associated with a source or speech frame in parent.",
        "- SEQUENCE: child is the next step in an explicitly ordered account or procedure.",
        "- META_COMMENT: child comments on, organizes, or transitions around another unit.",
        "- OTHER_COHERENCE: a clear local dependency exists but no specific relation fits.",
    )
)


class ParseAtomForest(dspy.Signature):
    __doc__ = f"""Build a compact discourse forest over the supplied Arabic atoms.

The atoms are exact source substrings in reading order. Return only clear,
local parent->child discourse dependencies. A child may have at most one
parent. The result may have several roots or no edges; do not force a single
hierarchy when the paragraph is fragmentary or relations are implicit. Never
invent atom IDs, rewrite text, or classify atoms into argumentative-unit
types. In particular, do not output any two-letter task label.

Relation inventory:
{_RELATION_GUIDANCE}
"""

    text: str = dspy.InputField(desc="the unchanged Arabic paragraph")
    genre: Literal["editorial", "debate"] = dspy.InputField()
    atom_inventory: list[str] = dspy.InputField(
        desc="exact atom IDs, offsets, and source substrings in reading order"
    )
    edges: list[ProposedDiscourseEdge] = dspy.OutputField(
        desc="clear parent->child edges only; [] is valid for an unstructured paragraph"
    )


class DiscourseForestParser(dspy.Module):
    """One frozen, label-blind parse call per multi-atom paragraph."""

    def __init__(self, *, cot: bool = False) -> None:
        super().__init__()
        predictor = dspy.ChainOfThought if cot else dspy.Predict
        self.parse_forest = predictor(ParseAtomForest)

    def forward(self, text: str, genre: str, atom_inventory: list[str]):
        inputs = {"text": text, "genre": genre, "atom_inventory": atom_inventory}
        last_error: Exception | None = None
        for rollout in range(3):
            try:
                if rollout == 0:
                    return self.parse_forest(**inputs)
                base_lm = dspy.settings.lm
                if base_lm is None:
                    break
                retry_lm = base_lm.copy(rollout_id=rollout, temperature=1.0)
                with dspy.context(lm=retry_lm):
                    return self.parse_forest(**inputs)
            except Exception as error:
                last_error = error
        assert last_error is not None
        raise last_error


assert tuple(RelationCode.__args__) == RELATIONS


__all__ = ["DiscourseForestParser", "ParseAtomForest", "ProposedDiscourseEdge"]
