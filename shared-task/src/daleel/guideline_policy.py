"""Guideline-native label policy: seeded from the OFFICIAL annotation guidelines.

Counterpart to daleel.policy (which predates the guidelines and was inferred
from Webis + the data audit). Every block here is distilled from the official
``data/Annotation Guidelines/Annotation Guidelines EN.md`` (official repo,
commit 49f000c, first public release 2026-07-29) — the document the
annotators were actually trained on. Full derivation, deltas vs policy.py,
and the U1-U11 rule table: guideline-native design note (not in this release).

Single-source rule (same as daleel.policy): the guideline-native signatures
(daleel.guideline_programs) and any future metric feedback for their
compiles must BOTH quote from this module, never re-phrase locally —
divergent phrasings make GEPA oscillate.

This module is deliberately ASCII/English-only (no Arabic bytes) so the
repository's mechanical staged-diff sweep stays clean; Arabic-language
adaptations of the rules (pro-drop subjects, qala-inna attribution) are
phrased descriptively in English.
"""

GUIDELINE_SOURCE = (
    "Daleel2026 official repo: data/Annotation Guidelines/"
    "Annotation Guidelines EN.md @ 49f000c"
)

GUIDELINE_CONTEXT = (
    "The text is one Arabic paragraph of argumentative discourse: either a "
    "newspaper editorial (edited MSA prose) or a transcribed WSDC-style "
    "debate speech (spoken register, procedural passages, occasional "
    "Latin-script code-switching, speaker markers). The author presents a "
    "thesis on a controversial issue and defends it, supporting their "
    "position with arguments while also considering counterarguments. "
    "Argumentative discourse units (units) are the propositions put "
    "forward, explicitly or implicitly, to discuss or defend that thesis."
)

# ---- Unit identity (guidelines §3.1; rule ids U1-U11 in the design doc) ----

UNIT_RULES = """Copy every unit VERBATIM from the paragraph — character-for-character, in
reading order, no paraphrase, no ellipsis, no words added or dropped.

What counts as one unit:
- One proposition, with a subject, a verb, and an object where grammar
  requires one. The subject may be a pronoun, a relative, or carried by the
  Arabic verb's own inflection (pro-drop); a fragment with no predication is
  not a unit.
- A unit does not cross a sentence boundary.
- Consecutive propositions inside one sentence are SEPARATE units, whether
  joined by a connective (and, but, rather) or simply juxtaposed.
- Coordinated propositions that share one predicate ("X was not done to A or
  to B") remain ONE unit.
- Attribution plus its content ("X said that ...") is ONE unit spanning the
  whole statement — never split the source from what it says.
- A proposition interrupted by a parenthetical keeps the interruption inside
  the unit.
- Rhetorical questions that convey a claim are units and keep their question
  mark; genuine information-seeking questions are not units.

What stays OUT of a unit's span:
- Transitional expressions and generic framing before or after the
  proposition ("therefore", "in either case, we see that").
- Edge punctuation — except quotation marks around quoted testimony and the
  question mark of a rhetorical question."""

# ---- Categories (guidelines §3.2, incl. the official contrastive razors) ----

GUIDELINE_LABEL_POLICY = {
    "CO": (
        "CO — Common Ground: generally accepted knowledge, self-evident "
        "facts, established truths, and near-universal values and moral "
        "commonplaces ('we should not encourage children to smoke'), "
        "including an objective explanation of how a process or legal "
        "procedure works. General propositions, not specific events. Test: "
        "would nearly all readers accept it WITHOUT supporting evidence? "
        "If yes, it is CO even if previously unknown to them."
    ),
    "AS": (
        "AS — Assumption: claims that require support — the author's "
        "inferences, opinions, judgments, evaluations, predictions, and "
        "contested generalizations. Claims with only vague attribution "
        "('some scholars claim ...') are AS, not TE."
    ),
    "TE": (
        "TE — Testimony: evidence that quotes or cites an IDENTIFIABLE "
        "source — a named expert, authority, witness, organization, or "
        "other identifiable voice. In debates, restating the opposing "
        "team's claims is TE."
    ),
    "ST": (
        "ST — Statistics: evidence from quantitative studies, statistical "
        "analyses, or empirical findings. The source does not have to be "
        "stated."
    ),
    "AN": (
        "AN — Anecdote: evidence from personal experience, stories, "
        "real-life events, or concrete examples. Quantities drawn from "
        "personal experience ('80% of the people I know ...') are AN, not "
        "ST; experiential generalizations ('everyone in my field ...') are "
        "AN."
    ),
    "OT": (
        "OT — Other: does not contribute meaningfully to the argumentative "
        "discourse — greetings, thanks, procedural debate speech, "
        "meta-comments."
    ),
}

DECISION_PROCEDURE = """Decide each unit's categories in three steps:
1. Does the unit contribute to the argument at all? If not, it is OT.
2. Does it present evidence anchored outside the author's own reasoning?
   An identifiable voice being quoted or cited is TE; quantitative or
   empirical findings are ST; personal experience, a story, a real event,
   or a concrete example is AN. Assign more than one category when several
   genuinely apply — a statistic with a stated source is 'ST/TE'.
3. Otherwise the unit is an unsupported proposition: CO if nearly all
   readers would accept it without evidence, AS if it needs support."""

# ---- Per-label razor lines for metric feedback on wrong predictions ----
# (the guideline-native analogue of policy.SPURIOUS_HINTS; same single-source
# rule — compiles against these signatures must quote these, not rephrase)

GUIDELINE_HINTS = {
    "CO": (
        "CO passes the audience test: nearly all readers accept it without "
        "evidence — this includes near-universal values and moral "
        "commonplaces; a contested claim the author argues FOR is AS."
    ),
    "AS": (
        "AS is for claims requiring support; a claim quoted from an "
        "identifiable source is TE, and an audience-accepted truth or "
        "shared value is CO."
    ),
    "TE": (
        "TE needs an IDENTIFIABLE stated source; vague attribution ('some "
        "scholars say') is AS."
    ),
    "ST": (
        "ST needs quantitative or empirical findings; numbers drawn from "
        "personal experience are AN."
    ),
    "AN": (
        "AN needs experience, a story, a real event, or a concrete example; "
        "a general contested claim is AS."
    ),
    "OT": (
        "OT only when the unit contributes nothing to the argument; any "
        "content that supports or attacks a position belongs to another "
        "label."
    ),
}
