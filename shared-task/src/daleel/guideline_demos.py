"""The official guideline examples as DSPy demos for the guideline-native
signatures.

Every example below is copied from the OFFICIAL annotation guidelines
(``data/Annotation Guidelines/Annotation Guidelines EN.md`` @ 49f000c):
§3.1's eight unit-identification examples -> SegmentUnits demos, §3.2's
seven category examples plus the debate-testimony inline example from the
TE definition -> CategorizeUnits demos. They are the guidelines' own
English (Webis-inherited) sentences — the exact material the annotators
trained on — and embed ZERO organizer dataset text, so unlike bootstrapped
demos they are committable and printable in the paper.

English demos over Arabic inputs = cross-lingual demonstration; the
ablation axis (EN as-is / self-translated AR / none) is design
question 3 of the guideline-native redesign.

Demo fields mirror the signature fields exactly (text, genre, units[,
unit_labels]); multi-category answers use the same 'ST/TE' slash form the
CategorizeUnits output field asks for.
"""

import dspy

# ---- §3.1: unit identification -------------------------------------------
# (text, units) — units must be verbatim substrings of text; brackets in the
# guidelines mark exactly these spans.

_SEGMENT_EXAMPLES: list[tuple[str, list[str]]] = [
    # ex.1: consecutive propositions in one sentence = separate units
    (
        "We should demolish the building, It is full of asbestos.",
        ["We should demolish the building", "It is full of asbestos"],
    ),
    # ex.2: referential subject (relative pronoun) still makes a unit
    (
        "That man admits his mistakes, which is why I trust him.",
        ["That man admits his mistakes", "which is why I trust him"],
    ),
    # ex.3: overlapping propositions sharing one predicate = ONE unit
    (
        "Viruses were not created to make money or to play practical jokes.",
        ["Viruses were not created to make money or to play practical jokes"],
    ),
    # ex.4: transitional expressions / generic framing stay outside the span
    (
        "In either case, we see that learning to cooperate is beneficial, "
        "she already knew that as well.",
        ["learning to cooperate is beneficial", "she already knew that"],
    ),
    # ex.5: interrupted proposition keeps the interruption inside the unit
    (
        "Many people—and you know some of them—are kinder in the morning.",
        ["Many people—and you know some of them—are kinder in the morning"],
    ),
    # ex.6: attribution + content ("said ... that") = ONE unit
    (
        "Professor Miller said in his lecture that drinking milk strengthens bones.",
        ["Professor Miller said in his lecture that drinking milk strengthens bones"],
    ),
    # ex.7: quote ending mid-sentence belongs to the sentence; "Trust me!"
    # is no unit (no explicit subject)
    (
        "“Trust me! You should drink as much milk as possible!” he concluded.",
        ["You should drink as much milk as possible!” he concluded"],
    ),
    # ex.8: rhetorical question = unit (question mark kept); genuine
    # information question = not a unit
    (
        "What are the other options? Who would be a better candidate than Obama?",
        ["Who would be a better candidate than Obama?"],
    ),
]

# ---- §3.2: categories -----------------------------------------------------
# (text, genre, units, unit_labels)

_CATEGORIZE_EXAMPLES: list[tuple[str, str, list[str], list[str]]] = [
    # ex.1: near-universal moral value -> CO
    (
        "We should not encourage our children to smoke.",
        "editorial",
        ["We should not encourage our children to smoke"],
        ["CO"],
    ),
    # ex.2: established truth (CO) vs contested generalization (AS)
    (
        "Although smoking is harmful to health, many Germans smoke.",
        "editorial",
        ["Although smoking is harmful to health", "many Germans smoke"],
        ["CO", "AS"],
    ),
    # ex.3: vague attribution (AS) vs identifiable source (TE)
    (
        "Some scholars claim that all Germans are smokers. However, "
        "Professor Miller explained that this is nonsense.",
        "editorial",
        [
            "Some scholars claim that all Germans are smokers",
            "Professor Miller explained that this is nonsense",
        ],
        ["AS", "TE"],
    ),
    # ex.4: quantitative finding (ST) vs experiential generalization (AN)
    (
        "While the study indicates that only 20% of Germans are smokers, "
        "everyone in this field smokes.",
        "editorial",
        [
            "the study indicates that only 20% of Germans are smokers",
            "everyone in this field smokes",
        ],
        ["ST", "AN"],
    ),
    # ex.5: numbers from personal experience (AN), opinion (AS)
    (
        "Eighty percent of the people I know are smokers, I do not like that.",
        "editorial",
        [
            "Eighty percent of the people I know are smokers",
            "I do not like that",
        ],
        ["AN", "AS"],
    ),
    # ex.6: gratitude contributes nothing -> OT
    (
        "I thank the Germans for allowing me to smoke.",
        "editorial",
        ["I thank the Germans for allowing me to smoke"],
        ["OT"],
    ),
    # ex.7: sourced statistic -> multi-category ST/TE
    (
        "According to a report published by the World Health Organization, "
        "smoking rates among young people declined by 25% over the past "
        "decade.",
        "editorial",
        [
            "According to a report published by the World Health "
            "Organization, smoking rates among young people declined by 25% "
            "over the past decade"
        ],
        ["ST/TE"],
    ),
    # TE definition, inline example: restating the opponent in a debate = TE
    (
        "The first speaker from their team said that truth is subjective.",
        "debate",
        ["The first speaker from their team said that truth is subjective"],
        ["TE"],
    ),
]


def segment_demos() -> list[dspy.Example]:
    """SegmentUnits demos from §3.1 (all editorial-register examples)."""
    return [
        dspy.Example(text=text, genre="editorial", units=units).with_inputs(
            "text", "genre"
        )
        for text, units in _SEGMENT_EXAMPLES
    ]


def categorize_demos() -> list[dspy.Example]:
    """CategorizeUnits demos from §3.2 + the TE-definition debate example."""
    return [
        dspy.Example(
            text=text, genre=genre, units=units, unit_labels=unit_labels
        ).with_inputs("text", "genre", "units")
        for text, genre, units, unit_labels in _CATEGORIZE_EXAMPLES
    ]
