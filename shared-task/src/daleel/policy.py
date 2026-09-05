"""Canonical label policy: one phrasing for definitions and corpus rules.

Both the DSPy seed instructions (dspy_programs) and the metric feedback
(dspy_metrics) quote from here. Keeping a single source is a hard
requirement from the D8 analysis: GEPA rewrites instructions against the
feedback text, and two divergent phrasings of the same label make the
optimization oscillate ("feedback drift").

Content policy (D4 seed (b)): the official label definitions merged with
the operational corrections from the training-data quality audit —
English meta-language, Arabic label glosses.
"""

LABEL_POLICY = {
    "CO": (
        "CO — Common ground (الأرضية المشتركة): premises presented as already "
        "shared or accepted by everyone — universally accepted truths, "
        "definitions, proverbs, and settled facts, including settled legal or "
        "procedural background. NOT the author's own contested stance (that "
        "is AS)."
    ),
    "AS": (
        "AS — Assumption (الافتراض): the author's or speaker's own claims, "
        "judgments, evaluations, predictions, or proposals, asserted without "
        "external backing. Rhetorical questions and emotional appeals count "
        "as AS."
    ),
    "TE": (
        "TE — Testimony (الشهادة): statements attributed to an identifiable "
        "source — direct quotes, reported speech, expert or official "
        "statements — including a debater reporting the opposing team's "
        "words."
    ),
    "ST": (
        "ST — Statistics (الإحصائيات): quantitative evidence — numbers, "
        "percentages, survey or study results — even when embedded inside "
        "reported speech."
    ),
    "AN": (
        "AN — Anecdote (المثال الشخصي): a concrete instance — personal "
        "experience, a specific incident, a historical example or case "
        "story."
    ),
    "OT": (
        "OT — Other (أخرى): discourse management without argumentative "
        "content — greetings, procedural debate speech (introducing "
        "speakers, announcing points), transitions and meta-comments. "
        "Common in debate transcripts."
    ),
}

# Short "why this label is probably wrong here" lines for metric feedback on
# spurious predictions — the known annotator-confusion axes from the audit.
SPURIOUS_HINTS = {
    "CO": (
        "CO is reserved for premises the audience already accepts; a claim "
        "the author is arguing FOR is AS."
    ),
    "AS": (
        "AS is for the author's own assertions; attributed statements are "
        "TE, and accepted/settled premises are CO."
    ),
    "TE": (
        "TE needs an identifiable source being quoted or reported; unsourced "
        "claims are AS."
    ),
    "ST": (
        "ST needs explicit quantities (numbers, percentages, study "
        "results); qualitative claims about evidence are not ST."
    ),
    "AN": (
        "AN needs a concrete specific instance or experience; general "
        "claims about the world are AS."
    ),
    "OT": (
        "OT is only for non-argumentative discourse management; any content "
        "that supports or attacks a position belongs to another label."
    ),
}

TASK_CONTEXT = (
    "The text is one Arabic paragraph of argumentative discourse: either a "
    "newspaper editorial (edited MSA prose) or a transcribed WSDC-style "
    "debate speech (spoken register, procedural passages, occasional "
    "Latin-script code-switching, speaker markers like "
    "*المتحدث الأول موالاة: (ذكر)/*). Several ADU types usually co-occur in "
    "one paragraph; some paragraphs contain none at all."
)
