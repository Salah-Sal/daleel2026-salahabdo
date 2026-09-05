"""Task-level constants for Daleel 2026.

Label codes, definitions, and formats follow the official Codabench Task 1
page.
"""

LABELS = ("CO", "AS", "TE", "ST", "AN", "OT")

# Fixed before the v3 campaign.  This is deliberately a project constant,
# not a run-time tuning knob: a new architecture replaces a champion only
# after an identical-pool official-score improvement of at least two points.
V3_ADOPTION_DELTA = 0.02

LABEL_NAMES = {
    "CO": "Common ground",
    "AS": "Assumption",
    "TE": "Testimony",
    "ST": "Statistics",
    "AN": "Anecdote",
    "OT": "Other",
}

# Project glosses — the organizers have not published
# official Arabic label names; reconcile if they release Arabic guidelines.
LABEL_NAMES_AR = {
    "CO": "الأرضية المشتركة",
    "AS": "الافتراض",
    "TE": "الشهادة",
    "ST": "الإحصائيات",
    "AN": "المثال الشخصي",
    "OT": "أخرى",
}

# Genre values as they appear in the "type" field of the official examples.
GENRES = ("editorial", "debate")

# Training settings; each gets its own submission and leaderboard entry.
TRAINING_SETTINGS = ("editorial", "debate", "both")

# Both filenames confirmed by the organizers (2026-06-17).
TASK1_FILENAME = "task_1.jsonl"
TASK2_FILENAME = "task_2.jsonl"

# Team registered 2026-07-09. The registered team name contains a
# space; archives use the space-free form to keep
# <team_name>_<training_setting>.zip unambiguous.
TEAM_NAME_REGISTERED = "Salah Abdo"
TEAM_NAME = "SalahAbdo"
CODABENCH_USERNAME = "salahabdo"

OFFICIAL_REPO_URL = "https://github.com/Argmining/Daleel2026"

CODABENCH_COMPETITIONS = {
    ("task1", "open"): 16941,
    ("task1", "closed"): 16942,
    ("task2", "open"): 16938,
    ("task2", "closed"): 16936,
}
