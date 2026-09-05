"""D9: gold-cleaning policy and the fixed local train/val split.

Cleaning policy (design item D9, audit-verified defects):

- Paragraph 964's ST/CO exist only as single-newline Task 2 spans; the
  generic rule "drop whitespace-only spans" removes exactly those, and the
  cleaned Task 1 gold is re-derived as the cleaned span-label set (Task 1
  gold equals the Task 2 span-label set for every official paragraph, so
  the derivation is the identity everywhere else — asserted below).
- Paragraphs 500/925 are suspected missed annotations and paragraph 964 is
  defective, so all three are forced to the TRAIN side: they never appear
  in val, where they would punish correct systems.

Cleaned gold feeds training inputs, optimizer metrics, and local val
scoring. Official leaderboard files are never touched.

The split itself is one fixed stratified 70/30 sample (genre x ST-presence
x CO-presence, so the 28 ST and 36 CO paragraphs distribute
proportionally), seeded and order-independent: the same records always
yield the same split. A final 5-fold pass for shortlisted programs is a
separate, later step (D9).
"""

import random

from .metrics import Span

SPLIT_SEED = 20260709
VAL_FRACTION = 0.30

# 964: newline-only phantom ST/CO spans; 500/925: suspected missed
# annotations. Never allowed into the val side.
FORCED_TRAIN_IDS = frozenset({500, 925, 964})


def clean_task2_gold(t2_records: list[dict]) -> dict[int, list[Span]]:
    """paragraph_id -> spans, with whitespace-only spans dropped."""
    gold = {}
    for r in t2_records:
        text = r["text"]
        gold[r["paragraph_id"]] = [
            Span(s["start_offset"], s["end_offset"], s["label"])
            for s in r["labels"]
            if text[s["start_offset"] : s["end_offset"]].strip()
        ]
    return gold


def clean_task1_gold(
    t1_records: list[dict], t2_records: list[dict]
) -> dict[int, set[str]]:
    """paragraph_id -> label set, re-derived from the cleaned Task 2 spans.

    Asserts the derivation matches the official Task 1 labels everywhere
    except the known-defective paragraphs, so silent data drift in either
    file fails loudly here.
    """
    spans = clean_task2_gold(t2_records)
    cleaned = {pid: {s.label for s in ss} for pid, ss in spans.items()}
    for r in t1_records:
        pid = r["paragraph_id"]
        if pid in FORCED_TRAIN_IDS:
            continue
        if cleaned.get(pid) != set(r["labels"]):
            raise AssertionError(
                f"paragraph {pid}: Task 1 labels {sorted(r['labels'])} != "
                f"cleaned span labels {sorted(cleaned.get(pid, ()))} — "
                "data changed; re-audit before trusting the cleaning policy"
            )
    return cleaned


def train_val_ids(t1_records: list[dict], t2_records: list[dict]) -> tuple[list[int], list[int]]:
    """The fixed local split: (train_ids, val_ids), each sorted.

    Stratified by (genre, has_ST, has_CO) over the CLEANED gold;
    FORCED_TRAIN_IDS are held out of the val pool before sampling.
    """
    gold = clean_task1_gold(t1_records, t2_records)
    genre = {r["paragraph_id"]: r["type"] for r in t1_records}

    strata: dict[tuple, list[int]] = {}
    forced_train = []
    for pid in sorted(gold):
        if pid in FORCED_TRAIN_IDS:
            forced_train.append(pid)
            continue
        key = (genre[pid], "ST" in gold[pid], "CO" in gold[pid])
        strata.setdefault(key, []).append(pid)

    rng = random.Random(SPLIT_SEED)
    train, val = list(forced_train), []
    for key in sorted(strata):
        ids = strata[key]  # already sorted (built from sorted(gold))
        rng.shuffle(ids)
        n_val = round(VAL_FRACTION * len(ids))
        val += ids[:n_val]
        train += ids[n_val:]
    return sorted(train), sorted(val)


OPT_SPLIT_SEED = 20260710
OPT_VAL_FRACTION = 0.35


def optimizer_split(t1_records: list[dict], t2_records: list[dict]) -> tuple[list[int], list[int]]:
    """Split the TRAIN side again for optimizer-internal use: (opt_train, opt_val).

    Compiles must never see the frozen D9 val (it is the selection set for
    every real decision), so optimizers bootstrap demos from opt_train and
    score candidates on opt_val — both carved out of the 430 train-side
    paragraphs with the same (genre, ST, CO) stratification. Deterministic;
    FORCED_TRAIN_IDS stay on the opt_train side (500/925 as opt-val items
    would punish correct systems, and 964 is defective).
    """
    train_ids, _ = train_val_ids(t1_records, t2_records)
    gold = clean_task1_gold(t1_records, t2_records)
    genre = {r["paragraph_id"]: r["type"] for r in t1_records}

    strata: dict[tuple, list[int]] = {}
    opt_train = []
    for pid in train_ids:  # already sorted
        if pid in FORCED_TRAIN_IDS:
            opt_train.append(pid)
            continue
        key = (genre[pid], "ST" in gold[pid], "CO" in gold[pid])
        strata.setdefault(key, []).append(pid)

    rng = random.Random(OPT_SPLIT_SEED)
    opt_val = []
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        n_val = round(OPT_VAL_FRACTION * len(ids))
        opt_val += ids[:n_val]
        opt_train += ids[n_val:]
    return sorted(opt_train), sorted(opt_val)


def filter_setting(records: list[dict], setting: str) -> list[dict]:
    """Restrict records to one training setting's genre ('both' = no-op)."""
    if setting == "both":
        return list(records)
    if setting not in ("editorial", "debate"):
        raise ValueError(f"unknown training setting: {setting!r}")
    return [r for r in records if r["type"] == setting]
