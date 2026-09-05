"""Pre-flight alignment study for D8 (metric design): which per-example
proxy best RANKS predictors the way the official corpus-level scorers do?

DSPy optimizers maximize a mean of per-example scores, but the official
Task 1 metric (label-macro F1) is not per-example decomposable, and the
official Task 2 metric normalizes by corpus-wide span counts. Before
trusting any proxy inside a compile loop we measure, on the real training
gold, how well each candidate proxy's mean tracks the official score
across a family of simulated predictors (gold degraded by parameterized
error processes: uniform noise, rare-label killers/spammers, CO->AS
confusion, boundary jitter, over-segmentation...).

Also reports the sampling noise of the official macro-F1 on val-sized
subsets, which sets the minimum score difference worth acting on during
candidate selection.

Run from shared-task/:  uv run python scripts/metric_alignment_check.py
No LLM calls; reads gold from the official repo clone; seeded.
"""

import math
import random
import sys
from collections import Counter

from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, "src")
from daleel.constants import LABELS
from daleel.data import TRAIN_TASK1, TRAIN_TASK2, load_records, task1_labels, task2_spans
from daleel.metrics import Span, span_partial_f1, task1_macro_f1

SEED = 20260709
N_PREDICTORS_T1 = 240
N_PREDICTORS_T2 = 120
VAL_FRACTION = 0.3


# ---------------------------------------------------------------- Task 1

def per_example_f1(g: set, p: set, weights=None) -> float:
    if not g and not p:
        return 1.0
    w = weights or dict.fromkeys(LABELS, 1.0)
    inter = sum(w[l] for l in g & p)
    denom = sum(w[l] for l in g) + sum(w[l] for l in p)
    return 2 * inter / denom if denom else 0.0


def jaccard(g: set, p: set) -> float:
    if not g and not p:
        return 1.0
    return len(g & p) / len(g | p)


def hamming_acc(g: set, p: set, weights=None) -> float:
    w = weights or dict.fromkeys(LABELS, 1.0)
    total = sum(w.values())
    return sum(w[l] for l in LABELS if (l in g) == (l in p)) / total


def make_weights(gold: dict, power: float) -> dict:
    freq = Counter(l for labels in gold.values() for l in labels)
    raw = {l: (1.0 / max(freq[l], 1)) ** power for l in LABELS}
    z = sum(raw.values())
    return {l: v * len(LABELS) / z for l, v in raw.items()}


def simulate_t1_predictor(gold: dict, rng: random.Random) -> tuple[dict, str]:
    """One predictor = per-label (p_drop, p_add) rates from a random regime."""
    regime = rng.choice(["uniform", "kill_rare", "spam_rare", "confuse", "mixed"])
    base_drop = rng.uniform(0.0, 0.5)
    base_add = rng.uniform(0.0, 0.15)
    p_drop = dict.fromkeys(LABELS, base_drop)
    p_add = dict.fromkeys(LABELS, base_add)
    if regime == "kill_rare":
        for l in ("ST", "CO"):
            p_drop[l] = rng.uniform(0.5, 1.0)
    elif regime == "spam_rare":
        for l in ("ST", "CO"):
            p_add[l] = rng.uniform(0.1, 0.5)
    elif regime == "mixed":
        for l in LABELS:
            p_drop[l] = rng.uniform(0.0, 0.6)
            p_add[l] = rng.uniform(0.0, 0.2)
    confuse = rng.uniform(0.3, 0.9) if regime == "confuse" else 0.0

    pred = {}
    for i, g in gold.items():
        # iterate in fixed LABELS order: set iteration order varies per
        # process (hash randomization) and would break seeded reproducibility
        p = {l for l in LABELS if l in g and rng.random() >= p_drop[l]}
        p |= {l for l in LABELS if l not in g and rng.random() < p_add[l]}
        if confuse and "CO" in p and rng.random() < confuse:
            p.discard("CO")
            p.add("AS")
        pred[i] = p
    return pred, regime


def study_task1(gold: dict, rng: random.Random) -> None:
    w_inv = make_weights(gold, 1.0)
    w_sqrt = make_weights(gold, 0.5)
    proxies = {
        "exact match": lambda g, p: float(g == p),
        "set F1 (unweighted)": per_example_f1,
        "Jaccard": jaccard,
        "Hamming accuracy": hamming_acc,
        "F1, 1/freq weights": lambda g, p: per_example_f1(g, p, w_inv),
        "F1, 1/sqrt(freq) weights": lambda g, p: per_example_f1(g, p, w_sqrt),
        "Hamming, 1/freq weights": lambda g, p: hamming_acc(g, p, w_inv),
    }

    official, regimes, proxy_means = [], [], {name: [] for name in proxies}
    for _ in range(N_PREDICTORS_T1):
        pred, regime = simulate_t1_predictor(gold, rng)
        regimes.append(regime)
        official.append(task1_macro_f1(gold, pred)["macro_f1"])
        for name, fn in proxies.items():
            proxy_means[name].append(
                sum(fn(set(gold[i]), pred[i]) for i in gold) / len(gold)
            )

    def flip_rate(means, pair_filter=None):
        flips = total = 0
        for a in range(len(official)):
            for b in range(a + 1, len(official)):
                if abs(official[a] - official[b]) <= 0.01:
                    continue
                if pair_filter and not pair_filter(a, b):
                    continue
                total += 1
                flips += (official[a] - official[b]) * (means[a] - means[b]) < 0
        return 100 * flips / total if total else float("nan")

    rare = {"kill_rare", "spam_rare", "confuse"}

    print(f"\n== Task 1: {N_PREDICTORS_T1} simulated predictors over {len(gold)} paragraphs ==")
    print(f"official macro-F1 range: [{min(official):.3f}, {max(official):.3f}]")
    print("flip% = ordering disagreements with official among pairs >0.01 apart;")
    print("rare-pairs = pairs where at least one predictor is kill_rare/spam_rare/confuse")
    print(f"{'proxy (per-example mean)':<28} {'spearman':>9} {'kendall':>9} {'flip%':>7} {'rare-pairs':>10}")
    for name, means in proxy_means.items():
        rho = spearmanr(means, official).statistic
        tau = kendalltau(means, official).statistic
        rare_flip = flip_rate(means, lambda a, b: regimes[a] in rare or regimes[b] in rare)
        print(f"{name:<28} {rho:>9.3f} {tau:>9.3f} {flip_rate(means):>6.1f}% {rare_flip:>9.1f}%")


def study_val_noise(gold: dict, rng: random.Random) -> None:
    """SD of official macro-F1 across val-sized subsamples, at three
    predictor strengths: sets the smallest difference worth acting on."""
    ids = list(gold)
    n_val = round(len(ids) * VAL_FRACTION)
    print(f"\n== Task 1 selection noise: official macro-F1 over random {n_val}-paragraph subsets ==")
    print(f"{'predictor strength':<24} {'full-set F1':>12} {'subset SD':>10} {'95% width':>10}")
    for label, (pd_, pa) in {"good (drop .1, add .02)": (0.1, 0.02),
                             "mid (drop .25, add .05)": (0.25, 0.05),
                             "weak (drop .4, add .10)": (0.4, 0.10)}.items():
        pred = {i: ({l for l in LABELS if l in g and rng.random() >= pd_}
                    | {l for l in LABELS if l not in g and rng.random() < pa})
                for i, g in gold.items()}
        full = task1_macro_f1(gold, pred)["macro_f1"]
        scores = []
        for _ in range(400):
            sub = rng.sample(ids, n_val)
            scores.append(task1_macro_f1({i: gold[i] for i in sub},
                                         {i: pred[i] for i in sub})["macro_f1"])
        mu = sum(scores) / len(scores)
        sd = math.sqrt(sum((s - mu) ** 2 for s in scores) / (len(scores) - 1))
        lo, hi = sorted(scores)[9], sorted(scores)[-10]
        print(f"{label:<24} {full:>12.3f} {sd:>10.3f} {hi - lo:>10.3f}")


# ---------------------------------------------------------------- Task 2

def simulate_t2_predictor(gold: dict, texts: dict, rng: random.Random) -> dict:
    p_drop = rng.uniform(0.0, 0.4)
    p_flip = rng.uniform(0.0, 0.3)
    jitter = rng.choice([0, 0, 5, 15, 40])
    p_split = rng.uniform(0.0, 0.5) if rng.random() < 0.5 else 0.0
    kill_rare = rng.random() < 0.25

    pred = {}
    for i, spans in gold.items():
        out = []
        for s in spans:
            if rng.random() < p_drop or (kill_rare and s.label in ("ST", "CO")):
                continue
            label = s.label
            if rng.random() < p_flip:
                label = rng.choice([l for l in LABELS if l != label])
            start = max(0, s.start + rng.randint(-jitter, jitter)) if jitter else s.start
            end = min(len(texts[i]), s.end + rng.randint(-jitter, jitter)) if jitter else s.end
            if end <= start:
                continue
            if p_split and rng.random() < p_split and end - start > 10:
                mid = (start + end) // 2
                out += [Span(start, mid, label), Span(mid + 1, end, label)]
            else:
                out.append(Span(start, end, label))
        pred[i] = out
    return pred


def char_label_f1(g_spans, p_spans, weights=None) -> float:
    w = weights or dict.fromkeys(LABELS, 1.0)
    g = {(c, s.label) for s in g_spans for c in range(s.start, s.end)}
    p = {(c, s.label) for s in p_spans for c in range(s.start, s.end)}
    if not g and not p:
        return 1.0
    inter = sum(w[l] for _, l in g & p)
    denom = sum(w[l] for _, l in g) + sum(w[l] for _, l in p)
    return 2 * inter / denom if denom else 0.0


def study_task2(gold: dict, texts: dict, rng: random.Random) -> None:
    span_freq = Counter(s.label for spans in gold.values() for s in spans)
    raw = {l: 1.0 / max(span_freq[l], 1) for l in LABELS}
    z = sum(raw.values())
    w_inv = {l: v * len(LABELS) / z for l, v in raw.items()}

    def para_official(g_spans, p_spans):
        if not g_spans and not p_spans:
            return 1.0
        return span_partial_f1({0: g_spans}, {0: p_spans})["f1"]

    proxies = {
        "per-para pair-sum F1": para_official,
        "char-label F1": char_label_f1,
        "char-label F1, 1/freq wts": lambda g, p: char_label_f1(g, p, w_inv),
    }

    official, proxy_means, weighted_means = [], {n: [] for n in proxies}, []
    for _ in range(N_PREDICTORS_T2):
        pred = simulate_t2_predictor(gold, texts, rng)
        official.append(span_partial_f1(gold, pred)["f1"])
        for name, fn in proxies.items():
            proxy_means[name].append(sum(fn(gold[i], pred[i]) for i in gold) / len(gold))
        num = sum(para_official(gold[i], pred[i]) * len(gold[i]) for i in gold)
        weighted_means.append(num / sum(len(gold[i]) for i in gold))
    proxy_means["pair-sum F1, span-count wtd"] = weighted_means

    print(f"\n== Task 2: {N_PREDICTORS_T2} simulated predictors over {len(gold)} paragraphs ==")
    print(f"official pair-sum F1 range: [{min(official):.3f}, {max(official):.3f}]")
    print(f"{'proxy (per-example mean)':<28} {'spearman':>9} {'kendall':>9}")
    for name, means in proxy_means.items():
        rho = spearmanr(means, official).statistic
        tau = kendalltau(means, official).statistic
        print(f"{name:<28} {rho:>9.3f} {tau:>9.3f}")


def main() -> None:
    rng = random.Random(SEED)
    gold_t1 = {i: set(v) for i, v in task1_labels(load_records(TRAIN_TASK1)).items()}
    t2_records = load_records(TRAIN_TASK2)
    gold_t2 = task2_spans(t2_records)
    texts = {r["paragraph_id"]: r["text"] for r in t2_records}
    study_task1(gold_t1, rng)
    study_val_noise(gold_t1, rng)
    study_task2(gold_t2, texts, rng)


if __name__ == "__main__":
    main()
