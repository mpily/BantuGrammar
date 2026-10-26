"""Progress statistics: means with t-based 95% CIs, Wilson intervals, chrF."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .util import normalize_kitaita

# two-sided 95% Student-t critical values, df = 1..30
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_crit(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df <= 30:
        return _T975[df - 1]
    return 1.96 + 2.4 / df  # within ~0.005 of the exact value for df > 30


def mean_ci(values: list[float], low_n_threshold: int = 5) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0, "low_n": True}
    m = sum(values) / n
    if n < 2:
        return {"mean": m, "ci95": None, "n": n, "low_n": n < low_n_threshold}
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    half = t_crit(n - 1) * sd / math.sqrt(n)
    return {"mean": m, "ci95": (m - half, m + half), "n": n, "low_n": n < low_n_threshold}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_ci(k: int, n: int, low_n_threshold: int = 5) -> dict:
    return {"rate": (k / n) if n else None, "ci95": wilson(k, n), "k": k, "n": n,
            "low_n": n < low_n_threshold}


def _char_ngrams(s: str, n: int) -> Counter:
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hypothesis: str, reference: str, max_order: int = 6, beta: float = 2.0) -> float:
    """Sentence chrF (Popovic 2015; sacrebleu defaults: char order 6, beta 2,
    whitespace removed). Returns 0-100."""
    hyp = "".join((hypothesis or "").split())
    ref = "".join((reference or "").split())
    if not hyp and not ref:
        return 100.0
    if not hyp or not ref:
        return 0.0
    precs, recs = [], []
    for n in range(1, max_order + 1):
        h, r = _char_ngrams(hyp, n), _char_ngrams(ref, n)
        if not h or not r:
            continue
        match = sum((h & r).values())
        precs.append(match / sum(h.values()))
        recs.append(match / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


def compute_progress(translations: dict, feedback: list[dict], verified: dict,
                     low_n_threshold: int = 5) -> dict:
    stars = [fb["stars"] for fb in feedback if fb.get("stars") is not None]
    hist = Counter(stars)
    four_plus = sum(1 for s in stars if s >= 4)

    exact_k = exact_n = 0
    chrf_scores = []
    for fb in feedback:
        ref = fb.get("native_output")
        tr = translations.get(fb["translation_id"])
        if not ref or tr is None:
            continue
        hyp = tr.get("kitaita") or ""
        exact_n += 1
        exact_k += int(normalize_kitaita(hyp) == normalize_kitaita(ref))
        chrf_scores.append(chrf(normalize_kitaita(hyp), normalize_kitaita(ref)))

    by_day = defaultdict(lambda: {"translations": 0, "feedback": 0, "stars": []})
    for tr in translations.values():
        by_day[tr["created_at"][:10]]["translations"] += 1
    for fb in feedback:
        day = by_day[fb["created_at"][:10]]
        day["feedback"] += 1
        if fb.get("stars") is not None:
            day["stars"].append(fb["stars"])
    trend = []
    for date in sorted(by_day):
        d = by_day[date]
        trend.append({"date": date, "translations": d["translations"], "feedback": d["feedback"],
                      "rated": len(d["stars"]),
                      "mean_stars": (sum(d["stars"]) / len(d["stars"])) if d["stars"] else None,
                      "low_n": len(d["stars"]) < low_n_threshold})

    return {
        "counts": {"translations": len(translations), "feedback": len(feedback),
                   "unique_speakers": len({fb["speaker_id"] for fb in feedback}),
                   "verified_pairs": len(verified)},
        "mean_stars": mean_ci([float(s) for s in stars], low_n_threshold),
        "four_plus_rate": rate_ci(four_plus, len(stars), low_n_threshold),
        "star_histogram": {str(i): hist.get(i, 0) for i in range(1, 6)},
        "exact_match": rate_ci(exact_k, exact_n, low_n_threshold),
        "chrf": mean_ci(chrf_scores, low_n_threshold),
        "trend": trend,
        "low_n_threshold": low_n_threshold,
    }
