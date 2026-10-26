import math

from api.stats import chrf, compute_progress, mean_ci, rate_ci, wilson


def test_mean_ci_matches_t_interval():
    m = mean_ci([1, 2, 3, 4, 5], low_n_threshold=5)
    assert m["mean"] == 3 and m["n"] == 5 and m["low_n"] is False
    half = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert math.isclose(m["ci95"][0], 3 - half) and math.isclose(m["ci95"][1], 3 + half)


def test_low_n_flag_and_empty():
    assert mean_ci([4, 5])["low_n"] is True
    assert mean_ci([]) == {"mean": None, "ci95": None, "n": 0, "low_n": True}
    assert mean_ci([4])["ci95"] is None
    assert rate_ci(0, 0)["rate"] is None and rate_ci(0, 0)["low_n"] is True
    assert rate_ci(3, 4)["low_n"] is True and rate_ci(3, 5)["low_n"] is False


def test_wilson_known_values():
    lo, hi = wilson(8, 10)
    assert math.isclose(lo, 0.4902, abs_tol=1e-3) and math.isclose(hi, 0.9433, abs_tol=1e-3)
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0


def test_chrf_bounds():
    assert chrf("choka yacha", "choka yacha") == 100.0
    assert chrf("abc", "xyz") == 0.0
    assert 0 < chrf("choka icha", "choka yacha") < 100


def _tr(i, kit, day="2026-10-27"):
    return {"id": f"tr_{i}", "kitaita": kit, "created_at": f"{day}T10:00:00+00:00"}


def _fb(tid, spk, stars=None, out=None, day="2026-10-27"):
    return {"translation_id": tid, "speaker_id": spk, "stars": stars, "native_output": out,
            "created_at": f"{day}T11:00:00+00:00"}


def test_compute_progress():
    trs = {t["id"]: t for t in [_tr(1, "Choka yacha."), _tr(2, "Mbwa"), _tr(3, None, "2026-10-28")]}
    fbs = [_fb("tr_1", "a", 5, "choka yacha"), _fb("tr_1", "b", 4), _fb("tr_2", "a", 2, "Koshi"),
           _fb("tr_2", "c", 1, day="2026-10-28")]
    p = compute_progress(trs, fbs, {"tr_1": {}}, low_n_threshold=5)
    assert p["counts"] == {"translations": 3, "feedback": 4, "unique_speakers": 3, "verified_pairs": 1}
    assert p["star_histogram"] == {"1": 1, "2": 1, "3": 0, "4": 1, "5": 1}
    assert p["mean_stars"]["mean"] == 3.0 and p["mean_stars"]["n"] == 4 and p["mean_stars"]["low_n"]
    assert p["four_plus_rate"]["k"] == 2 and p["four_plus_rate"]["n"] == 4
    assert p["exact_match"]["k"] == 1 and p["exact_match"]["n"] == 2
    assert p["chrf"]["n"] == 2
    assert [d["date"] for d in p["trend"]] == ["2026-10-27", "2026-10-28"]
    assert p["trend"][0]["translations"] == 2 and p["trend"][1]["feedback"] == 1


def test_progress_endpoint_always_shows_n(client):
    body = client.get("/api/progress").json()
    assert body["api_version"] == "1"
    for key in ("mean_stars", "four_plus_rate", "exact_match", "chrf"):
        assert body[key]["n"] == 0 and body[key]["low_n"] is True
    tid = client.post("/api/translate", json={"text": "Nyoka aja"}).json()["id"]
    for i, s in enumerate([5, 4, 4, 3, 5]):
        client.post("/api/feedback", json={"translation_id": tid, "stars": s, "speaker_id": f"s{i}"})
    body = client.get("/api/progress").json()
    assert body["mean_stars"]["n"] == 5 and body["mean_stars"]["low_n"] is False
    assert body["counts"]["unique_speakers"] == 5
