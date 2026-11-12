"""haiku_mt guard logic (no API calls): not configured, length limit, daily cap, per-visitor limit."""
import haiku_mt as H


def test_not_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = H.translate("Habari")
    assert r["kitaita"] is None and "not configured" in r["status"]


def test_length_cap_and_daily_cap(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert "too long" in H.translate("a" * (H.MAX_CHARS + 1))["status"]
    monkeypatch.setattr(H, "DAILY_CAP_USD", 0.0)
    assert "daily limit" in H.translate("Habari")["status"]


def test_visitor_limit(monkeypatch):
    monkeypatch.setattr(H, "PER_VISITOR_PER_HOUR", 2)
    v = "visitor-test"
    assert H._visitor_ok(v) and H._visitor_ok(v) and not H._visitor_ok(v)
    assert H._visitor_ok("someone-else")


def test_budget_client_books_cost(monkeypatch):
    from types import SimpleNamespace
    u = SimpleNamespace(input_tokens=1_000_000, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    inner = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(usage=u)))
    monkeypatch.setattr(H, "DAILY_CAP_USD", 0.05)
    H._spent.update(day=H._today(), usd=0.0)
    c = H._GuardedClient(inner, "claude-haiku-5-5")
    c.create()                                   # $0.10 booked: cap overshot by one request at most
    assert round(c.usd, 4) == 0.10
    try:
        c.create()
        assert False, "second request should be refused"
    except H.BudgetExceeded:
        pass
    H._spent.update(day=H._today(), usd=0.0)


def test_precheck_counts_one_visit(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(H, "DAILY_CAP_USD", 10.0)
    monkeypatch.setattr(H, "PER_VISITOR_PER_HOUR", 1)
    H._visits.pop("one-visit", None)
    assert H.precheck("Habari", "one-visit") is None          # arms C and E share this one visit
    assert "per hour" in H.precheck("Habari", "one-visit")
