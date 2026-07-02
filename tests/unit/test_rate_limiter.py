from observability.dashboard.services.rate_limiter import RateLimiter, owner_llm_allowed, upload_allowed


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_upload_limit_three_per_session_per_hour() -> None:
    clock = FakeClock()
    limiter = RateLimiter(clock=clock)

    assert upload_allowed(limiter, "s1") is True
    assert upload_allowed(limiter, "s1") is True
    assert upload_allowed(limiter, "s1") is True
    assert upload_allowed(limiter, "s1") is False
    assert upload_allowed(limiter, "s2") is True

    clock.now += 3601
    assert upload_allowed(limiter, "s1") is True


def test_owner_llm_hourly_limit_shared_across_sessions() -> None:
    clock = FakeClock()
    limiter = RateLimiter(clock=clock)

    for _ in range(30):
        assert owner_llm_allowed(limiter) is True
    assert owner_llm_allowed(limiter) is False

    clock.now += 3601
    assert owner_llm_allowed(limiter) is True


def test_owner_llm_daily_limit_shared_across_sessions() -> None:
    clock = FakeClock()
    limiter = RateLimiter(clock=clock)

    for hour in range(6):
        for _ in range(30):
            assert owner_llm_allowed(limiter) is True
        clock.now += 3601
    for _ in range(20):
        assert owner_llm_allowed(limiter) is True
    assert limiter._windows["owner-llm:day"][1] == 200
    assert owner_llm_allowed(limiter) is False
