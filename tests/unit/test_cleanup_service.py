import json
import time
from pathlib import Path

from observability.dashboard.services.cleanup_service import CleanupService


class FakeClock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_session_dir(root: Path, name: str, expires_at: float) -> Path:
    target = root / name
    target.mkdir(parents=True)
    (target / "session.json").write_text(json.dumps({"session_id": name, "mode": "guest", "created_at": expires_at - 3600, "expires_at": expires_at}), encoding="utf-8")
    return target


def test_cleanup_removes_expired_keeps_fresh(tmp_path) -> None:
    clock = FakeClock()
    expired = make_session_dir(tmp_path, "old", expires_at=clock.now - 1)
    fresh = make_session_dir(tmp_path, "new", expires_at=clock.now + 3600)

    removed = CleanupService(root=tmp_path, clock=clock).cleanup_expired()

    assert removed == 1
    assert not expired.exists()
    assert fresh.exists()


def test_cleanup_removes_stale_markerless_directories(tmp_path) -> None:
    clock = FakeClock()
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "x.txt").write_text("x", encoding="utf-8")
    old_time = clock.now - 7200
    import os

    os.utime(stale, (old_time, old_time))

    assert CleanupService(root=tmp_path, clock=clock).cleanup_expired() == 1
    assert not stale.exists()


def test_cleanup_never_removes_outside_root(tmp_path) -> None:
    clock = FakeClock()
    root = tmp_path / "sessions"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "evil"
    link.symlink_to(outside, target_is_directory=True)

    removed = CleanupService(root=root, clock=clock).cleanup_expired()

    assert removed == 0
    assert outside.exists()
    assert link.exists()


def test_cleanup_missing_root_is_noop(tmp_path) -> None:
    assert CleanupService(root=tmp_path / "missing").cleanup_expired() == 0


def test_cleanup_thread_runs_as_daemon(tmp_path) -> None:
    service = CleanupService(root=tmp_path)
    service.ensure_cleanup_thread(interval_seconds=3600)
    service.ensure_cleanup_thread(interval_seconds=3600)

    threads = [thread for thread in __import__("threading").enumerate() if thread.name == "synapserag-cleanup"]
    assert len(threads) == 1
    assert threads[0].daemon
