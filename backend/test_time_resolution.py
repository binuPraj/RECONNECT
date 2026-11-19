from datetime import datetime
from zoneinfo import ZoneInfo

from app.memory.time_resolution import resolve_time


ANCHOR = datetime(2026, 9, 16, 14, 0, tzinfo=ZoneInfo("Asia/Kathmandu"))


def test_relative_weekday_windows_are_stable():
    assert resolve_time("last Monday", ANCHOR, "past")["start"].startswith("2026-09-14")
    assert resolve_time("next Monday", ANCHOR, "future")["start"].startswith("2026-09-21")
    evening = resolve_time("Monday evening", ANCHOR, "future")
    assert evening["start"].startswith("2026-09-21") and evening["part"] == "evening"


def test_relative_and_explicit_dates_share_the_same_resolver():
    assert resolve_time("tomorrow", ANCHOR, "future")["start"].startswith("2026-09-17")
    assert resolve_time("next week", ANCHOR, "future")["start"].startswith("2026-09-21")
    assert resolve_time("in two days", ANCHOR, "future")["start"].startswith("2026-09-18")
    assert resolve_time("3 days ago", ANCHOR, "past")["start"].startswith("2026-09-13")
    assert resolve_time("October 4, 2026", ANCHOR, "future")["start"].startswith("2026-10-04")
    assert resolve_time("soon", ANCHOR, "future") is None
