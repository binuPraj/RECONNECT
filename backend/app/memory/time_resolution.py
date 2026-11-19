"""Deterministic interpretation of conversational time phrases.

This module is deliberately shared by memory writing and future retrieval so
"last Monday" can never mean one thing when stored and another when queried.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Kathmandu")
WEEKDAYS = {name: index for index, name in enumerate(
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
)}
PARTS = ("morning", "afternoon", "evening", "night")
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7}


def _day(value: datetime) -> datetime:
    return value.astimezone(TZ).replace(hour=0, minute=0, second=0, microsecond=0)


def _window(start: datetime, end: datetime, part: str | None, granularity: str) -> dict:
    return {"start": start.isoformat(), "end": end.isoformat(), "part": part,
            "granularity": granularity}


def resolve_time(phrase: str, anchor: datetime | str, tense: str = "any") -> dict | None:
    """Resolve a supported phrase against its source-event timestamp.

    Returns an inclusive-start/exclusive-end window, or ``None`` for vague or
    unsupported language.  No LLM calendar arithmetic is used.
    """
    anchor_dt = datetime.fromisoformat(anchor) if isinstance(anchor, str) else anchor
    today = _day(anchor_dt)
    text = (phrase or "").strip().casefold().replace("tonight", "today night")
    if not text or text in {"now", "right now", "soon", "recently", "a long time"}:
        return None
    part = next((item for item in PARTS if re.search(rf"\b{item}\b", text)), None)
    if part:
        text = re.sub(rf"\b{part}\b", "", text).strip(" ,")

    def day_window(day: datetime) -> dict:
        return _window(day, day + timedelta(days=1), part, "day")

    simple = {
        "today": 0, "this": 0, "yesterday": -1, "tomorrow": 1,
        "day before yesterday": -2, "day after tomorrow": 2,
    }
    if text in simple:
        return day_window(today + timedelta(days=simple[text]))

    weekday = re.fullmatch(r"(?:on\s+)?(?:(this|next|last)\s+)?(" + "|".join(WEEKDAYS) + r")", text)
    if weekday:
        qualifier, weekday_name = weekday.groups()
        target = WEEKDAYS[weekday_name]
        forward, backward = (target - today.weekday()) % 7, (today.weekday() - target) % 7
        if qualifier == "last":
            day = today - timedelta(days=backward or 7)
        elif qualifier == "next":
            day = today + timedelta(days=forward or 7)
        elif tense == "past":
            day = today - timedelta(days=backward)
        else:
            day = today + timedelta(days=forward)
        return day_window(day)

    period = re.fullmatch(r"(this|next|last) (week|month)", text)
    if period:
        qualifier, unit = period.groups()
        shift = {"last": -1, "this": 0, "next": 1}[qualifier]
        if unit == "week":
            start = today - timedelta(days=today.weekday()) + timedelta(weeks=shift)
            return _window(start, start + timedelta(weeks=1), None, "week")
        month_number = today.year * 12 + today.month - 1 + shift
        year, month0 = divmod(month_number, 12)
        start = today.replace(year=year, month=month0 + 1, day=1)
        next_number = month_number + 1
        next_year, next_month0 = divmod(next_number, 12)
        return _window(start, start.replace(year=next_year, month=next_month0 + 1), None, "month")

    relative = re.fullmatch(r"(?:in )?(\d+|" + "|".join(NUMBERS) + r") (days?|weeks?)(?: ago)?", text)
    if relative:
        amount_text, unit = relative.groups()
        amount = int(amount_text) if amount_text.isdigit() else NUMBERS[amount_text]
        sign = -1 if text.endswith(" ago") else 1
        offset = amount * (7 if unit.startswith("week") else 1) * sign
        day = today + timedelta(days=offset)
        if unit.startswith("week"):
            start = day - timedelta(days=day.weekday())
            return _window(start, start + timedelta(weeks=1), None, "week")
        return day_window(day)

    # Explicit calendar dates are deterministic too.  ``dateutil`` is already
    # part of this backend; use the event's local date as the missing-year
    # default instead of allowing the model to invent calendar arithmetic.
    try:
        from dateutil import parser as date_parser
        parsed = date_parser.parse(text, default=today.replace(tzinfo=None))
        parsed = parsed.replace(tzinfo=TZ)
        return day_window(_day(parsed))
    except (OverflowError, TypeError, ValueError):
        pass
    return None


def direction_ok(window: dict, anchor: datetime | str, tense: str) -> bool:
    anchor_dt = datetime.fromisoformat(anchor) if isinstance(anchor, str) else anchor
    if tense == "future":
        return datetime.fromisoformat(window["end"]) > anchor_dt
    if tense == "past":
        return datetime.fromisoformat(window["start"]) < anchor_dt
    return True
