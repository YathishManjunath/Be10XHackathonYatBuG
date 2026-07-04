"""
time_utils.py
--------------
Small, dependency-free helpers for parsing the free-text time ranges found in
the synthetic dataset (data.py) and reasoning about interval overlap. Shared
by data_models.py and scheduler_engine.py so time parsing logic lives in
exactly one place.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])")


def parse_time_to_minutes(time_str: str) -> int:
    """Convert 'H:MM AM/PM' into minutes-since-midnight."""
    match = _TIME_RE.search(time_str.strip())
    if not match:
        raise ValueError(f"Cannot parse time string: {time_str!r}")
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3).upper()
    if meridiem == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour * 60 + minute


def minutes_to_time_str(total_minutes: int) -> str:
    """Convert minutes-since-midnight back into a clean 'H:MM AM/PM' string."""
    hour, minute = divmod(total_minutes, 60)
    hour %= 24
    period = "AM" if hour < 12 else "PM"
    hour12 = hour % 12
    hour12 = 12 if hour12 == 0 else hour12
    return f"{hour12}:{minute:02d} {period}"


def extract_clock_time(text: str) -> Optional[str]:
    """Pull the first 'H:MM AM/PM' substring out of an arbitrary free-text note."""
    if not text:
        return None
    match = _TIME_RE.search(text)
    if not match:
        return None
    hour, minute, meridiem = match.group(1), match.group(2), match.group(3).upper()
    return f"{int(hour)}:{minute} {meridiem}"


def parse_time_range(text: str) -> Tuple[str, str]:
    """Parse 'H:MM AM - H:MM PM' into (start, end) clock-time strings."""
    parts = [p.strip() for p in str(text).split("-")]
    if len(parts) != 2:
        raise ValueError(f"Cannot parse time range: {text!r}")
    return parts[0], parts[1]


def duration_minutes(start_str: str, end_str: str) -> int:
    return parse_time_to_minutes(end_str) - parse_time_to_minutes(start_str)


def interval_for(start_str: str, duration_min: int) -> Tuple[int, int]:
    start = parse_time_to_minutes(start_str)
    return start, start + duration_min


def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def contains(outer_start: int, outer_end: int, inner_start: int, inner_end: int) -> bool:
    return outer_start <= inner_start and inner_end <= outer_end


def parse_av_list(text: str) -> List[str]:
    if not text:
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def extract_capacity(text: str, default: int = 1) -> int:
    """Pull the first integer out of a free-text resource description."""
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else default


def generate_time_slots(start: str = "9:00 AM", end: str = "5:00 PM", step_minutes: int = 30) -> List[str]:
    start_m, end_m = parse_time_to_minutes(start), parse_time_to_minutes(end)
    slots, m = [], start_m
    while m <= end_m:
        slots.append(minutes_to_time_str(m))
        m += step_minutes
    return slots
