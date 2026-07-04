"""
data_models.py
---------------
Typed data structures for EventSync AI. The synthetic ("dummy") dataset now
lives in `data.py` (`get_messy_event_data()` / `get_venue_constraints()`) --
this module is the ONLY place that translates that raw, free-text data into
the structured `Room` / `Session` objects the rest of the engine (and the
LLM tool-calling agent) operates on.

`load_dummy_data()` always rebuilds everything from scratch, so the synthetic
template can never be mutated by a running session (data isolation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from data import get_messy_event_data, get_venue_constraints
from time_utils import (
    duration_minutes,
    extract_capacity,
    extract_clock_time,
    generate_time_slots,
    parse_av_list,
    parse_time_range,
)

# Half-hour granularity so we can land sessions on slots like "11:30 AM"
# straight out of the raw dataset, not just on-the-hour times.
TIME_SLOTS: List[str] = generate_time_slots("9:00 AM", "5:00 PM", step_minutes=30)

DEFAULT_ROOM_CAPACITY = 100
_ROOM_CAPACITY_HINTS: Dict[str, int] = {
    "Main Ballroom": 500,
    "Auditorium B": 300,
    "Room 105": 60,
    "Conference Hall C": 150,
}

# The raw dataset never specifies fixed per-room AV gear -- every officially
# listed room is assumed to carry standard conference equipment. The one
# genuinely *scarce* resource ("High-End AV Rig (GPU Demo)") is tracked
# separately as shared venue-wide equipment, not a per-room capability.
GENERIC_AV_CAPABILITIES: List[str] = [
    "Projector", "Dual Projectors", "Mic", "Lavalier Mic",
    "Standard HDMI", "Podiums", "Whiteboard", "Standard Projector", "Livestream",
]


@dataclass
class Room:
    """A conference room. `available=False` marks a room that isn't on the
    venue's official room list (a data-entry mistake the agent must catch)."""

    name: str
    capacity: int
    av_capabilities: List[str]
    available: bool = True
    maintenance_until: Optional[str] = None  # unusable for bookings starting before this time
    unavailable_reason: str = ""

    def supports(self, required_av: List[str]) -> bool:
        return set(required_av).issubset(set(self.av_capabilities))


@dataclass
class Session:
    """A single speaker's talk, as originally (messily) submitted."""

    session_id: str
    speaker: str
    email: str
    title: str
    track: str
    room: str
    time: str
    duration_min: int
    required_av: List[str]
    availability_start: str
    availability_end: str
    priority: int = 1

    original_room: Optional[str] = None
    original_time: Optional[str] = None
    status: str = "unscheduled"  # unscheduled | ok | moved | flagged
    resolution_note: str = ""


def _synth_email(speaker_name: str) -> str:
    slug = "-".join(speaker_name.lower().replace(".", "").replace("'", "").split())
    return f"{slug}@eventsync-speakers.demo"


def _build_rooms(available_room_names: List[str], maintenance: Dict[str, str], mentioned_rooms: set) -> List[Room]:
    all_names = set(available_room_names) | set(maintenance.keys()) | mentioned_rooms
    rooms: List[Room] = []
    for name in sorted(all_names):
        is_available = name in available_room_names
        note = maintenance.get(name, "")
        rooms.append(Room(
            name=name,
            capacity=_ROOM_CAPACITY_HINTS.get(name, DEFAULT_ROOM_CAPACITY),
            av_capabilities=list(GENERIC_AV_CAPABILITIES),
            available=is_available,
            maintenance_until=extract_clock_time(note) if note else None,
            unavailable_reason="" if is_available else "Not on the official venue room list.",
        ))
    return rooms


def load_dummy_data() -> Tuple[List[Room], List[Session], Dict[str, int]]:
    """Build fresh Room/Session objects + the shared-equipment registry from data.py."""
    raw_df = get_messy_event_data()
    constraints = get_venue_constraints()

    available_room_names = list(constraints.get("Available Rooms", []))
    maintenance = dict(constraints.get("Room Maintenance", {}))
    limited_equipment_raw = dict(constraints.get("Limited Equipment", {}))

    mentioned_rooms = {str(v).strip() for v in raw_df["Preferred Venue"]}
    rooms = _build_rooms(available_room_names, maintenance, mentioned_rooms)

    sessions: List[Session] = []
    for idx, row in raw_df.reset_index(drop=True).iterrows():
        start, end = parse_time_range(row["Preferred Time"])
        avail_start, avail_end = parse_time_range(row["Speaker Availability"])
        speaker = str(row["Speaker Name"]).strip()
        session = Session(
            session_id=f"S{idx + 1:02d}",
            speaker=speaker,
            email=_synth_email(speaker),
            title=str(row["Topic"]).strip(),
            track="General Session",
            room=str(row["Preferred Venue"]).strip(),
            time=start,
            duration_min=duration_minutes(start, end),
            required_av=parse_av_list(row["AV Requirements"]),
            availability_start=avail_start,
            availability_end=avail_end,
        )
        session.original_room = session.room
        session.original_time = session.time
        sessions.append(session)

    limited_equipment = {
        name: extract_capacity(desc, default=1)
        for name, desc in limited_equipment_raw.items()
    }

    return rooms, sessions, limited_equipment
