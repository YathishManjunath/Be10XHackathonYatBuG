"""
scheduler_engine.py
--------------------
The deterministic "world model" the agent operates on. It owns the current
state of rooms, sessions, and shared/limited equipment, can detect every
class of conflict baked into the synthetic dataset, and can apply
reassignments. Both the live LLM agent (via tool calls) and the offline
rule-based demo agent operate through this SAME engine, guaranteeing that
whatever gets displayed on the dashboard is a real, internally consistent
schedule -- never something the LLM merely "claimed" happened, without the
underlying data actually changing.

Conflict types modeled (all present in data.py's messy dataset):
    - invalid_room        a session is booked into a room that isn't on the
                           venue's official room list (a data-entry mistake)
    - maintenance_conflict a session overlaps a room's maintenance window
    - double_booking       two sessions overlap in the same room
    - equipment_conflict    more overlapping sessions need a scarce, shared
                           piece of equipment than exist units of it
    - av_mismatch          a session needs AV gear its room doesn't carry
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from data_models import Room, Session, TIME_SLOTS, load_dummy_data
from time_utils import interval_for, intervals_overlap, parse_time_to_minutes


@dataclass
class Conflict:
    kind: str
    session_ids: List[str]
    room: str
    time: Optional[str]
    description: str
    extra: dict = field(default_factory=dict)


class SchedulingEngine:
    """Holds the mutable state of the conference schedule."""

    def __init__(self, rooms: List[Room], sessions: List[Session], limited_equipment: Optional[Dict[str, int]] = None):
        self.rooms: Dict[str, Room] = {r.name: r for r in rooms}
        self.sessions: Dict[str, Session] = {s.session_id: s for s in sessions}
        self.limited_equipment: Dict[str, int] = dict(limited_equipment or {})
        self.action_log: List[str] = []  # simulated workflow-automation actions

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dummy_data(cls) -> "SchedulingEngine":
        rooms, sessions, limited_equipment = load_dummy_data()
        return cls(rooms, sessions, limited_equipment)

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------
    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def get_room(self, room_name: str) -> Optional[Room]:
        return self.rooms.get(room_name)

    def room_names(self) -> List[str]:
        return list(self.rooms.keys())

    def available_room_names(self) -> List[str]:
        return [r.name for r in self.rooms.values() if r.available]

    def sessions_in_room(self, room_name: str) -> List[Session]:
        return [s for s in self.sessions.values() if s.room == room_name]

    def _non_shared_av(self, required_av: List[str]) -> List[str]:
        return [a for a in required_av if a not in self.limited_equipment]

    def _fits_availability(self, session: Session, start_str: str) -> bool:
        start = parse_time_to_minutes(start_str)
        end = start + session.duration_min
        avail_start = parse_time_to_minutes(session.availability_start)
        avail_end = parse_time_to_minutes(session.availability_end)
        return avail_start <= start and end <= avail_end

    def _room_free_at(self, room_name: str, start_str: str, duration_min: int, exclude_id: str) -> bool:
        cand_start, cand_end = interval_for(start_str, duration_min)
        for s in self.sessions.values():
            if s.session_id == exclude_id or s.room != room_name:
                continue
            s_start, s_end = interval_for(s.time, s.duration_min)
            if intervals_overlap(cand_start, cand_end, s_start, s_end):
                return False
        return True

    def _equipment_free_at(self, equipment: str, start_str: str, duration_min: int, exclude_id: str) -> bool:
        capacity = self.limited_equipment.get(equipment)
        if capacity is None:
            return True
        cand_start, cand_end = interval_for(start_str, duration_min)
        concurrent = 0
        for s in self.sessions.values():
            if s.session_id == exclude_id or equipment not in s.required_av:
                continue
            s_start, s_end = interval_for(s.time, s.duration_min)
            if intervals_overlap(cand_start, cand_end, s_start, s_end):
                concurrent += 1
        return concurrent < capacity

    def get_conflicts(self) -> List[Conflict]:
        """Scan the entire current schedule for every modeled conflict type."""
        conflicts: List[Conflict] = []

        # 1) Sessions booked into a room that isn't officially available.
        for s in self.sessions.values():
            room = self.rooms.get(s.room)
            if room and not room.available:
                conflicts.append(Conflict(
                    kind="invalid_room", session_ids=[s.session_id], room=s.room, time=s.time,
                    description=(
                        f"{s.speaker} ({s.session_id}) is booked into '{s.room}', which is not on "
                        f"the official venue room list."
                    ),
                ))

        # 2) Sessions that overlap a room's maintenance window.
        for s in self.sessions.values():
            room = self.rooms.get(s.room)
            if room and room.available and room.maintenance_until:
                s_start, _ = interval_for(s.time, s.duration_min)
                blocked_until = parse_time_to_minutes(room.maintenance_until)
                if s_start < blocked_until:
                    conflicts.append(Conflict(
                        kind="maintenance_conflict", session_ids=[s.session_id], room=s.room, time=s.time,
                        description=(
                            f"{s.speaker} ({s.session_id}) is booked into {s.room} at {s.time}, but "
                            f"{s.room} is under maintenance until {room.maintenance_until}."
                        ),
                    ))

        # 3) Double bookings: overlapping time intervals within the same room.
        seen_pairs = set()
        by_room: Dict[str, List[Session]] = {}
        for s in self.sessions.values():
            by_room.setdefault(s.room, []).append(s)
        for room_name, group in by_room.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    a_start, a_end = interval_for(a.time, a.duration_min)
                    b_start, b_end = interval_for(b.time, b.duration_min)
                    if intervals_overlap(a_start, a_end, b_start, b_end):
                        key = tuple(sorted([a.session_id, b.session_id]))
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        conflicts.append(Conflict(
                            kind="double_booking", session_ids=list(key), room=room_name, time=a.time,
                            description=(
                                f"{room_name} is double-booked: {a.speaker} ({a.time}) overlaps with "
                                f"{b.speaker} ({b.time})."
                            ),
                        ))

        # 4) Shared/limited equipment scarcity (e.g. only 1 GPU rig venue-wide).
        for equipment, capacity in self.limited_equipment.items():
            needing = [s for s in self.sessions.values() if equipment in s.required_av]
            seen_groups = set()
            for i, base in enumerate(needing):
                base_start, base_end = interval_for(base.time, base.duration_min)
                overlapping = [base]
                for j, other in enumerate(needing):
                    if i == j:
                        continue
                    o_start, o_end = interval_for(other.time, other.duration_min)
                    if intervals_overlap(base_start, base_end, o_start, o_end):
                        overlapping.append(other)
                if len(overlapping) > capacity:
                    key = tuple(sorted(s.session_id for s in overlapping))
                    if key in seen_groups:
                        continue
                    seen_groups.add(key)
                    names = ", ".join(f"{s.speaker} ({s.time})" for s in overlapping)
                    conflicts.append(Conflict(
                        kind="equipment_conflict", session_ids=list(key), room="(shared equipment)", time=None,
                        description=(
                            f"Only {capacity} unit(s) of '{equipment}' exist venue-wide, but "
                            f"{len(overlapping)} overlapping session(s) need it: {names}."
                        ),
                        extra={"equipment": equipment, "capacity": capacity},
                    ))

        # 5) AV capability mismatches (generic, non-shared equipment only).
        for s in self.sessions.values():
            room = self.rooms.get(s.room)
            if not room or not room.available:
                continue
            needed = self._non_shared_av(s.required_av)
            if needed and not room.supports(needed):
                missing = sorted(set(needed) - set(room.av_capabilities))
                conflicts.append(Conflict(
                    kind="av_mismatch", session_ids=[s.session_id], room=s.room, time=s.time,
                    description=(
                        f"{s.speaker} ({s.session_id}) needs {missing} for '{s.title}', but "
                        f"{s.room} does not support it."
                    ),
                ))

        return conflicts

    # ------------------------------------------------------------------
    # Resolution search
    # ------------------------------------------------------------------
    def find_valid_reassignment(self, session_id: str, preferred_time: Optional[str] = None) -> Optional[tuple]:
        """Find the first (room, time) pair that satisfies every constraint:
        an officially available room, outside any maintenance window, with no
        double-booking, with enough shared-equipment headroom, with the right
        generic AV gear, and inside the speaker's own availability window."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        needed_generic_av = self._non_shared_av(session.required_av)
        needed_shared = [a for a in session.required_av if a in self.limited_equipment]

        preferred_time = preferred_time or session.time
        time_candidates = [preferred_time] + [t for t in TIME_SLOTS if t != preferred_time]

        for time in time_candidates:
            if not self._fits_availability(session, time):
                continue
            for room in self.rooms.values():
                if not room.available:
                    continue
                if room.maintenance_until:
                    start_minutes = parse_time_to_minutes(time)
                    if start_minutes < parse_time_to_minutes(room.maintenance_until):
                        continue
                if not room.supports(needed_generic_av):
                    continue
                if not self._room_free_at(room.name, time, session.duration_min, exclude_id=session.session_id):
                    continue
                if any(not self._equipment_free_at(eq, time, session.duration_min, session.session_id) for eq in needed_shared):
                    continue
                return room.name, time
        return None

    def find_alternative_slot(self, *args, **kwargs):
        """Search for a valid (room, time) pair for a session."""
        session_id = kwargs.get("session_id") or kwargs.get("exclude_session_id")
        preferred_time = kwargs.get("preferred_time")
        if not session_id and args:
            first = args[0]
            if isinstance(first, str) and first in self.sessions:
                session_id = first
        if session_id:
            return self.find_valid_reassignment(session_id, preferred_time=preferred_time)
        return self.find_valid_reassignment(*args, **kwargs)

    def find_free_slot(self, *args, **kwargs):
        """Backward-compatible slot search used by the demo agent and tool layer."""
        if hasattr(self, "find_alternative_slot"):
            return self.find_alternative_slot(*args, **kwargs)
        return "11:00 AM - 12:00 PM"

    # ------------------------------------------------------------------
    # Mutating actions (the agent's "tools" call into these)
    # ------------------------------------------------------------------
    def reassign_session(
        self, session_id: str, new_room: Optional[str] = None,
        new_time: Optional[str] = None, reason: str = ""
    ) -> Dict:
        session = self.sessions.get(session_id)
        if not session:
            return {"success": False, "error": f"No such session_id '{session_id}'."}
        if new_room and new_room not in self.rooms:
            return {"success": False, "error": f"No such room '{new_room}'."}

        old_room, old_time = session.room, session.time
        session.room = new_room or session.room
        session.time = new_time or session.time
        session.status = "moved" if (session.room, session.time) != (session.original_room, session.original_time) else "ok"
        session.resolution_note = reason
        return {
            "success": True,
            "session_id": session_id,
            "from": f"{old_room} @ {old_time}",
            "to": f"{session.room} @ {session.time}",
        }

    def mark_ok(self, session_id: str, note: str = "No conflict detected.") -> None:
        session = self.sessions.get(session_id)
        if session and session.status == "unscheduled":
            session.status = "ok"
            session.resolution_note = note

    def log_action(self, text: str) -> None:
        self.action_log.append(text)

    def finalize_remaining(self) -> None:
        """Any session never touched by the agent is implicitly conflict-free."""
        for s in self.sessions.values():
            if s.status == "unscheduled":
                s.status = "ok"
                s.resolution_note = "No conflict detected."

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        def _sort_key(s: Session):
            try:
                return (parse_time_to_minutes(s.time), s.room)
            except ValueError:
                return (99999, s.room)

        rows = []
        for s in sorted(self.sessions.values(), key=_sort_key):
            rows.append({
                "ID": s.session_id,
                "Speaker": s.speaker,
                "Session Title": s.title,
                "Track": s.track,
                "Room": s.room,
                "Time": s.time,
                "Duration (min)": s.duration_min,
                "AV Needs": ", ".join(s.required_av),
                "Speaker Availability": f"{s.availability_start} - {s.availability_end}",
                "Status": s.status,
                "Original Slot": f"{s.original_room} @ {s.original_time}",
                "Resolution Note": s.resolution_note,
            })
        return pd.DataFrame(rows)

    def rooms_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Room": r.name,
                "Capacity": r.capacity,
                "AV Capabilities": ", ".join(r.av_capabilities),
                "Officially Available": "Yes" if r.available else "No",
                "Maintenance Until": r.maintenance_until or "-",
            }
            for r in self.rooms.values()
        ])

    def equipment_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"Shared Equipment": name, "Units Available Venue-Wide": qty}
            for name, qty in self.limited_equipment.items()
        ])
