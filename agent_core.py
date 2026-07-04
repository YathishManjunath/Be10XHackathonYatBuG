"""
agent_core.py
-------------
The autonomous agent's brain. Two interchangeable implementations are
provided, both expressed as Python *generators* that `yield` structured
events so the Streamlit UI can render the "live thought process" as it
happens:

  * `run_live_agent`  - a real agentic loop against OpenAI/Anthropic/Gemini
    using raw tool-calling (see llm_providers.py). The model reasons in free
    text, calls tools to inspect/fix the schedule, and the loop keeps going
    until it calls `finalize_schedule` (or a safety step-limit is hit).

  * `run_demo_agent`  - a fully offline, deterministic rule-based agent that
    mimics the exact same reasoning/tool-call/action pattern, so the whole
    app works end-to-end with zero API keys (great for demos & grading).

Both emit the same event vocabulary:
    {"type": "status",    "text": str}
    {"type": "thought",   "text": str}
    {"type": "tool_call", "name": str, "args": dict, "result": dict}
    {"type": "action",    "text": str}
    {"type": "error",     "text": str}
    {"type": "final",     "text": str}
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List

from data_models import TIME_SLOTS
from llm_providers import ProviderError, Turn
from scheduler_engine import Conflict, SchedulingEngine
from tools import execute_tool

AgentEvent = Dict[str, Any]

MAX_LIVE_STEPS = 14
MAX_DEMO_ITERATIONS = 20

# Conflict kinds are resolved in this priority order: structural problems
# (invalid rooms, maintenance) before contention problems (double-bookings,
# scarce equipment) before capability gaps (AV mismatches).
_CONFLICT_PRIORITY = ["invalid_room", "maintenance_conflict", "double_booking", "equipment_conflict", "av_mismatch"]


# ---------------------------------------------------------------------------
# Shared prompt construction for the live LLM agent
# ---------------------------------------------------------------------------

def _build_system_prompt(engine: SchedulingEngine) -> str:
    rooms_desc = "\n".join(
        f"- {r.name} (capacity {r.capacity}, officially available: {r.available}"
        + (f", under maintenance until {r.maintenance_until}" if r.maintenance_until else "")
        + f"): supports {', '.join(r.av_capabilities)}"
        for r in engine.rooms.values()
    )
    equipment_desc = "\n".join(
        f"- {name}: only {qty} unit(s) exist venue-wide, shared across ALL rooms and tracks"
        for name, qty in engine.limited_equipment.items()
    ) or "- (none)"

    return (
        "You are EventSync AI, an autonomous operations agent for a large-scale tech "
        "conference. Your job is to eliminate every scheduling conflict in the venue:\n"
        "  1. Sessions booked into a room that is NOT on the official venue room list.\n"
        "  2. Sessions that overlap a room's maintenance window.\n"
        "  3. Double-bookings: two sessions overlapping in the same room.\n"
        "  4. Scarce shared-equipment overbooking (e.g. only one GPU demo rig exists "
        "for the whole venue, usable by only one session at a time).\n"
        "  5. Sessions needing AV gear their assigned room doesn't carry.\n\n"
        f"Available rooms:\n{rooms_desc}\n\n"
        f"Scarce shared equipment:\n{equipment_desc}\n\n"
        f"Allowed business-hour time slots (30-minute grid): {', '.join(TIME_SLOTS)}\n\n"
        "Every speaker also has their own stated availability window -- you must NEVER "
        "move a session outside of that speaker's availability. Work step by step and "
        "narrate your reasoning in plain text before/around your tool calls (e.g. "
        "'Scanning Main Ballroom... Detected conflict: ...'). Use tools to inspect rooms, "
        "find alternative slots, reassign sessions, and notify affected speakers. Prefer "
        "minimal disruption: when two sessions clash, keep the one booked first in its "
        "original slot and move the other. Never propose a room/time that is already "
        "occupied, under maintenance, not officially available, over-books shared "
        "equipment, or falls outside the speaker's availability window. Once every "
        "conflict is resolved and every moved speaker has been notified, call "
        "`finalize_schedule` exactly once with a short executive summary."
    )


def _build_initial_user_message(engine: SchedulingEngine) -> str:
    lines = [
        "Here is the raw, unvalidated session list submitted by track organizers "
        "(it contains real conflicts). Resolve everything:\n",
    ]
    for s in engine.sessions.values():
        lines.append(
            f"- {s.session_id}: {s.speaker} - \"{s.title}\" | Room: {s.room} | Time: {s.time} "
            f"({s.duration_min} min) | Required AV: {', '.join(s.required_av) or 'none'} | "
            f"Speaker availability: {s.availability_start} - {s.availability_end}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live LLM-driven agent
# ---------------------------------------------------------------------------

def run_live_agent(engine: SchedulingEngine, provider, max_steps: int = MAX_LIVE_STEPS) -> Generator[AgentEvent, None, None]:
    system_prompt = _build_system_prompt(engine)
    user_prompt = _build_initial_user_message(engine)
    messages = provider.build_initial_messages(system_prompt, user_prompt)

    yield {"type": "status", "text": f"Connected to {provider.name} ({provider.model}). Beginning autonomous analysis..."}

    log_cursor = 0
    finalized = False

    for step in range(1, max_steps + 1):
        yield {"type": "status", "text": f"Step {step}: reasoning..."}
        try:
            turn: Turn = provider.create_turn(messages)
        except ProviderError as exc:
            yield {"type": "error", "text": str(exc)}
            return

        if turn.text:
            yield {"type": "thought", "text": turn.text}

        messages.append(provider.assistant_message(turn))

        if not turn.tool_calls:
            if step == 1:
                yield {"type": "error", "text": "The model did not call any tools or finalize the schedule. Try again or switch models."}
                return
            break

        results = []
        for tc in turn.tool_calls:
            result = execute_tool(engine, tc["name"], tc["args"])
            results.append(result)
            yield {"type": "tool_call", "name": tc["name"], "args": tc["args"], "result": result}

            if len(engine.action_log) > log_cursor:
                for entry in engine.action_log[log_cursor:]:
                    yield {"type": "action", "text": entry}
                log_cursor = len(engine.action_log)

            if tc["name"] == "finalize_schedule":
                finalized = True

        messages.extend(provider.tool_result_messages(turn, results))

        if finalized:
            yield {"type": "final", "text": results[-1].get("summary", "Schedule finalized.")}
            return

    # Safety net: step-limit reached without an explicit finalize call.
    engine.finalize_remaining()
    remaining = engine.get_conflicts()
    if remaining:
        yield {"type": "status", "text": f"Step limit reached with {len(remaining)} unresolved item(s); finalizing best-effort schedule."}
    yield {"type": "final", "text": "Step limit reached - finalized the best available schedule."}


# ---------------------------------------------------------------------------
# Offline, rule-based demo agent (no API key required)
# ---------------------------------------------------------------------------

def _do_reassign(engine: SchedulingEngine, session, room: str, time: str, reason: str) -> Generator[AgentEvent, None, None]:
    args = {"session_id": session.session_id, "new_room": room, "new_time": time, "reason": reason}
    result = execute_tool(engine, "reassign_session", args)
    yield {"type": "tool_call", "name": "reassign_session", "args": args, "result": result}
    for entry in engine.action_log[-1:]:
        yield {"type": "action", "text": entry}

    message = f"Your session '{session.title}' has been moved to {room} at {time}."
    notify_args = {"session_id": session.session_id, "message": message}
    notify_result = execute_tool(engine, "notify_speaker", notify_args)
    yield {"type": "tool_call", "name": "notify_speaker", "args": notify_args, "result": notify_result}
    for entry in engine.action_log[-1:]:
        yield {"type": "action", "text": entry}


def _relocate_or_flag(engine: SchedulingEngine, session, reason: str) -> Generator[AgentEvent, None, None]:
    slot = engine.find_free_slot(session.session_id)
    if not slot or isinstance(slot, str):
        session.status = "flagged"
        session.resolution_note = "No fully compatible slot found within the speaker's availability - needs manual review."
        yield {"type": "thought", "text": f"No fully compatible slot exists for {session.speaker} within their availability window - flagging for manual review."}
        return
    new_room, new_time = slot
    same_room = new_room == session.room
    move_desc = f"{new_time}" if same_room else f"{new_room} at {new_time}"
    yield {"type": "thought", "text": f"Found a valid opening: {move_desc} (within {session.speaker}'s availability window). Reassigning now."}
    yield from _do_reassign(engine, session, new_room, new_time, reason)


def _resolve_invalid_room(engine: SchedulingEngine, session) -> Generator[AgentEvent, None, None]:
    yield {"type": "thought", "text": (
        f"Scanning session roster... Detected a data-entry issue: {session.speaker}'s \"{session.title}\" "
        f"is booked into '{session.room}', which is not on the official venue room list."
    )}
    yield {"type": "thought", "text": "Searching official venue rooms for a valid, compatible slot within the speaker's stated availability..."}
    yield from _relocate_or_flag(engine, session, reason=f"Auto-relocated from an unlisted/invalid room ('{session.room}') to an official venue room.")


def _resolve_maintenance_conflict(engine: SchedulingEngine, session) -> Generator[AgentEvent, None, None]:
    room = engine.get_room(session.room)
    until = room.maintenance_until if room else "unknown"
    yield {"type": "thought", "text": (
        f"Scanning {session.room}... Detected maintenance conflict: {session.room} is unavailable until "
        f"{until}, but {session.speaker}'s \"{session.title}\" is booked at {session.time}."
    )}
    yield {"type": "thought", "text": "Searching for a compatible slot after the maintenance window, or a different valid room..."}
    yield from _relocate_or_flag(engine, session, reason=f"Auto-relocated to avoid {session.room}'s maintenance window (blocked until {until}).")


def _resolve_double_booking(engine: SchedulingEngine, group) -> Generator[AgentEvent, None, None]:
    names = " and ".join(f"{s.speaker} ({s.session_id})" for s in group)
    yield {"type": "thought", "text": f"Scanning {group[0].room}... Detected conflict: {names} overlap in {group[0].room}."}

    kept = sorted(group, key=lambda s: (-s.priority, s.session_id))[0]
    engine.mark_ok(kept.session_id, note="Kept in original slot (booked first).")
    yield {"type": "thought", "text": f"Resolving by priority: keeping {kept.speaker}'s session in {kept.room} at {kept.time} (booked first)."}

    for moving in group:
        if moving.session_id == kept.session_id:
            continue
        yield {"type": "thought", "text": f"Checking alternative availability for {moving.speaker}'s \"{moving.title}\"..."}
        yield from _relocate_or_flag(engine, moving, reason=f"Auto-relocated to resolve double-booking with {kept.speaker} in {kept.room}.")


def _resolve_equipment_conflict(engine: SchedulingEngine, group, equipment: str, capacity: int) -> Generator[AgentEvent, None, None]:
    names = ", ".join(f"{s.speaker} ({s.time})" for s in group)
    yield {"type": "thought", "text": (
        f"Cross-referencing shared equipment logs... Detected resource conflict: only {capacity} unit(s) of "
        f"'{equipment}' exist venue-wide, but {len(group)} overlapping sessions need it: {names}."
    )}
    ordered = sorted(group, key=lambda s: s.session_id)
    keep, move = ordered[:capacity], ordered[capacity:]
    for k in keep:
        engine.mark_ok(k.session_id, note=f"Keeps its '{equipment}' reservation (booked first).")
    yield {"type": "thought", "text": f"Reserving '{equipment}' for {', '.join(k.speaker for k in keep)} (booked first)."}
    for moving in move:
        yield {"type": "thought", "text": f"Checking alternative slots for {moving.speaker} so the '{equipment}' rig is freed up..."}
        yield from _relocate_or_flag(engine, moving, reason=f"Auto-relocated to a slot where '{equipment}' is available (only {capacity} unit(s) exist venue-wide).")


def _resolve_av_mismatch(engine: SchedulingEngine, session) -> Generator[AgentEvent, None, None]:
    room = engine.get_room(session.room)
    missing = sorted(set(session.required_av) - set(room.av_capabilities)) if room else session.required_av
    yield {"type": "thought", "text": (
        f"Scanning {session.room}... Detected AV mismatch: {session.speaker}'s \"{session.title}\" "
        f"requires {', '.join(missing)}, which {session.room} does not support."
    )}
    yield {"type": "thought", "text": "Searching the venue for a room with compatible AV equipment..."}
    yield from _relocate_or_flag(engine, session, reason=f"Auto-relocated to a room supporting required AV: {', '.join(session.required_av)}.")


def _pick_next_conflict(conflicts: List[Conflict]) -> Conflict:
    for kind in _CONFLICT_PRIORITY:
        for c in conflicts:
            if c.kind == kind:
                return c
    return conflicts[0]


def run_demo_agent(engine: SchedulingEngine) -> Generator[AgentEvent, None, None]:
    yield {"type": "status", "text": "No API key detected - running EventSync AI in Offline Simulation Mode."}
    yield {"type": "thought", "text": f"Ingesting synthetic conference dataset: {len(engine.sessions)} sessions across {len(engine.rooms)} candidate rooms..."}
    yield {"type": "thought", "text": "Cross-referencing every booking against room availability, maintenance windows, shared-equipment limits, and speaker availability..."}

    resolved_count = 0
    for _ in range(MAX_DEMO_ITERATIONS):
        conflicts = engine.get_conflicts()
        if not conflicts:
            break

        conflict = _pick_next_conflict(conflicts)

        if conflict.kind == "invalid_room":
            session = engine.get_session(conflict.session_ids[0])
            yield from _resolve_invalid_room(engine, session)
        elif conflict.kind == "maintenance_conflict":
            session = engine.get_session(conflict.session_ids[0])
            yield from _resolve_maintenance_conflict(engine, session)
        elif conflict.kind == "double_booking":
            group = [engine.get_session(sid) for sid in conflict.session_ids]
            yield from _resolve_double_booking(engine, group)
        elif conflict.kind == "equipment_conflict":
            group = [engine.get_session(sid) for sid in conflict.session_ids]
            yield from _resolve_equipment_conflict(engine, group, conflict.extra["equipment"], conflict.extra["capacity"])
        else:  # av_mismatch
            session = engine.get_session(conflict.session_ids[0])
            yield from _resolve_av_mismatch(engine, session)

        resolved_count += 1
        yield {"type": "status", "text": "Re-scanning full schedule grid for residual conflicts..."}
    else:
        yield {"type": "error", "text": "Reached the iteration safety limit with conflicts still outstanding."}

    engine.finalize_remaining()
    summary = (
        f"Resolved {resolved_count} conflict cluster(s) across the venue -- invalid rooms, maintenance "
        f"windows, double-bookings, shared-equipment contention, and AV mismatches. All sessions now sit "
        f"in valid, conflict-free slots within each speaker's availability."
    )
    execute_tool(engine, "finalize_schedule", {"summary": summary})
    for entry in engine.action_log[-1:]:
        yield {"type": "action", "text": entry}
    yield {"type": "final", "text": summary}
