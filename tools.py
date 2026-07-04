"""
tools.py
--------
Canonical tool ("function-calling") definitions shared by every LLM
provider. Each tool is described once, using plain JSON-schema, and
converted into the provider-specific wire format inside `llm_providers.py`.

Executing a tool call always goes through `execute_tool`, which mutates the
shared `SchedulingEngine` and returns a small JSON-serializable result that
gets fed back to the model as the "tool result".
"""
from __future__ import annotations

from typing import Any, Dict, List

from scheduler_engine import SchedulingEngine

# ---------------------------------------------------------------------------
# Canonical tool specs (name, description, JSON-schema parameters)
# ---------------------------------------------------------------------------

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "inspect_room",
        "description": (
            "Look up everything currently booked into a given room, plus whether it is "
            "an officially available venue room, any maintenance window blocking it, and "
            "its generic AV capabilities. Use this to scan for potential clashes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "room_name": {"type": "string", "description": "Exact room name to inspect."}
            },
            "required": ["room_name"],
        },
    },
    {
        "name": "find_alternative_slot",
        "description": (
            "Given a session that is in conflict, search the venue for the best available "
            "(room, time) pair that: is an officially listed room, is free of any "
            "maintenance window, has no other booking, has enough headroom on any scarce "
            "shared equipment (e.g. a venue only has 1 GPU demo rig), carries the required "
            "generic AV gear, and falls fully inside the speaker's own stated availability "
            "window. Returns the first viable option, preferring to keep the session's "
            "original time if possible."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID of the session needing a new slot, e.g. 'S03'."}
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "reassign_session",
        "description": (
            "Move a session to a new room and/or time slot to resolve a conflict. Always "
            "call this AFTER confirming the target slot is free (e.g. via "
            "find_alternative_slot) so you never create a new collision, maintenance "
            "violation, equipment overbooking, or availability-window violation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID of the session to move."},
                "new_room": {"type": "string", "description": "New room name (omit to keep current room)."},
                "new_time": {"type": "string", "description": "New time slot, e.g. '3:00 PM' (omit to keep current time)."},
                "reason": {"type": "string", "description": "One-sentence human-readable justification for this change."},
            },
            "required": ["session_id", "reason"],
        },
    },
    {
        "name": "notify_speaker",
        "description": (
            "Simulate dispatching an automated email/SMS notification to a speaker "
            "informing them of their finalized or updated slot. This is a workflow-"
            "automation action, not a scheduling change -- call it once a session's "
            "final slot is confirmed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "ID of the session/speaker to notify."},
                "message": {"type": "string", "description": "Short message summarizing the update sent to the speaker."},
            },
            "required": ["session_id", "message"],
        },
    },
    {
        "name": "finalize_schedule",
        "description": (
            "Call this exactly once, after every conflict has been resolved and every "
            "affected speaker notified, to end the session and hand off the final, "
            "conflict-free schedule."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Short executive summary of everything the agent resolved."}
            },
            "required": ["summary"],
        },
    },
]

TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def execute_tool(engine: SchedulingEngine, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a single tool call onto the SchedulingEngine. Returns a JSON-safe dict."""

    if name == "inspect_room":
        room = engine.get_room(args.get("room_name", ""))
        if not room:
            return {"success": False, "error": f"Unknown room '{args.get('room_name')}'."}
        occupants = engine.sessions_in_room(room.name)
        return {
            "success": True,
            "room": room.name,
            "officially_available": room.available,
            "unavailable_reason": room.unavailable_reason,
            "capacity": room.capacity,
            "maintenance_until": room.maintenance_until,
            "av_capabilities": room.av_capabilities,
            "bookings": [
                {"session_id": s.session_id, "speaker": s.speaker, "time": s.time, "required_av": s.required_av}
                for s in occupants
            ],
        }

    if name == "find_alternative_slot":
        session = engine.get_session(args.get("session_id", ""))
        if not session:
            return {"success": False, "error": f"Unknown session_id '{args.get('session_id')}'."}
        result = engine.find_free_slot(session.session_id)
        if not result or isinstance(result, str):
            return {"success": False, "error": "No fully compatible free slot exists anywhere in the venue within this speaker's availability."}
        room, time = result
        return {"success": True, "session_id": session.session_id, "suggested_room": room, "suggested_time": time}

    if name == "reassign_session":
        result = engine.reassign_session(
            session_id=args.get("session_id", ""),
            new_room=args.get("new_room"),
            new_time=args.get("new_time"),
            reason=args.get("reason", ""),
        )
        if result.get("success"):
            engine.log_action(
                f"Schedule updated: {result['session_id']} moved from {result['from']} to {result['to']}."
            )
        return result

    if name == "notify_speaker":
        session = engine.get_session(args.get("session_id", ""))
        if not session:
            return {"success": False, "error": f"Unknown session_id '{args.get('session_id')}'."}
        message = args.get("message", "Your session slot has been updated.")
        engine.log_action(
            f"Simulated Action: Dispatched automated email to {session.speaker} <{session.email}> - \"{message}\""
        )
        return {"success": True, "notified": session.speaker, "email": session.email}

    if name == "finalize_schedule":
        engine.finalize_remaining()
        engine.log_action("Simulated Action: Published finalized schedule to the conference mobile app & website.")
        return {"success": True, "summary": args.get("summary", "Schedule finalized.")}

    return {"success": False, "error": f"Unknown tool '{name}'."}
