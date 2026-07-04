"""
EventSync AI
============
An autonomous multi-agent Streamlit application that solves Speaker, Session,
and AV coordination for large-scale conferences.

Run with:
    streamlit run app.py

Architecture (kept intentionally framework-free & modular):
    data_models.py      -> synthetic/dummy data (rooms, sessions)
    scheduler_engine.py -> the deterministic scheduling "world model"
    tools.py             -> agent tool specs + tool executor
    llm_providers.py     -> raw OpenAI / Anthropic tool-calling wrappers
    agent_core.py         -> the agentic loop (live LLM) + offline demo agent
    app.py                -> this file: the Streamlit UI/UX layer
"""
from __future__ import annotations

import importlib
import time
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

# Force-reload engine modules on every Streamlit rerun so code edits (e.g.
# new SchedulingEngine methods) are picked up without a manual server restart.
import agent_core as _agent_core
import scheduler_engine as _scheduler_engine
import tools as _tools
importlib.reload(_scheduler_engine)
importlib.reload(_tools)
importlib.reload(_agent_core)

from agent_core import run_demo_agent, run_live_agent
from data_models import TIME_SLOTS
from llm_providers import PROVIDERS, ProviderError
from scheduler_engine import SchedulingEngine

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EventSync AI",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .es-hero {
        padding: 1.4rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #4338CA 0%, #7C3AED 55%, #C026D3 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .es-hero h1 { margin: 0 0 0.2rem 0; font-size: 2rem; }
    .es-hero p { margin: 0; opacity: 0.92; font-size: 1.02rem; }

    .thought-bubble {
        background: #F1F5F9;
        border-left: 4px solid #6366F1;
        padding: 0.55rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
        color: #1E293B;
    }
    .status-line {
        color: #64748B;
        font-size: 0.82rem;
        font-style: italic;
        margin-bottom: 0.4rem;
    }
    .tool-line {
        background: #0F172A;
        color: #7DD3FC;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.82rem;
        margin-bottom: 0.4rem;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .action-line {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #065F46;
        margin-bottom: 0.4rem;
    }
    .final-banner {
        background: #EEF2FF;
        border: 1px solid #6366F1;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #312E81;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] { color: #4338CA; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

DEFAULTS = {
    "raw_engine": None,        # pristine copy, used only for "before" preview
    "engine": None,            # working copy, mutated by the agent
    "events": [],              # full transcript of agent events (persisted)
    "run_complete": False,
    "data_loaded": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_run_state() -> None:
    st.session_state["events"] = []
    st.session_state["run_complete"] = False
    st.session_state["engine"] = None


# ---------------------------------------------------------------------------
# Sidebar: configuration + synthetic data controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    provider_choice = st.radio(
        "Agent Intelligence Source",
        options=["Demo Mode (No API Key)", "OpenAI", "Anthropic", "Google Gemini (AI Studio)"],
        index=0,
        help="Use Demo Mode for an instant, fully offline simulation, or plug in a real LLM for live tool-calling reasoning.",
    )

    api_key = ""
    model_choice = None
    if provider_choice in PROVIDERS:
        api_key = st.text_input(
            f"{provider_choice} API Key",
            type="password",
            placeholder=PROVIDERS[provider_choice]["key_placeholder"],
            help="Your key is only kept in-memory for this session and is never stored or logged.",
        )
        model_choice = st.selectbox("Model", options=PROVIDERS[provider_choice]["models"], index=0)
        if not api_key:
            st.caption("🔒 No key entered yet — the Run button will stay disabled until you add one.")
        if provider_choice == "Google Gemini (AI Studio)":
            st.caption("Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).")
    else:
        st.caption("🧪 Fully offline rule-based simulation — no network calls, no API key needed.")

    st.divider()
    st.markdown("## 🗂️ Synthetic Conference Data")
    st.caption(
        "Sourced live from `data.py` — messy speaker requests plus venue constraints: "
        "double-bookings, an invalid room, a maintenance window, and a scarce shared AV rig."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📥 Load Data", width="stretch"):
            raw_rooms_engine = SchedulingEngine.from_dummy_data()
            st.session_state["raw_engine"] = raw_rooms_engine
            st.session_state["data_loaded"] = True
            reset_run_state()
            st.toast("Synthetic conference dataset loaded.", icon="✅")
    with col_b:
        if st.button("♻️ Reset All", width="stretch"):
            for key, value in DEFAULTS.items():
                st.session_state[key] = value
            st.toast("Session reset.", icon="🔄")

    if st.session_state["data_loaded"]:
        n_sessions = len(st.session_state["raw_engine"].sessions)
        n_conflicts = len(st.session_state["raw_engine"].get_conflicts())
        st.success(f"Loaded {n_sessions} sessions · {n_conflicts} conflict clusters detected.")
    else:
        st.info("Click **Load Data** to populate the synthetic dataset.")

    st.divider()
    st.caption(
        "EventSync AI · autonomous multi-agent scheduling assistant. "
        "Built with raw OpenAI / Anthropic / Gemini tool-calling — no LangChain, no CrewAI."
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="es-hero">
        <h1>🎪 EventSync AI</h1>
        <p>An autonomous agent that untangles Speaker, Session & AV conflicts for large-scale conferences —
        in real time, with full visibility into its reasoning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["data_loaded"]:
    st.warning("👈 Start by loading the synthetic conference dataset from the sidebar.")
    st.stop()

raw_engine: SchedulingEngine = st.session_state["raw_engine"]

tab_raw, tab_agent, tab_dashboard = st.tabs(
    ["🗂️ Raw Data (Before)", "🧠 Agent Console", "📅 Optimized Schedule (After)"]
)


# ---------------------------------------------------------------------------
# TAB 1 — Raw / messy data preview
# ---------------------------------------------------------------------------

with tab_raw:
    st.subheader("Unvalidated submissions from track organizers")
    st.caption("This is exactly what was fed to the agent — conflicts included, on purpose. Source: `data.py`.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(
            raw_engine.to_dataframe()[["ID", "Speaker", "Session Title", "Room", "Time", "AV Needs", "Speaker Availability"]],
            width="stretch",
            hide_index=True,
        )
    with c2:
        st.markdown("**Venue Rooms**")
        st.dataframe(raw_engine.rooms_dataframe(), width="stretch", hide_index=True)
        st.markdown("**Scarce Shared Equipment**")
        st.dataframe(raw_engine.equipment_dataframe(), width="stretch", hide_index=True)

    conflicts = raw_engine.get_conflicts()
    st.markdown(f"### 🚨 {len(conflicts)} Conflict Cluster(s) Detected")
    if not conflicts:
        st.success("No conflicts found in the raw data.")
    icons = {
        "invalid_room": "🚫",
        "maintenance_conflict": "🛠️",
        "double_booking": "🔁",
        "equipment_conflict": "🎛️",
        "av_mismatch": "🎥",
    }
    for c in conflicts:
        st.error(f"{icons.get(c.kind, '⚠️')} **{c.kind.replace('_', ' ').title()}** — {c.description}")


# ---------------------------------------------------------------------------
# TAB 2 — Agent console (live thought process)
# ---------------------------------------------------------------------------

def render_events(events, thought_ph, tool_ph, action_ph, status_ph):
    thoughts = [e for e in events if e["type"] == "thought"]
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    actions = [e for e in events if e["type"] == "action"]
    statuses = [e for e in events if e["type"] == "status"]
    errors = [e for e in events if e["type"] == "error"]

    if statuses:
        status_ph.markdown(f'<div class="status-line">⏳ {statuses[-1]["text"]}</div>', unsafe_allow_html=True)

    with thought_ph.container(height=380, border=True):
        if not thoughts:
            st.caption("Agent reasoning will stream here once you click **Run EventSync AI Agent**...")
        for e in thoughts:
            st.markdown(f'<div class="thought-bubble">🧠 {e["text"]}</div>', unsafe_allow_html=True)
        for e in errors:
            st.error(e["text"])

    with tool_ph.container(height=380, border=True):
        if not tool_calls:
            st.caption("Tool calls (inspect_room, find_alternative_slot, reassign_session, ...) will appear here.")
        for e in tool_calls:
            args_str = ", ".join(f"{k}={v!r}" for k, v in e["args"].items())
            ok = e["result"].get("success", True)
            icon = "✅" if ok else "⚠️"
            st.markdown(
                f'<div class="tool-line">{icon} {e["name"]}({args_str})\n→ {e["result"]}</div>',
                unsafe_allow_html=True,
            )

    with action_ph.container(height=380, border=True):
        if not actions:
            st.caption("Simulated workflow-automation actions (emails, publishing) will appear here.")
        for e in actions:
            st.markdown(f'<div class="action-line">📨 {e["text"]}</div>', unsafe_allow_html=True)


with tab_agent:
    st.subheader("Autonomous Agent Console")
    st.caption("Watch EventSync AI scan, reason about, and resolve every conflict — live.")

    key_missing = provider_choice in PROVIDERS and not api_key
    run_disabled = key_missing
    run_clicked = st.button(
        "▶️ Run EventSync AI Agent",
        type="primary",
        disabled=run_disabled,
        width="stretch",
    )
    if key_missing:
        st.error(f"Please enter your {provider_choice} API key in the sidebar, or switch to **Demo Mode**.")

    status_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    col1.markdown("**🧠 Reasoning Log**")
    col2.markdown("**🔧 Tool Execution Log**")
    col3.markdown("**📨 Workflow Automation Log**")
    thought_placeholder = col1.empty()
    tool_placeholder = col2.empty()
    action_placeholder = col3.empty()

    if run_clicked:
        reset_run_state()
        fresh_engine = SchedulingEngine.from_dummy_data()

        try:
            if provider_choice == "Demo Mode (No API Key)":
                generator = run_demo_agent(fresh_engine)
            else:
                provider_cls = PROVIDERS[provider_choice]["class"]
                provider_instance = provider_cls(api_key=api_key, model=model_choice)
                generator = run_live_agent(fresh_engine, provider_instance)

            events = []
            with st.spinner("EventSync AI is coordinating your conference..."):
                for event in generator:
                    events.append(event)
                    st.session_state["events"] = events
                    render_events(events, thought_placeholder, tool_placeholder, action_placeholder, status_placeholder)
                    if event["type"] == "error":
                        break
                    time.sleep(0.25)

            st.session_state["engine"] = fresh_engine
            final_events = [e for e in events if e["type"] == "final"]
            if final_events:
                st.session_state["run_complete"] = True
                status_placeholder.empty()
                st.markdown(f'<div class="final-banner">✅ {final_events[-1]["text"]}</div>', unsafe_allow_html=True)
                st.balloons()

        except ProviderError as exc:
            st.error(f"⚠️ {exc}")
        except Exception as exc:  # noqa: BLE001 - surface any unexpected failure gracefully
            st.error(f"⚠️ Unexpected error while running the agent: {exc}")

    elif st.session_state["events"]:
        render_events(st.session_state["events"], thought_placeholder, tool_placeholder, action_placeholder, status_placeholder)
        if st.session_state["run_complete"]:
            final_events = [e for e in st.session_state["events"] if e["type"] == "final"]
            if final_events:
                st.markdown(f'<div class="final-banner">✅ {final_events[-1]["text"]}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TAB 3 — Final dashboard
# ---------------------------------------------------------------------------

def build_timeline_dataframe(engine: SchedulingEngine) -> pd.DataFrame:
    base_date = datetime(2026, 9, 15)
    slot_to_dt = {}
    for slot in TIME_SLOTS:
        dt = datetime.strptime(slot, "%I:%M %p")
        slot_to_dt[slot] = base_date.replace(hour=dt.hour, minute=dt.minute)

    rows = []
    for s in engine.sessions.values():
        start = slot_to_dt.get(s.time, base_date)
        finish = start + timedelta(minutes=s.duration_min)
        rows.append({
            "Speaker": s.speaker,
            "Session": s.title,
            "Room": s.room,
            "Start": start,
            "Finish": finish,
            "Status": s.status,
        })
    return pd.DataFrame(rows)


with tab_dashboard:
    if not st.session_state["run_complete"] or st.session_state["engine"] is None:
        st.info("Run the agent from the **Agent Console** tab to generate the optimized, conflict-free schedule.")
    else:
        engine: SchedulingEngine = st.session_state["engine"]
        df = engine.to_dataframe()

        moved = (df["Status"] == "moved").sum()
        flagged = (df["Status"] == "flagged").sum()
        remaining_conflicts = len(engine.get_conflicts())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Sessions", len(df))
        m2.metric("Sessions Relocated", int(moved))
        m3.metric("Rooms Utilized", df["Room"].nunique())
        m4.metric("Remaining Conflicts", remaining_conflicts, delta=None if remaining_conflicts == 0 else "needs review")

        if flagged:
            st.warning(f"{flagged} session(s) could not be automatically resolved and were flagged for manual review.")
        elif remaining_conflicts == 0:
            st.success("🎉 Schedule is fully optimized — zero double-bookings, zero AV mismatches.")

        st.markdown("### 📋 Final Conflict-Free Schedule")

        def _highlight_status(row):
            color = {"ok": "#F0FDF4", "moved": "#FFFBEB", "flagged": "#FEF2F2"}.get(row["Status"], "white")
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            df.style.apply(_highlight_status, axis=1),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Final Schedule (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="eventsync_final_schedule.csv",
            mime="text/csv",
        )

        st.markdown("### 🗓️ Interactive Room Calendar")
        timeline_df = build_timeline_dataframe(engine)
        fig = px.timeline(
            timeline_df,
            x_start="Start", x_end="Finish", y="Room", color="Status",
            hover_data=["Speaker", "Session"],
            color_discrete_map={"ok": "#22C55E", "moved": "#F59E0B", "flagged": "#EF4444"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="Status")
        st.plotly_chart(fig, width="stretch")

        st.markdown("### 📜 Full Automation Log")
        if engine.action_log:
            for i, entry in enumerate(engine.action_log, 1):
                st.markdown(f'<div class="action-line">{i}. {entry}</div>', unsafe_allow_html=True)
        else:
            st.caption("No automated actions were logged.")
