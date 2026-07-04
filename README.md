# 🎪 EventSync AI

An autonomous multi-agent Streamlit application that solves the operational nightmare of
**Speaker, Session, and AV coordination** for large-scale conferences.

Feed it a messy, conflict-ridden session list — double-booked rooms, a session booked into
a room that doesn't officially exist, a room shut for maintenance, a scarce shared AV rig
two speakers both need — and watch the agent reason through it step by step, resolve every
conflict, and simulate the downstream workflow automation (speaker email notifications,
schedule publishing) it would trigger in production.

## ✨ Features

1. **Sidebar configuration** — plug in an OpenAI, Anthropic, or **Google Gemini** API key,
   pick a model, or run entirely offline in **Demo Mode** (zero network calls, zero API key
   required).
2. **One-click synthetic data** — sourced live from `data.py`'s `get_messy_event_data()` and
   `get_venue_constraints()`: messy speaker requests plus venue constraints (an unofficial
   room, a maintenance window, and a shared AV rig with only one unit).
3. **Agentic core** — a real agentic loop built on **raw OpenAI / Anthropic / Gemini
   tool-calling** (no LangChain/CrewAI/AutoGen). The agent inspects rooms, searches for
   alternative slots, reassigns sessions, and notifies speakers, all via explicit tool calls
   against a deterministic scheduling engine — so nothing the model "says" can drift from
   what actually happened to the data.
4. **Visual agent thought process** — a live, three-pane console showing the agent's
   reasoning, tool calls, and simulated workflow-automation actions as they happen.
5. **Interactive dashboard** — the final conflict-free schedule as a styled DataFrame, a
   Plotly room/time timeline ("interactive calendar"), and a CSV export.

## 🧩 Conflict types modeled

The engine (`scheduler_engine.py`) understands every constraint baked into `data.py`:

| Kind | Example |
|---|---|
| `invalid_room` | A session is booked into a room that isn't on the official venue list |
| `maintenance_conflict` | A session overlaps a room's maintenance window |
| `double_booking` | Two sessions overlap in the same room |
| `equipment_conflict` | More overlapping sessions need a scarce shared item (e.g. one GPU rig) than exist |
| `av_mismatch` | A session needs AV gear its room doesn't carry |

Every reassignment the agent makes is also checked against each speaker's own stated
**availability window**, so a fix is never "valid on paper" but impossible for the speaker.

## 🗂️ Project layout

```
app.py                # Streamlit UI/UX layer
data.py               # Raw synthetic dataset: get_messy_event_data(), get_venue_constraints()
data_models.py        # Translates data.py into structured Room/Session objects
time_utils.py         # Shared time-range parsing & interval-overlap helpers
scheduler_engine.py   # Deterministic scheduling "world model" (conflict detection & fixes)
tools.py               # Canonical agent tool specs + tool executor
llm_providers.py       # Thin, normalized wrappers around raw OpenAI/Anthropic/Gemini SDK calls
agent_core.py           # The agentic loop (live LLM) + offline rule-based demo agent
requirements.txt
```

## 🚀 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then, in the sidebar:

1. Choose **Demo Mode** (instant, offline) or plug in an **OpenAI** / **Anthropic** /
   **Google Gemini** API key.
2. Click **Load Data** to populate the synthetic, messy conference schedule.
3. Open the **Agent Console** tab and click **Run EventSync AI Agent**.
4. Check the **Optimized Schedule** tab for the final, conflict-free result.

## 🔐 Notes on API keys & data

- API keys are only ever held in Streamlit's in-memory session state for the current run —
  they are never written to disk or logged.
- If a live provider is selected without a key, the **Run** button is disabled and a clear
  error is shown instead of silently failing.
- All conference data is synthetic and generated locally from `data.py`; nothing is fetched
  from or sent to any external data source besides the LLM provider you explicitly choose.

## 🧠 A note on the Gemini integration

Google fully deprecated the legacy `google-generativeai` package (end-of-life November 30,
2025). `llm_providers.py` uses the current, officially supported **`google-genai`** SDK
(`pip install google-genai`, `from google import genai`) so Gemini tool-calling actually
works today, while keeping the exact same reasoning/tool-call/notification UX as the OpenAI
and Anthropic providers.
