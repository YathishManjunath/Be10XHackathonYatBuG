"""
llm_providers.py
-----------------
Thin, dependency-isolated wrappers around the official OpenAI, Anthropic, and
Google Gemini SDKs (raw API calls -- no LangChain/CrewAI/AutoGen). Each
provider exposes the same normalized interface so `agent_core.py` can drive
the agentic loop without caring which vendor is behind it:

    provider.build_initial_messages(system_prompt, user_prompt) -> messages
    provider.create_turn(messages)                              -> Turn
    provider.assistant_message(turn)                            -> message to append
    provider.tool_result_messages(turn, results)                -> messages to append

`tool_calls` items on a `Turn` are normalized dicts: {"id": str, "name": str, "args": dict}

Note on Gemini: Google fully deprecated the legacy `google-generativeai`
package (end-of-life November 30, 2025). This module uses the current,
officially supported `google-genai` SDK (`pip install google-genai`,
`from google import genai`) so Gemini support actually works today, while
keeping the exact same tool-calling UX described in the product spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from tools import TOOL_SPECS


class ProviderError(RuntimeError):
    """Raised for any authentication / network / API-level failure."""


@dataclass
class Turn:
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIProvider:
    name = "OpenAI"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("The 'openai' package is not installed. Run: pip install openai") from exc

        if not api_key or not api_key.strip():
            raise ProviderError("An OpenAI API key is required but was not provided.")

        self.model = model
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"Failed to initialize OpenAI client: {exc}") from exc

    @staticmethod
    def _wire_tools() -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in TOOL_SPECS
        ]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def create_turn(self, messages: List[Dict[str, Any]]) -> Turn:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._wire_tools(),
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI API call failed: {exc}") from exc

        msg = response.choices[0].message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return Turn(text=msg.content or "", tool_calls=tool_calls, raw=msg)

    def assistant_message(self, turn: Turn) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": turn.text or None}
        if turn.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])},
                }
                for tc in turn.tool_calls
            ]
        return msg

    def tool_result_messages(self, turn: Turn, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for tc, result in zip(turn.tool_calls, results):
            out.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result)})
        return out


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "Anthropic"

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc

        if not api_key or not api_key.strip():
            raise ProviderError("An Anthropic API key is required but was not provided.")

        self.model = model
        self._system_prompt = ""
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"Failed to initialize Anthropic client: {exc}") from exc

    @staticmethod
    def _wire_tools() -> List[Dict[str, Any]]:
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in TOOL_SPECS
        ]

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        self._system_prompt = system_prompt
        return [{"role": "user", "content": user_prompt}]

    def create_turn(self, messages: List[Dict[str, Any]]) -> Turn:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=self._system_prompt,
                messages=messages,
                tools=self._wire_tools(),
                temperature=0.2,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic API call failed: {exc}") from exc

        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "args": block.input or {}})
        return Turn(text="\n".join(text_parts).strip(), tool_calls=tool_calls, raw=response)

    def assistant_message(self, turn: Turn) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if turn.text:
            content.append({"type": "text", "text": turn.text})
        for tc in turn.tool_calls:
            content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
        return {"role": "assistant", "content": content}

    def tool_result_messages(self, turn: Turn, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content = [
            {"type": "tool_result", "tool_use_id": tc["id"], "content": json.dumps(result)}
            for tc, result in zip(turn.tool_calls, results)
        ]
        return [{"role": "user", "content": content}] if content else []


# ---------------------------------------------------------------------------
# Google Gemini (via the current `google-genai` SDK)
# ---------------------------------------------------------------------------

class GeminiProvider:
    name = "Google Gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "The 'google-genai' package is not installed. Run: pip install google-genai"
            ) from exc

        if not api_key or not api_key.strip():
            raise ProviderError("A Google API key is required but was not provided.")

        self.model = model
        self._types = types
        self._system_prompt = ""
        try:
            self._client = genai.Client(api_key=api_key)
        except Exception as exc:  # pragma: no cover
            raise ProviderError(f"Failed to initialize Gemini client: {exc}") from exc

    def _wire_tool(self):
        types = self._types
        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
            )
            for t in TOOL_SPECS
        ]
        return types.Tool(function_declarations=declarations)

    def build_initial_messages(self, system_prompt: str, user_prompt: str) -> List[Any]:
        """Seed the conversation as plain message dicts; `create_turn` converts
        them into Gemini `types.Content` objects before every API call."""
        self._system_prompt = system_prompt
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def create_turn(self, messages: List[Any]) -> Turn:
        types = self._types

        # --- Convert message dicts into proper types.Content objects ---
        # The google-genai SDK strictly prohibits passing raw dicts into `contents`.
        gemini_contents: List[Any] = []
        sys_instruct: str | None = self._system_prompt or None

        for m in messages:
            # Already a SDK-native Content (model turns, function responses, etc.)
            if isinstance(m, types.Content):
                gemini_contents.append(m)
                continue

            role = m.get("role", "user")
            content = m.get("content", "")

            if role == "system":
                sys_instruct = str(content)
            else:
                g_role = "model" if role == "assistant" else "user"
                gemini_contents.append(
                    types.Content(role=g_role, parts=[types.Part.from_text(text=str(content))])
                )

        if not gemini_contents:
            raise ProviderError("Gemini conversation history is empty after normalization.")

        config = types.GenerateContentConfig(
            system_instruction=sys_instruct,
            temperature=0.0,
            tools=[self._wire_tool()],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=gemini_contents,
                config=config,
            )
        except Exception as exc:
            raise ProviderError(f"Gemini API call failed: {exc}") from exc

        if not response.candidates:
            raise ProviderError("Gemini returned no candidates (the response may have been blocked).")

        content = response.candidates[0].content
        parts = content.parts or [] if content else []

        text_parts, tool_calls = [], []
        for i, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_calls.append({"id": f"call_{i}", "name": fc.name, "args": dict(fc.args or {})})
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        return Turn(text="\n".join(text_parts).strip(), tool_calls=tool_calls, raw=content)

    def assistant_message(self, turn: Turn) -> Any:
        types = self._types
        # Prefer the SDK-native Content object returned by Gemini (preserves
        # exact function_call parts). Fall back to reconstructing one ourselves.
        if isinstance(turn.raw, types.Content):
            return turn.raw

        parts: List[Any] = []
        if turn.text:
            parts.append(types.Part.from_text(text=turn.text))
        for tc in turn.tool_calls:
            parts.append(types.Part.from_function_call(name=tc["name"], args=tc["args"]))
        return types.Content(role="model", parts=parts)

    def tool_result_messages(self, turn: Turn, results: List[Dict[str, Any]]) -> List[Any]:
        types = self._types
        parts = [
            types.Part.from_function_response(name=tc["name"], response={"result": result})
            for tc, result in zip(turn.tool_calls, results)
        ]
        # Function responses are sent back as a user turn per Gemini's manual
        # tool-calling loop (see google-genai SDK docs).
        return [types.Content(role="user", parts=parts)] if parts else []


PROVIDERS = {
    "OpenAI": {
        "class": OpenAIProvider,
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"],
        "key_placeholder": "sk-...",
    },
    "Anthropic": {
        "class": AnthropicProvider,
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
        "key_placeholder": "sk-ant-...",
    },
    "Google Gemini (AI Studio)": {
        "class": GeminiProvider,
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5-flash"],
        "key_placeholder": "AIza...",
    },
}
