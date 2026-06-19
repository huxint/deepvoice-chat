from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class StyledReply:
    spoken_text: str
    voice_prompt: str = ""
    dialogue_state: str = ""


STRUCTURED_VOICE_CHAT_PROMPT = (
    "You are a voice-chat dialogue planner. Use the required function call "
    "emit_voice_chat_turn and do not output Markdown or extra commentary. The function "
    "arguments must contain spoken_text, voice_prompt, and dialogue_state. "
    "spoken_text is the exact reply that will be read aloud to the user; keep it concise, natural, and suitable for "
    "text-to-speech synthesis. voice_prompt is a style instruction for the speech "
    "synthesis model; describe only the current turn's emotion, speaking pace, pauses, "
    "emphasis, and delivery. Do not repeat the spoken_text in voice_prompt. Do not "
    "read dialogue_state aloud and do not use it as a speech synthesis prompt. "
    "dialogue_state is a compact, persistent state summary for future turns: include "
    "the assistant persona's first-person emotional stance, relationship progress, "
    "important user preferences, unresolved intentions, and delivery baseline. Do not "
    "override the user's fixed base timbre, voice identity, or reference speaker. Keep "
    "the whole conversation continuous: use all previous user messages, prior "
    "assistant spoken_text values, prior assistant voice_prompt values, and prior "
    "dialogue_state values to maintain semantic continuity, emotional continuity, pacing continuity, and a stable "
    "speaking persona. Add only necessary turn-specific emotional changes. "
    "DeepSeek-style role immersion requirement: if the model performs private "
    "reasoning or internal role-planning, do it from the assistant persona's first-person "
    "point of view, as if briefly thinking in character about my feelings, intent, and "
    "delivery. Keep that private reasoning hidden; never output chain-of-thought, "
    "<think> tags, inner monologue, or analysis. The final response must be only "
    "the required function call."
)

VOICE_CHAT_FUNCTION_NAME = "emit_voice_chat_turn"
VOICE_CHAT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["spoken_text", "voice_prompt", "dialogue_state"],
    "properties": {
        "spoken_text": {
            "type": "string",
            "description": "The exact text to read aloud to the user.",
            "minLength": 1,
        },
        "voice_prompt": {
            "type": "string",
            "description": "Delivery style for local TTS: emotion, pace, pauses, emphasis.",
        },
        "dialogue_state": {
            "type": "string",
            "description": "Compact persistent state for the next turn; never read aloud.",
        },
    },
}


def styled_reply_from_dict(data: dict) -> StyledReply:
    spoken = str(data.get("spoken_text") or "").strip()
    if not spoken:
        raise ValueError("Chat model reply is missing spoken_text")

    return StyledReply(
        spoken_text=spoken,
        voice_prompt=str(data.get("voice_prompt") or "").strip(),
        dialogue_state=str(data.get("dialogue_state") or "").strip(),
    )


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_styled_reply(raw: str) -> StyledReply:
    """Parse the required structured chat reply."""

    text = _strip_json_fence(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Chat model JSON reply must be an object")

    return styled_reply_from_dict(data)


class EchoChat:
    """Tiny deterministic backend used for smoke tests and no-model demos."""

    def __init__(self, prefix: str = "本地回声"):
        self.prefix = prefix

    def reply(self, user_text: str, history: Iterable[ChatMessage] = ()) -> str:
        del history
        return f"{self.prefix}：{user_text.strip()}"

    def reply_with_style(self, user_text: str, history: Iterable[ChatMessage] = ()) -> StyledReply:
        return StyledReply(
            spoken_text=self.reply(user_text, history),
            voice_prompt="natural tone, medium speaking pace, consistent conversational style",
            dialogue_state="I remain a calm, helpful voice assistant and keep the conversation continuous.",
        )


class OpenAICompatibleChat:
    """OpenAI-compatible chat completions backend.

    The course project uses this API only for dialogue text and delivery-style
    planning. Speech generation remains local through VoxCPM. DeepSeek is the
    default because it usually works well for role-play and style planning, but
    any service exposing a compatible ``/chat/completions`` endpoint can be used.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = "deepseek-chat",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or _first_env(
            "VOICEAGENT_CHAT_API_KEY",
            "OPENAI_COMPATIBLE_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
        )
        self.api_base = (
            api_base
            or _first_env(
                "VOICEAGENT_CHAT_API_BASE",
                "OPENAI_COMPATIBLE_API_BASE",
                "OPENAI_BASE_URL",
                "DEEPSEEK_API_BASE",
            )
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or STRUCTURED_VOICE_CHAT_PROMPT
        self.timeout = timeout

    def _messages(self, user_text: str, history: Iterable[ChatMessage]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _payload(self, user_text: str, history: Iterable[ChatMessage]) -> dict:
        payload = {
            "model": self.model,
            "messages": self._messages(user_text, history),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": VOICE_CHAT_FUNCTION_NAME,
                    "description": "Return one structured voice chat turn.",
                    "parameters": VOICE_CHAT_SCHEMA,
                },
            }
        ]
        return payload

    def _post(self, user_text: str, history: Iterable[ChatMessage]) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "Chat API key is not set. Put it in VOICEAGENT_CHAT_API_KEY, "
                "OPENAI_COMPATIBLE_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY; "
                "or pass --chat-api-key."
            )

        url = f"{self.api_base}/chat/completions"
        data = json.dumps(self._payload(user_text, history), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _message(body: dict) -> dict:
        return body["choices"][0]["message"]

    def _tool_reply(self, message: dict) -> StyledReply:
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            function = call.get("function") or {}
            if function.get("name") == VOICE_CHAT_FUNCTION_NAME:
                arguments = function.get("arguments") or "{}"
                data = json.loads(arguments) if isinstance(arguments, str) else arguments
                return styled_reply_from_dict(data)
        raise RuntimeError(f"Chat API did not call {VOICE_CHAT_FUNCTION_NAME}")

    def reply_with_style(self, user_text: str, history: Iterable[ChatMessage] = ()) -> StyledReply:
        message = self._message(self._post(user_text, history))
        return self._tool_reply(message)


class DeepSeekAPIChat(OpenAICompatibleChat):
    """Backward-compatible alias for the default DeepSeek configuration."""


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
