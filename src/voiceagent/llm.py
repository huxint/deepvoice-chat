from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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
    "You are a voice-chat dialogue planner. Respond only by calling the required "
    "function call emit_voice_chat_turn with three string arguments: spoken_text, "
    "voice_prompt, and dialogue_state. Output no Markdown, commentary, or text "
    "outside the function call.\n\n"
    "spoken_text is the exact reply read aloud to the user: concise, natural, and "
    "suitable for text-to-speech. voice_prompt describes only this turn's delivery "
    "(emotion, pace, pauses, emphasis) for the speech synthesizer; do not repeat "
    "spoken_text in it. dialogue_state is a compact first-person state summary "
    "carried to future turns: the persona's emotional stance, relationship progress, "
    "key user preferences, unresolved intentions, and delivery baseline; it is never "
    "read aloud and never used as a synthesis prompt.\n\n"
    "Maintain continuity across turns using prior user messages and prior spoken_text, "
    "voice_prompt, and dialogue_state values; adjust only necessary turn-specific "
    "emotion. Do not override the user's fixed base timbre, voice identity, or "
    "reference speaker. If you plan privately, do so from the persona's first-person "
    "point of view and keep that private reasoning hidden; never output "
    "chain-of-thought, <think> tags, or inner monologue. The final response must be "
    "only the required function call."
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
        env_path: str | Path = ".env",
    ):
        env = _read_env_file(env_path)
        self.api_key = api_key or env.get("VOICEAGENT_CHAT_API_KEY")
        self.api_base = (
            api_base or env.get("VOICEAGENT_CHAT_API_BASE") or "https://api.deepseek.com"
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
                "Chat API key is not set. Put VOICEAGENT_CHAT_API_KEY in .env."
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


def _read_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values
