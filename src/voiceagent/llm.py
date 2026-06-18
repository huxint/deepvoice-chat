from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["system", "user", "assistant"]
StructuredOutputMode = Literal["tool", "json_object"]


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
    "You are a voice-chat dialogue planner. Always output exactly one JSON object "
    "and do not output Markdown or extra commentary. The JSON object must contain "
    "three string fields: spoken_text, voice_prompt, and dialogue_state. spoken_text is the exact reply "
    "that will be read aloud to the user; keep it concise, natural, and suitable for "
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
    "<think> tags, inner monologue, or analysis. The final response must still be only "
    "the JSON object."
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
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Chat model did not return valid JSON: {text}") from exc

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
        structured_output: StructuredOutputMode = "tool",
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
        self.structured_output = structured_output
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
        if self.structured_output == "tool":
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
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": VOICE_CHAT_FUNCTION_NAME},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Chat API request failed: {exc.reason}") from exc

        return body

    @staticmethod
    def _message(body: dict) -> dict:
        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat API response: {body}") from exc

    def _tool_reply(self, message: dict) -> StyledReply:
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            function = call.get("function") or {}
            if function.get("name") == VOICE_CHAT_FUNCTION_NAME:
                arguments = function.get("arguments") or "{}"
                data = json.loads(arguments) if isinstance(arguments, str) else arguments
                return styled_reply_from_dict(data)
        raise RuntimeError(f"Chat API did not call {VOICE_CHAT_FUNCTION_NAME}")

    def reply(self, user_text: str, history: Iterable[ChatMessage] = ()) -> str:
        message = self._message(self._post(user_text, history))
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Chat API response has no text content: {message}")
        return content.strip()

    def reply_with_style(self, user_text: str, history: Iterable[ChatMessage] = ()) -> StyledReply:
        message = self._message(self._post(user_text, history))
        if self.structured_output == "tool":
            return self._tool_reply(message)
        return parse_styled_reply(message.get("content") or "")


class DeepSeekAPIChat(OpenAICompatibleChat):
    """Backward-compatible alias for the default DeepSeek configuration."""


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


class LocalDeepSeekChat:
    """Local Hugging Face causal-LM chat wrapper for DeepSeek-family models."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt or STRUCTURED_VOICE_CHAT_PROMPT
        self._tokenizer = None
        self._model = None

    def _torch_dtype(self):
        import torch

        if self.dtype == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return mapping[self.dtype]

    def load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        device_map = "auto" if self.device == "auto" and use_cuda else None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self._torch_dtype(),
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if device_map is None:
            target_device = "cuda" if use_cuda and self.device in {"auto", "cuda"} else self.device
            if target_device == "auto":
                target_device = "cpu"
            self._model.to(target_device)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_messages(self, user_text: str, history: Iterable[ChatMessage]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend({"role": m.role, "content": m.content} for m in history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def reply(self, user_text: str, history: Iterable[ChatMessage] = ()) -> str:
        self.load()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        messages = self._format_messages(user_text, history)
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        model_device = next(self._model.parameters()).device
        input_ids = input_ids.to(model_device)
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def reply_with_style(self, user_text: str, history: Iterable[ChatMessage] = ()) -> StyledReply:
        return parse_styled_reply(self.reply(user_text, history))
