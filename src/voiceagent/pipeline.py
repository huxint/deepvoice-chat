from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .audio import play_audio
from .llm import ChatMessage, StyledReply
from .tts import VoiceStyle, VoxCPMSynthesizer


@dataclass
class TurnResult:
    user_text: str
    assistant_text: str
    voice_prompt: str
    dialogue_state: str
    audio_path: str
    audio_duration_sec: float


@dataclass
class VoiceChatAgent:
    chat_backend: object
    synthesizer: VoxCPMSynthesizer
    style: VoiceStyle = field(default_factory=VoiceStyle)
    output_dir: Path = field(default_factory=lambda: Path("outputs/chat"))
    play: bool = False
    history: list[ChatMessage] = field(default_factory=list)

    def _reply_with_style(self, user_text: str) -> StyledReply:
        method = getattr(self.chat_backend, "reply_with_style", None)
        history = self._history_for_model()
        if method is not None:
            return method(user_text, history)
        return StyledReply(spoken_text=self.chat_backend.reply(user_text, history))

    def _history_for_model(self) -> list[ChatMessage]:
        context_parts = []
        if self.style.description.strip():
            context_parts.append(f"fixed base voice description: {self.style.description.strip()}")
        if self.style.reference_audio:
            context_parts.append("a fixed reference speaker audio is provided")
        if self.style.prompt_audio:
            context_parts.append("a fixed prompt audio and transcript are provided")
        if not context_parts:
            return list(self.history)

        context = (
            "Fixed local VoxCPM voice context: "
            + "; ".join(context_parts)
            + ". Generate voice_prompt only for turn-level delivery, emotion, pace, pauses, "
            "and emphasis. Keep it compatible with the fixed voice context and do not change "
            "speaker identity or base timbre."
        )
        return [ChatMessage("system", context), *self.history]

    def _turn_style(self, voice_prompt: str) -> VoiceStyle:
        parts = [self.style.description.strip(), voice_prompt.strip()]
        description = ", ".join(part for part in parts if part)
        return VoiceStyle(
            description=description,
            reference_audio=self.style.reference_audio,
            prompt_audio=self.style.prompt_audio,
            prompt_text=self.style.prompt_text,
        )

    def run_turn(self, user_text: str) -> TurnResult:
        styled_reply = self._reply_with_style(user_text)
        assistant_text = styled_reply.spoken_text
        voice_prompt = styled_reply.voice_prompt
        dialogue_state = styled_reply.dialogue_state
        turn_id = len([m for m in self.history if m.role == "assistant"]) + 1
        audio_path = self.output_dir / f"turn_{turn_id:03d}.wav"
        meta = self.synthesizer.synthesize_to_file(
            assistant_text,
            audio_path,
            style=self._turn_style(voice_prompt),
        )

        self.history.append(ChatMessage("user", user_text))
        self.history.append(
            ChatMessage(
                "assistant",
                json.dumps(
                    {
                        "spoken_text": assistant_text,
                        "voice_prompt": voice_prompt,
                        "dialogue_state": dialogue_state,
                    },
                    ensure_ascii=False,
                ),
            )
        )

        if self.play:
            play_audio(audio_path)

        return TurnResult(
            user_text=user_text,
            assistant_text=assistant_text,
            voice_prompt=voice_prompt,
            dialogue_state=dialogue_state,
            audio_path=str(audio_path),
            audio_duration_sec=float(meta["duration_sec"]),
        )
