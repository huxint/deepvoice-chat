import json

from voiceagent.pipeline import VoiceChatAgent
from voiceagent.tts import VoiceStyle


class FakeChat:
    def __init__(self):
        self.histories = []

    def reply_with_style(self, user_text, history):
        from voiceagent.llm import StyledReply

        self.histories.append(list(history))
        return StyledReply(
            spoken_text=f"reply to {user_text}",
            voice_prompt="calm tone, medium pace",
            dialogue_state="I stay calm and remember the user wants continuity.",
        )


class FakeSynth:
    def __init__(self):
        self.calls = []

    def synthesize_to_file(self, text, output_path, *, style):
        self.calls.append((text, output_path, style))
        return {"duration_sec": 1.0}


def test_pipeline_keeps_voice_prompt_in_assistant_history(tmp_path):
    chat = FakeChat()
    synth = FakeSynth()
    agent = VoiceChatAgent(
        chat_backend=chat,
        synthesizer=synth,
        style=VoiceStyle(description="fixed base voice"),
        output_dir=tmp_path,
    )

    first = agent.run_turn("hello")
    second = agent.run_turn("continue")

    assert first.voice_prompt == "calm tone, medium pace"
    assert first.dialogue_state == "I stay calm and remember the user wants continuity."
    assert second.audio_path.endswith("turn_002.wav")

    style_context = chat.histories[0][0]
    assert style_context.role == "system"
    assert "fixed base voice description: fixed base voice" in style_context.content

    previous_assistant = chat.histories[1][2]
    payload = json.loads(previous_assistant.content)
    assert payload == {
        "spoken_text": "reply to hello",
        "voice_prompt": "calm tone, medium pace",
        "dialogue_state": "I stay calm and remember the user wants continuity.",
    }

    _, _, style = synth.calls[0]
    assert style.description == "fixed base voice, calm tone, medium pace"
