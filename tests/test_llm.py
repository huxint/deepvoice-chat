import pytest
import json

from voiceagent.llm import (
    ChatMessage,
    OpenAICompatibleChat,
    STRUCTURED_VOICE_CHAT_PROMPT,
    VOICE_CHAT_FUNCTION_NAME,
)


def test_deepseek_api_payload_contains_history():
    chat = OpenAICompatibleChat(api_key="test", model="deepseek-chat")
    payload = chat._payload(
        "你好",
        [ChatMessage("user", "上一轮"), ChatMessage("assistant", "上一轮回复")],
    )

    assert payload["model"] == "deepseek-chat"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][-1] == {"role": "user", "content": "你好"}
    assert payload["stream"] is False
    assert "tool_choice" not in payload
    assert payload["tools"][0]["function"]["parameters"]["required"] == [
        "spoken_text",
        "voice_prompt",
        "dialogue_state",
    ]


def test_system_prompt_uses_private_first_person_role_planning_without_leaking_cot():
    assert "first-person" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "dialogue_state" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "private reasoning hidden" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "never output chain-of-thought" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "required function call" in STRUCTURED_VOICE_CHAT_PROMPT


def test_tool_reply_uses_function_arguments():
    chat = OpenAICompatibleChat(api_key="test")
    reply = chat._tool_reply(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": VOICE_CHAT_FUNCTION_NAME,
                        "arguments": (
                            '{"spoken_text": "hello", '
                            '"voice_prompt": "warm", '
                            '"dialogue_state": "steady"}'
                        ),
                    }
                }
            ]
        }
    )

    assert reply.spoken_text == "hello"
    assert reply.voice_prompt == "warm"
    assert reply.dialogue_state == "steady"


def test_tool_reply_rejects_truncated_arguments():
    chat = OpenAICompatibleChat(api_key="test")
    with pytest.raises(json.JSONDecodeError):
        chat._tool_reply(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": VOICE_CHAT_FUNCTION_NAME,
                            "arguments": '{"spoken_text": "hello"',
                        }
                    }
                ]
            }
        )
