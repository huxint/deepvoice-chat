from voiceagent.llm import ChatMessage, OpenAICompatibleChat, STRUCTURED_VOICE_CHAT_PROMPT


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


def test_system_prompt_uses_private_first_person_role_planning_without_leaking_cot():
    assert "first-person" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "dialogue_state" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "private reasoning hidden" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "never output chain-of-thought" in STRUCTURED_VOICE_CHAT_PROMPT
    assert "JSON object" in STRUCTURED_VOICE_CHAT_PROMPT
