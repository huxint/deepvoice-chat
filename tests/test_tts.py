from voiceagent.tts import apply_voice_control


def test_apply_voice_control_wraps_plain_text():
    assert apply_voice_control("  你好   世界 ", "年轻女声") == "(年轻女声)你好 世界"


def test_apply_voice_control_preserves_existing_control():
    assert apply_voice_control("(沉稳男声)你好", "年轻女声") == "(沉稳男声)你好"
