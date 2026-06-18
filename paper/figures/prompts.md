# Image2 科研图占位提示词

检索到的科研图提示词经验可以概括为：先明确科学目标和结构，再指定面板布局、实体、箭头关系、字体、色盲友好配色和白底出版风格；多面板图保持相同字体、线宽和色号。以下提示词按这个结构写好，生成后把图片保存到 `paper/figures/` 并在 `main.tex` 中替换占位图。

## Prompt 1: 系统流程图

Create a publication-ready scientific workflow diagram for a voice chat system. White background, clean vector-like style, 16:9 aspect ratio, colorblind-safe palette using blue #0072B2, orange #E69F00, green #009E73, gray #666666. Left-to-right pipeline with modules: User Input, OpenAI-Compatible Chat API (DeepSeek by default), Voice Control Layer, Local VoxCPM / VoxCPM2 Speech Synthesizer, Audio Playback. Inside the chat API module show two outputs: spoken_text and voice_prompt. spoken_text goes to the synthesizer as readable content; voice_prompt goes to the Voice Control Layer and is not spoken aloud. Show conversation history feeding back into the chat API module. Show optional reference speaker audio and optional LoRA adapter entering the local VoxCPM module from below. Use thin rounded rectangles, consistent 1.5 pt arrows, sans-serif labels, no decorative background, no 3D effects. Add a small note inside the VoxCPM block: "local model: voice design / cloning / LoRA". Ensure all text is legible and correctly spelled.

## Prompt 2: VoxCPM 层次化语音生成示意

Create a clean academic architecture diagram illustrating tokenizer-free hierarchical speech synthesis in continuous latent space. White background, two-row layout. Top row: Text Prompt and Voice Description tokens entering a Text-Semantic Language Model. Middle: semantic-prosodic plan flows into Residual Acoustic Language Model. Bottom: Local DiT / flow matching decoder generates continuous audio latents, then AudioVAE decoder outputs waveform. Include an optional Reference Audio branch encoded by Local Encoder and connected to the conditioning stream. Use blue for text/semantic modules, orange for acoustic modules, green for waveform output, gray for optional conditioning. Use precise arrows, simple blocks, no photorealistic elements, no meaningless icons. Style should resemble a NeurIPS method figure.

## Prompt 3: 实验评价图

Create a publication-ready multi-panel evaluation figure template for a speech synthesis course project. White background, 2x2 panels with consistent margins. Panel A: bar chart placeholder for RTF comparison across VoxCPM2 CPU, VoxCPM-0.5B, VoxCPM2 LoRA. Panel B: radar chart placeholder for subjective ratings: intelligibility, naturalness, speaker similarity, style match. Panel C: waveform and mel-spectrogram schematic of generated speech. Panel D: ablation diagram showing inference timesteps 4, 6, 10, 20 and quality-speed tradeoff. Use colorblind-safe colors #0072B2, #E69F00, #009E73, #CC79A7. Keep labels large and editable-looking, avoid fake numerical values, use "TBD" placeholders where data is not available.

## Prompt 4: 课程论文 graphical abstract

Create a concise graphical abstract for a paper titled "Chat API + Local VoxCPM Voice Chat". White background, horizontal composition. Show a cloud chat API box connected to a laptop running a local speech model, a text bubble transformed into a speech waveform, and three controllable voice options represented as small labeled sliders: timbre, emotion, speed. Include small badges "DeepSeek by default", "OpenAI-compatible API", "Local VoxCPM2 TTS", "LoRA Adaptation". Use flat scientific illustration style, restrained colors, no cartoon character, no stock-photo look, no dark gradient, no unnecessary decoration. Text must be sharp and correctly spelled.
