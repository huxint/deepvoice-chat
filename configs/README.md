# Configs

Run:

```bash
uv run voiceagent lora-config \
  --pretrained-path models/VoxCPM2 \
  --train-manifest data/manifests/train.jsonl \
  --output configs/voxcpm_lora.yaml
```

The generated YAML follows the upstream VoxCPM LoRA training script.
