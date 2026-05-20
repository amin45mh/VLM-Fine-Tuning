# VLM-Fine-Tuning — Qwen3-VL Video Rating

Fine-tune **Qwen3-VL-2B-Instruct** to rate video quality on a 0–100 scale using
QLoRA, Flash Attention 2, and DeepSpeed ZeRO-2.

## Architecture Decisions

| Component | Choice | Why |
|---|---|---|
| ViT | **Frozen** | Already strong; fine-tuning destabilizes features without massive data |
| Vision-Language Merger | **Frozen** | Same reasoning as ViT |
| LLM Decoder | **QLoRA** (4-bit NF4) | Fits RTX 5080 16 GB; targets `q/k/v/o/gate/up/down_proj` |
| Attention | **Flash Attention 2** | ~2× faster, lower VRAM |
| Distributed | **DeepSpeed ZeRO-2** | Shards optimizer + gradients; works single- and multi-GPU |
| Pixel Tokens | **token × 32 × 32** | Qwen3-VL resolution (Qwen2.5-VL used 28 × 28) |

## Data Format (ShareGPT Multimodal)

Each sample is a standard ShareGPT conversation. Videos are represented as a
chunk of frame images:

```json
[
    {
        "messages": [
            {"role": "system", "content": "You are a video quality assessment expert..."},
            {"role": "user", "content": [
                {"type": "video", "video": [
                    "data/frames/video_001/frame_001.jpg",
                    "data/frames/video_001/frame_002.jpg"
                ]},
                {"type": "text", "text": "Rate this video from 0 to 100."}
            ]},
            {"role": "assistant", "content": "85\n\nGood overall quality..."}
        ]
    }
]
```

Put your training data at `data/train.json` (see `data/sample_train.json` for a
full example).

## Setup

```bash
pip install torch torchvision
pip install transformers accelerate peft bitsandbytes
pip install deepspeed flash-attn --no-build-isolation
pip install qwen-vl-utils python-dotenv
```

Create a `.env` with your Hugging Face token:

```
HF_TOKEN=hf_...
```

## Prepare Your Frames

Organise video frames into directories:

```
data/
  frames/
    video_001/
      frame_001.jpg
      frame_002.jpg
      ...
    video_002/
      ...
  train.json
```

## Train

**Single GPU (RTX 5080):**

```bash
python Qwen3VL.py --mode train
```

**Multi-GPU with DeepSpeed:**

```bash
deepspeed --num_gpus=2 Qwen3VL.py --mode train
```

Or with Accelerate:

```bash
accelerate launch --config_file accelerate_config.yaml Qwen3VL.py --mode train
```

## Inference

```bash
# From a directory of frames
python Qwen3VL.py --mode infer --frames-dir data/frames/video_001/

# From explicit frame paths
python Qwen3VL.py --mode infer \
    --frames data/frames/video_001/frame_001.jpg data/frames/video_001/frame_002.jpg \
    --question "Rate this video from 0 to 100."
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `MIN_PIXELS` | `128 × 32 × 32` | Min pixels per frame |
| `MAX_PIXELS` | `512 × 32 × 32` | Max pixels per frame (lower = less VRAM) |
| `MAX_SEQ_LEN` | `4096` | Total sequence length (text + visual tokens) |
| `LORA_R` | `16` | LoRA rank |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `8` | Effective batch = `BATCH_SIZE × GRAD_ACCUM × num_gpus` |
