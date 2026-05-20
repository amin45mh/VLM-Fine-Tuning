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

## Pipeline Overview

```
raw videos + annotations ──► preprocess ──► ShareGPT JSON + frames
                                                │
                                                ▼
                                             train
                                                │
                                                ▼
              new video ──────────────► infer (auto-extracts frames)
                                                │
                                                ▼
                                          0-100 rating
```

## Setup

```bash
pip install torch torchvision
pip install transformers accelerate peft bitsandbytes
pip install deepspeed flash-attn --no-build-isolation
pip install qwen-vl-utils python-dotenv opencv-python
```

Create a `.env` with your Hugging Face token:

```
HF_TOKEN=hf_...
```

## Step 1 — Prepare Annotations

Create an annotations JSON listing your raw video files with their target
ratings:

```json
[
    {
        "video": "raw_videos/clip_001.mp4",
        "rating": 85,
        "analysis": "Good overall quality with stable framing..."
    },
    {
        "video": "raw_videos/clip_002.mp4",
        "rating": 42,
        "analysis": "Heavy compression artifacts, shaky footage..."
    }
]
```

See `data/sample_annotations.json` for a full example.

## Step 2 — Preprocess (Extract Frames)

This reads your raw videos, extracts frames at the target FPS (default 6),
and writes the ShareGPT-format training JSON automatically:

```bash
python Qwen3VL.py --mode preprocess \
    --annotations data/annotations.json \
    --fps 6
```

Output:
- Extracted frames saved to `data/frames/<video_stem>/frame_000001.jpg, ...`
- Training JSON written to `data/train.json`

You can customise the output paths:

```bash
python Qwen3VL.py --mode preprocess \
    --annotations data/annotations.json \
    --frames-root data/frames \
    --output-json data/train.json \
    --fps 6
```

## Step 3 — Train

**Single GPU (RTX 5080):**

```bash
python Qwen3VL.py --mode train
```

**Multi-GPU with DeepSpeed:**

```bash
deepspeed --num_gpus=2 Qwen3VL.py --mode train
```

## Step 4 — Inference

Just point at a raw video file — frames are extracted automatically:

```bash
python Qwen3VL.py --mode infer --video path/to/video.mp4
```

You can control the extraction FPS:

```bash
python Qwen3VL.py --mode infer --video path/to/video.mp4 --fps 6
```

Or use pre-extracted frames directly:

```bash
# From a directory
python Qwen3VL.py --mode infer --frames-dir data/frames/clip_001/

# From explicit paths
python Qwen3VL.py --mode infer \
    --frames frame_01.jpg frame_02.jpg frame_03.jpg
```

## Data Format (ShareGPT Multimodal)

The `preprocess` step generates this format automatically. Each sample is a
standard ShareGPT conversation where videos are chunks of frame images:

```json
{
    "messages": [
        {"role": "system", "content": "You are a video quality assessment expert..."},
        {"role": "user", "content": [
            {"type": "video", "video": [
                "data/frames/clip_001/frame_000001.jpg",
                "data/frames/clip_001/frame_000002.jpg"
            ]},
            {"type": "text", "text": "Rate this video from 0 to 100."}
        ]},
        {"role": "assistant", "content": "85\n\nGood overall quality..."}
    ]
}
```

## Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `EXTRACT_FPS` | `6` | Frames per second for video extraction |
| `MIN_PIXELS` | `128 × 32 × 32` | Min pixels per frame |
| `MAX_PIXELS` | `512 × 32 × 32` | Max pixels per frame (lower = less VRAM) |
| `MAX_SEQ_LEN` | `4096` | Total sequence length (text + visual tokens) |
| `LORA_R` | `16` | LoRA rank |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACCUM` | `8` | Effective batch = `BATCH_SIZE × GRAD_ACCUM × num_gpus` |
