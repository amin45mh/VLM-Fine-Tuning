"""
Qwen3-VL Video Rating Fine-Tuning (ShareGPT Multimodal Format)
===============================================================
Strategy:
  - Freeze the ViT (strong visual backbone, keep it stable)
  - Freeze the vision-language merger
  - QLoRA on the LLM decoder: q/k/v/o_proj, gate/up/down_proj
  - DeepSpeed ZeRO-2 for efficient training
  - Flash Attention 2 for memory-efficient attention
  - Video input as chunk of frame images

Data format (ShareGPT multimodal):
  [
      {
          "messages": [
              {"role": "system", "content": "..."},
              {"role": "user", "content": [
                  {"type": "video", "video": ["frame_01.jpg", "frame_02.jpg", ...]},
                  {"type": "text", "text": "Rate this video from 0 to 100."}
              ]},
              {"role": "assistant", "content": "85"}
          ]
      }
  ]
"""

import os
import json
import torch
from dataclasses import dataclass, field
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR = "./output/qwen3vl-video-rating"
DATA_PATH = "./data/train.json"
DS_CONFIG = "./ds_zero2.json"

SYSTEM_PROMPT = (
    "You are a video quality assessment expert. "
    "Watch the provided video carefully and rate it on a scale from 0 to 100, "
    "where 0 is the worst possible quality and 100 is perfect. "
    "Respond with the numerical score first, then a brief analysis."
)

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 4096

# Qwen3-VL uses 32x32 per visual token (NOT 28x28 which was Qwen2.5-VL)
MIN_PIXELS = 128 * 32 * 32   # 131_072
MAX_PIXELS = 512 * 32 * 32   # 524_288  (conservative for video on RTX 5080 16 GB)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VideoRatingDataset(torch.utils.data.Dataset):
    """
    ShareGPT-style multimodal dataset for video rating.

    Each sample is a dict with a ``messages`` list following the standard
    ShareGPT multi-turn format.  Vision content uses Qwen-VL content items::

        {"type": "video", "video": ["frame_01.jpg", ...]}
        {"type": "image", "image": "path/to/img.jpg"}

    The last message must be the assistant response containing the rating.
    """

    def __init__(self, data_path: str, processor, max_len: int = MAX_SEQ_LEN):
        with open(data_path) as f:
            self.samples = json.load(f)
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]

        prompt_messages = messages[:-1]

        image_inputs, video_inputs = process_vision_info(messages)

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        vision_kwargs: dict = {}
        if image_inputs:
            vision_kwargs["images"] = image_inputs
        if video_inputs:
            vision_kwargs["videos"] = video_inputs

        full_inputs = self.processor(
            text=[full_text],
            **vision_kwargs,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            **vision_kwargs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[-1]
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        for key in (
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
            "mm_token_type_ids",
        ):
            if key in full_inputs:
                val = full_inputs[key]
                if key == "mm_token_type_ids":
                    result[key] = val.squeeze(0)
                else:
                    result[key] = val

        return result


# ─── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class VLMCollator:
    """Pads text-like tensors and concatenates vision tensors across the batch."""

    pad_token_id: int = 0

    _SEQ_KEYS: list = field(
        default_factory=lambda: ["mm_token_type_ids"],
        repr=False,
    )
    _VISION_KEYS: list = field(
        default_factory=lambda: [
            "pixel_values", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw",
        ],
        repr=False,
    )

    def __call__(self, examples):
        max_len = max(ex["input_ids"].shape[0] for ex in examples)

        input_ids, attention_masks, labels_list = [], [], []
        for ex in examples:
            pad_len = max_len - ex["input_ids"].shape[0]
            input_ids.append(torch.cat([
                ex["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=ex["input_ids"].dtype),
            ]))
            attention_masks.append(torch.cat([
                ex["attention_mask"],
                torch.zeros(pad_len, dtype=ex["attention_mask"].dtype),
            ]))
            labels_list.append(torch.cat([
                ex["labels"],
                torch.full((pad_len,), -100, dtype=ex["labels"].dtype),
            ]))

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list),
        }

        for key in self._SEQ_KEYS:
            if key in examples[0]:
                padded = []
                for ex in examples:
                    pad_len = max_len - ex[key].shape[0]
                    padded.append(torch.cat([
                        ex[key],
                        torch.zeros(pad_len, dtype=ex[key].dtype),
                    ]))
                batch[key] = torch.stack(padded)

        for key in self._VISION_KEYS:
            tensors = [ex[key] for ex in examples if key in ex]
            if tensors:
                batch[key] = torch.cat(tensors, dim=0)

        return batch


# ─── Model Setup ──────────────────────────────────────────────────────────────

def build_model_and_processor():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    for name, param in model.named_parameters():
        if "visual" in name or "merger" in name:
            param.requires_grad = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    model, processor = build_model_and_processor()

    dataset = VideoRatingDataset(DATA_PATH, processor)
    collator = VLMCollator(pad_token_id=processor.tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=DS_CONFIG,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


# ─── Inference ────────────────────────────────────────────────────────────────

def inference(frame_paths: list[str], question: str | None = None):
    """Load the fine-tuned LoRA adapter and rate a video."""
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(
        OUTPUT_DIR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    if question is None:
        question = "Rate this video from 0 to 100."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "video", "video": frame_paths},
            {"type": "text", "text": question},
        ]},
    ]

    image_inputs, video_inputs = process_vision_info(messages)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    proc_kwargs: dict = {"text": [text], "return_tensors": "pt"}
    if image_inputs:
        proc_kwargs["images"] = image_inputs
    if video_inputs:
        proc_kwargs["videos"] = video_inputs

    inputs = processor(**proc_kwargs).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    print(f"\n{'=' * 60}")
    print(f"Rating: {response}")
    print(f"{'=' * 60}")
    return response


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Qwen3-VL Video Rating Fine-Tuning",
    )
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument(
        "--frames", type=str, nargs="+",
        help="Frame image paths for inference (space-separated)",
    )
    parser.add_argument(
        "--frames-dir", type=str,
        help="Directory of frame images (sorted alphabetically)",
    )
    parser.add_argument(
        "--question", type=str,
        default="Rate this video from 0 to 100.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        if args.frames_dir:
            frame_dir = Path(args.frames_dir)
            frame_paths = sorted(
                str(p) for p in frame_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
        elif args.frames:
            frame_paths = args.frames
        else:
            parser.error("--frames or --frames-dir required for inference mode")

        inference(frame_paths, args.question)
