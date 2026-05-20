"""
Qwen3-VL-2B Gym Coach Fine-Tuning
==================================
Strategy:
  - Freeze the ViT (strong visual backbone, keep it stable)
  - Freeze the vision-language merger
  - QLoRA on the LLM decoder: q/k/v/o_proj, gate/up/down_proj
"""

import os
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR = "./output/qwen3vl-gym-coach"
DATA_PATH = "./data/train.json"

SYSTEM_PROMPT = (
    "You are an expert gym coach and certified personal trainer. "
    "Analyze the user's exercise form from the provided image, identify "
    "mistakes, and provide clear, actionable corrections. Suggest "
    "appropriate modifications based on the user's level and always "
    "prioritize safety and proper technique."
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
MAX_SEQ_LEN = 2048

# Qwen VL dynamic resolution bounds
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GymCoachDataset(torch.utils.data.Dataset):
    """
    Expects a JSON file:
    [
        {
            "image": "data/images/squat_001.jpg",
            "question": "Analyze my squat form.",
            "answer": "Your squat shows slight knee valgus..."
        },
        ...
    ]
    The "image" key is optional; text-only samples are supported.
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
        has_image = "image" in sample and sample["image"]

        image = None
        if has_image:
            image = Image.open(sample["image"]).convert("RGB")
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["question"]},
            ]
        else:
            user_content = [{"type": "text", "text": sample["question"]}]

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["answer"]},
        ]
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        images = [image] if has_image else None

        full_inputs = self.processor(
            text=[full_text],
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_len,
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=images,
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
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"]

        return result


# ─── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class VLMCollator:
    """Pads text sequences and concatenates vision tensors across the batch."""

    pad_token_id: int = 0

    def __call__(self, examples):
        max_len = max(ex["input_ids"].shape[0] for ex in examples)

        input_ids, attention_masks, labels_list = [], [], []
        for ex in examples:
            seq_len = ex["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids.append(
                torch.cat([ex["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=ex["input_ids"].dtype)])
            )
            attention_masks.append(
                torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=ex["attention_mask"].dtype)])
            )
            labels_list.append(
                torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=ex["labels"].dtype)])
            )

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels_list),
        }
        if "pixel_values" in examples[0]:
            batch["pixel_values"] = torch.cat([ex["pixel_values"] for ex in examples], dim=0)
        if "image_grid_thw" in examples[0]:
            batch["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in examples], dim=0)

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
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Freeze ViT and vision-language merger
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

    dataset = GymCoachDataset(DATA_PATH, processor)
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

def inference(image_path: str, question: str):
    """Load the fine-tuned LoRA adapter and run a single prediction."""
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

    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Coach: {response}")
    print(f"{'='*60}")
    return response


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-VL-2B Gym Coach Fine-Tuning")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--image", type=str, help="Image path (inference mode)")
    parser.add_argument(
        "--question", type=str,
        default="Analyze my exercise form and suggest improvements.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        if not args.image:
            parser.error("--image is required for inference mode")
        inference(args.image, args.question)
