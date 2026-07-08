"""
05_finetuning_recipe.py — Complete LoRA fine-tuning recipe (structure only)
=============================================================================

This file contains the CANONICAL structure of a fine-tuning script.
Real execution needs a GPU — the code is the reference; run on Colab / your box.

PIPELINE
--------
    prepare_dataset()   ← JSONL → HF Dataset with 'messages' column
    train_lora()        ← SFTTrainer with LoRA/QLoRA config
    evaluate()          ← Run on held-out test set
    merge_and_save()    ← Combine adapter with base for deployment
    serve_via_vllm()    ← Multi-adapter serving instructions
"""

# NOTE: this file uses the transformers / peft / trl ecosystem. Install with:
#   pip install transformers peft trl bitsandbytes accelerate datasets

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 1. DATASET PREP
# ─────────────────────────────────────────────────────────────
def prepare_dataset(jsonl_path: str) -> "Dataset":
    """
    JSONL format expected — one JSON object per line:
        {"messages": [
            {"role": "system", "content": "You are ..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

    HF's chat template applies formatting at tokenization time.
    """
    from datasets import load_dataset
    ds = load_dataset("json", data_files=jsonl_path, split="train")

    # Basic sanity checks
    assert "messages" in ds.column_names, "Each row must have a 'messages' field"
    print(f"Loaded {len(ds)} examples")
    return ds


# ─────────────────────────────────────────────────────────────
# 2. TRAIN LORA
# ─────────────────────────────────────────────────────────────
def train_lora(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    train_ds=None,
    eval_ds=None,
    output_dir: str = "./lora-out",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig

    # QLoRA: 4-bit base, LoRA on top
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        max_seq_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Adapter saved to {output_dir}")


# ─────────────────────────────────────────────────────────────
# 3. EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate(model_dir: str, test_examples: list[dict]) -> dict:
    """Run through Phase 14 eval harness — expect a real difference vs base."""
    # Load model + tokenizer
    # For each test example: generate, compare vs expected
    # Return: pass_rate, avg_score
    return {"pass_rate": None}


# ─────────────────────────────────────────────────────────────
# 4. MERGE ADAPTER FOR DEPLOYMENT
# ─────────────────────────────────────────────────────────────
def merge_and_save(base_id: str, adapter_dir: str, out_dir: str):
    """Combine LoRA adapter with base weights into a single model.

    Use when you want a self-contained deployment (no adapter hot-swap).
    Skip if you plan to serve multiple adapters via vLLM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(base_id)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(out_dir)


# ─────────────────────────────────────────────────────────────
# 5. SERVE VIA vLLM
# ─────────────────────────────────────────────────────────────
"""
For multi-adapter serving (great for multi-tenant SaaS):

    vllm serve meta-llama/Meta-Llama-3-8B-Instruct \\
        --enable-lora \\
        --lora-modules customer1=./lora-c1 customer2=./lora-c2

Then in requests:
    POST /v1/chat/completions
    { "model": "customer1", "messages": [...] }

One base model, N adapters, N tenants — served from a single GPU.
"""


if __name__ == "__main__":
    print("""
This is a reference recipe. To run:

    # 1. Prepare your dataset as JSONL
    python prepare_data.py > train.jsonl

    # 2. Run LoRA training (GPU required)
    python 05_finetuning_recipe.py

    # 3. Evaluate on held-out test
    #    Use Phase 14 eval harness

    # 4. Deploy: either merge weights or serve via vLLM's LoRA module

Real fine-tuning requires:
    • 1 GPU with ≥ 24GB VRAM (RTX 3090/4090, A5000/A6000, A100)
    • 500-5000 labeled examples
    • 3-6 hours of training time (for 8B model, 3 epochs)
    • $10-50 in GPU rental (Runpod / Modal / Colab Pro)

Total budget to make a task-tuned 8B model: < $100.
""")
