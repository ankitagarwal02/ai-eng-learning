# LoRA / QLoRA / PEFT — Adapter Fine-Tuning

## The problem full fine-tuning has

Llama 3 8B has ~8 billion parameters. Full fine-tuning updates ALL of them.

- Storage: 32GB per checkpoint (fp32) × dozens of checkpoints
- GPU RAM: needs 8× A100 minimum for 8B model
- Cost: $$$$

Every downstream task requires a fresh copy of the whole model. Bad.

## LoRA — the trick

Instead of updating full weight matrix W (size dxd), decompose the *update*
ΔW as a product of two small matrices:

```
   ΔW  ≈  A × B     where A is d×r  and  B is r×d,  r << d
```

Typical r = 8 to 64 (called the "rank").

For an 8B model with 4096-dim layers:
- Full ΔW: 4096 × 4096 = 16.7M params per layer
- LoRA (r=16): 4096×16 + 16×4096 = 131k params per layer
- **~128× fewer parameters trained**

At inference, ΔW is added to W (or kept separate as an adapter). Multiple
adapters can be loaded/unloaded per request → one base model serves 10 use cases.

## QLoRA — LoRA on 4-bit quantized base

Same math, but the frozen base weights are stored in 4-bit precision (via
`bitsandbytes`). Result: fits a 65B model on a single 48GB GPU. Quality is
within 1-2% of full-precision LoRA on most benchmarks.

## PEFT (Parameter-Efficient Fine-Tuning) — the library

`peft` is Hugging Face's library that implements LoRA, QLoRA, Prefix Tuning,
IA³, and more. It wraps your model with adapters via a config.

## Minimum viable LoRA training script

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit=True,          # QLoRA
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=your_dataset,   # HF Dataset with 'text' column
    args=TrainingArguments(
        output_dir="./out",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
    ),
)
trainer.train()
model.save_pretrained("./my_lora_adapter")
```

## Inference: load base + adapter

```python
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = PeftModel.from_pretrained(base, "./my_lora_adapter")
# ... generate as usual
```

## Serving multiple adapters efficiently

**vLLM's LoRAServing feature** loads the base once, hot-swaps adapters per
request. Serve 20 fine-tuned variants from a single 80GB GPU.

## When LoRA fails

- **Very small rank (r=2)** — can't capture enough signal → poor quality.
- **Wrong target modules** — must include attention projections for meaningful learning.
- **Learning rate too high** — LoRA is sensitive; typical 1e-4 to 5e-4.
- **Data too small** — <500 examples: prompt-engineer instead.
- **Task incompatible with base** — fine-tuning a chat model to do image recognition doesn't work.
