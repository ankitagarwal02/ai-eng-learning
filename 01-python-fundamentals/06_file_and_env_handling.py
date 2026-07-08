"""
06_file_and_env_handling.py — .env, pathlib, logging, argparse
===============================================================

WHAT THIS FILE TEACHES
----------------------
• os.getenv() and python-dotenv for secrets/config.
• pathlib.Path — modern, cross-platform file paths.
• Reading/writing JSON, CSV, and text files.
• The `logging` module (levels, formatters, file handlers).
• argparse for CLI scripts.

HOW TO RUN
----------
    python 06_file_and_env_handling.py --input sample_input.csv --model gpt-4o-mini
    python 06_file_and_env_handling.py --help

REAL-WORLD SCENARIO
-------------------
A batch-scoring script:
    1. Load API key from .env
    2. Load a prompt template from a .txt file
    3. Read CSV of inputs
    4. Call the (mock) LLM for each row
    5. Write results.csv
    6. Log everything with timestamps to console + rotating file
"""

import os
import csv
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# python-dotenv is optional — fall back gracefully if not installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MOCK_MODE = not os.getenv("OPENAI_API_KEY")


# ─────────────────────────────────────────────────────────────
# SECTION 1: os.getenv() + .env pattern
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Environment variables")
print("=" * 70)

# Pattern used by every AI project — read secrets from env, never hardcode:
API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o-mini")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

print(f"MOCK_MODE:      {MOCK_MODE}")
print(f"DEFAULT_MODEL:  {DEFAULT_MODEL}")
print(f"LOG_LEVEL:      {LOG_LEVEL}")
print(f"API_KEY set:    {bool(API_KEY)}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: pathlib — the modern way to do file paths
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2: pathlib")
print("=" * 70)

here = Path(__file__).resolve().parent
print(f"This file lives in: {here}")

# Build paths portably (works on Windows, Linux, Mac):
data_dir = here / "data"
data_dir.mkdir(exist_ok=True)   # like mkdir -p

sample_csv = data_dir / "sample_input.csv"
prompt_file = data_dir / "prompt_template.txt"
results_csv = data_dir / "results.csv"

# Check existence, extensions, filename parts:
print(f"data_dir exists?     {data_dir.exists()}")
print(f"sample_csv suffix:   {sample_csv.suffix}")
print(f"sample_csv stem:     {sample_csv.stem}")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Write sample data (setup)
# ─────────────────────────────────────────────────────────────
sample_rows = [
    {"id": 1, "text": "My subscription was double-charged."},
    {"id": 2, "text": "The site is down since 2pm."},
    {"id": 3, "text": "How do I export my data?"},
    {"id": 4, "text": "Feature request: dark mode."},
    {"id": 5, "text": "I love this product!"},
]

with open(sample_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text"])
    writer.writeheader()
    writer.writerows(sample_rows)
print(f"Wrote sample CSV to: {sample_csv}")

prompt_template = "Classify this customer message into one of: BILLING, OUTAGE, HOWTO, FEATURE, PRAISE.\n\nMessage: {text}\nCategory:"
prompt_file.write_text(prompt_template, encoding="utf-8")
print(f"Wrote prompt template to: {prompt_file}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Read a text file (the prompt template)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4: Reading files")
print("=" * 70)

# Modern one-liner:
template_text = prompt_file.read_text(encoding="utf-8")
print(f"Prompt template ({len(template_text)} chars):")
print(template_text)


# ─────────────────────────────────────────────────────────────
# SECTION 5: `logging` — real logs, not print()
# ─────────────────────────────────────────────────────────────
def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """The canonical logging setup.

    Console handler (colored-ish) + optional file handler.
    """
    logger = logging.getLogger("batch_scorer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────
# SECTION 6: Mock LLM scoring loop
# ─────────────────────────────────────────────────────────────
def mock_classify(text: str) -> str:
    """Deterministic mock classifier — good enough to demonstrate the pipeline."""
    text_l = text.lower()
    if "charge" in text_l or "bill" in text_l or "refund" in text_l:
        return "BILLING"
    if "down" in text_l or "outage" in text_l or "not working" in text_l:
        return "OUTAGE"
    if "how do i" in text_l or "how to" in text_l:
        return "HOWTO"
    if "feature" in text_l or "request" in text_l:
        return "FEATURE"
    if "love" in text_l or "great" in text_l or "awesome" in text_l:
        return "PRAISE"
    return "UNKNOWN"


def run_pipeline(input_path: Path, output_path: Path, model: str, logger: logging.Logger) -> dict:
    logger.info(f"Starting pipeline (model={model}, input={input_path.name})")
    started = datetime.now()

    with open(input_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Loaded {len(rows)} rows from {input_path.name}")

    results = []
    for i, row in enumerate(rows, 1):
        try:
            category = mock_classify(row["text"])
            results.append({"id": row["id"], "text": row["text"], "category": category})
            logger.debug(f"Row {i}/{len(rows)}  id={row['id']}  → {category}")
        except Exception as e:
            logger.error(f"Row {i} failed: {e}", exc_info=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "category"])
        writer.writeheader()
        writer.writerows(results)

    elapsed = (datetime.now() - started).total_seconds()
    logger.info(f"Wrote {len(results)} results to {output_path.name} in {elapsed:.2f}s")

    return {"processed": len(results), "elapsed_s": elapsed}


# ─────────────────────────────────────────────────────────────
# SECTION 7: argparse — CLI interface
# ─────────────────────────────────────────────────────────────
def parse_args():
    """CLI:  python 06_file_and_env_handling.py --input X.csv --model gpt-4o-mini"""
    p = argparse.ArgumentParser(
        prog="batch_scorer",
        description="Classify a CSV of customer messages (MOCK_MODE safe)."
    )
    p.add_argument("--input", type=Path, default=sample_csv, help="CSV with 'id' + 'text' columns")
    p.add_argument("--output", type=Path, default=results_csv, help="Path for results CSV")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model to use")
    p.add_argument("--log-level", type=str, default=LOG_LEVEL,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    logger = setup_logging(args.log_level, args.log_file)
    stats = run_pipeline(args.input, args.output, args.model, logger)

    # Read the output back and print a summary — round-trip sanity check:
    with open(args.output, "r", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    counts: dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    print("\n" + "=" * 70)
    print("Category distribution")
    print("=" * 70)
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        bar = "█" * n
        print(f"  {cat:<10} {n}  {bar}")

    # Save a run summary as JSON — the pattern for run manifests:
    run_manifest = {
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "model": args.model,
        "input": str(args.input),
        "output": str(args.output),
        "processed": stats["processed"],
        "elapsed_s": stats["elapsed_s"],
        "categories": counts,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    logger.info(f"Manifest written: {manifest_path}")

    print("\n" + "=" * 70)
    print("✅ Summary")
    print("=" * 70)
    print(f"""
Complete end-to-end batch-scoring script — the shape of 90% of prod AI jobs:

  • .env loaded via python-dotenv (fallback OK)
  • pathlib.Path for cross-OS file paths
  • csv module for structured tabular I/O
  • logging with formatter + optional file handler
  • argparse for a real CLI (--help works!)
  • Run manifest JSON for observability / reproducibility

Run:
  python {Path(__file__).name} --input {args.input.name} --log-level DEBUG

Phase 1 complete. Continue to Phase 2 — AI/ML Basics.
""")
