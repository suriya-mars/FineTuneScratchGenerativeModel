"""
Phase 1 — CoT Rationale Generation

Calls OpenAI gpt-4o-mini for each training example, extracts the
chain-of-thought, verifies the answer against ground truth, and writes
verified triples to data/processed/train_cot.jsonl.

Resume-safe: skips IDs already in the output file.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/generate_rationales.py \
        --input      data/raw/train.csv \
        --output     data/processed/train_cot.jsonl \
        --concurrency 20 \
        --max_rows   0        # 0 = all rows
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, RateLimitError

load_dotenv()  # loads .env from project root

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocess import (
    SYSTEM_PROMPT,
    categorize,
    build_teacher_user_message,
    extract_answer,
    format_training_example,
)

MODEL = "gpt-4o-mini"
MAX_TOKENS = 512
TEMPERATURE = 0          # deterministic — we verify exact match
SAVE_EVERY = 50          # flush to disk every N verified rows
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5.0        # seconds between retries on rate-limit


def load_processed_ids(output_path: str) -> set[str]:
    """Return IDs already written to the output file (for resuming)."""
    seen = set()
    path = Path(output_path)
    if not path.exists():
        return seen
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


def load_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def generate_rationale(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict,
) -> dict | None:
    """
    Call gpt-4o-mini for one row.
    Returns a result dict if the extracted answer matches ground truth, else None.
    """
    prompt = row["prompt"]
    ground_truth = row["answer"].strip()
    category = categorize(prompt)
    user_msg = build_teacher_user_message(prompt, category)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                break
            except RateLimitError:
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None
            except APIError as exc:
                print(f"  [API error] id={row['id']}: {exc}", flush=True)
                return None

    response_text = response.choices[0].message.content or ""

    extracted = extract_answer(response_text)
    if extracted is None:
        return None

    # Everything before the last "Answer:" is the chain-of-thought
    parts = response_text.rsplit("Answer:", 1)
    cot = parts[0].strip() if len(parts) > 1 else response_text.strip()

    # Exact-match verification (case-insensitive, whitespace-normalised)
    if extracted.lower().strip() != ground_truth.lower().strip():
        return None

    training_text = format_training_example(prompt, cot, ground_truth)
    return {
        "id":            row["id"],
        "category":      category,
        "prompt":        prompt,
        "cot":           cot,
        "answer":        ground_truth,
        "training_text": training_text,
    }


async def run(args: argparse.Namespace) -> None:
    rows = load_csv(args.input)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    processed_ids = load_processed_ids(args.output)
    pending = [r for r in rows if r["id"] not in processed_ids]

    print(f"Total rows     : {len(rows)}")
    print(f"Already done   : {len(processed_ids)}")
    print(f"Pending        : {len(pending)}")
    print(f"Model          : {MODEL}")
    print(f"Concurrency    : {args.concurrency}")
    print()

    if not pending:
        print("Nothing to do.")
        return

    client = AsyncOpenAI()            # reads OPENAI_API_KEY from env
    semaphore = asyncio.Semaphore(args.concurrency)

    verified = 0
    failed = 0
    buffer: list[dict] = []
    start = time.time()

    tasks = [generate_rationale(client, semaphore, row) for row in pending]

    outfile = open(args.output, "a", buffering=1)
    try:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            result = await coro
            if result is not None:
                verified += 1
                buffer.append(result)
            else:
                failed += 1

            if len(buffer) >= SAVE_EVERY:
                for item in buffer:
                    outfile.write(json.dumps(item) + "\n")
                buffer.clear()

            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start
                rate = i / elapsed
                remaining = (len(tasks) - i) / rate if rate > 0 else 0
                pct = verified / i * 100
                print(
                    f"  [{i:>5}/{len(tasks)}] "
                    f"verified={verified} ({pct:.0f}%)  "
                    f"failed={failed}  "
                    f"{rate:.1f} rows/s  "
                    f"ETA {remaining/60:.1f}min",
                    flush=True,
                )

        for item in buffer:
            outfile.write(json.dumps(item) + "\n")
    finally:
        outfile.close()

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"Done in           {elapsed/60:.1f} min")
    print(f"Verified triples  {verified}")
    print(f"Failed / skipped  {failed}")
    if verified + failed > 0:
        print(f"Pass rate         {verified/(verified+failed)*100:.1f}%")
    print(f"Output            {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CoT rationales via OpenAI gpt-4o-mini"
    )
    parser.add_argument("--input",       default="data/raw/train.csv")
    parser.add_argument("--output",      default="data/processed/train_cot.jsonl")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Parallel API calls (default 20)")
    parser.add_argument("--max_rows",    type=int, default=0,
                        help="Limit rows for testing (0 = all)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
