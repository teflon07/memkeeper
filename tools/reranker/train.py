#!/usr/bin/env python3
"""Fine-tune a compact pairwise cross-encoder for Memkeeper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


def load_examples(path: Path) -> list[InputExample]:
    examples: list[InputExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = row["query"]
            positive = row["positive"]
            negative = row["negative"]
            # Pairwise expansion gives the model both ordering directions.
            examples.append(InputExample(texts=[query, positive], label=1.0))
            examples.append(InputExample(texts=[query, negative], label=0.0))
    if not examples:
        raise ValueError(f"No training examples found in {path}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    examples = load_examples(args.train)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)
    warmup_steps = max(1, int(len(loader) * args.epochs * args.warmup_ratio))

    args.output.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(args.output),
        show_progress_bar=True,
        use_amp=True,
    )
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
