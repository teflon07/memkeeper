#!/usr/bin/env python3
"""Export a trained sequence-classification reranker to ONNX and optional INT8."""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    model_path = args.output / "model.onnx"
    if args.quantize:
        quantized_path = args.output / "model.int8.onnx"
        quantize_dynamic(model_path, quantized_path, weight_type=QuantType.QInt8)
        print(f"saved quantized model to {quantized_path}")
    else:
        print(f"saved model to {model_path}")


if __name__ == "__main__":
    main()
