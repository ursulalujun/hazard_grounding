#!/usr/bin/env python3
"""
Merge LoRA adapter with base model.

Usage:
    python merge_lora.py --base_model /path/to/base/model --adapter /path/to/lora/adapter --output_dir /path/to/output
"""

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os


def merge_lora(base_model_path, adapter_path, output_path):
    """
    Merge LoRA adapter with base model and save the merged model.

    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the LoRA adapter checkpoint
        output_path: Path to save the merged model
    """
    print(f"Loading base model from: {base_model_path}")

    # Detect model type and load appropriate model class
    if "Qwen3-VL" in base_model_path:
        print("Detected Qwen3-VL model")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = None  # Qwen3-VL uses processor
    else:
        print("Detected causal LM model")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print(f"Loading LoRA adapter from: {adapter_path}")

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    print("Merging completed...")

    # Save merged model
    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    model.save_pretrained(output_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)

    print("Model saved successfully!")
    print(f"\nMerged model saved at: {output_path}")
    print("You can now use this model for inference without loading the adapter separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the merged model"
    )

    args = parser.parse_args()

    merge_lora(args.base_model, args.adapter, args.output_dir)
