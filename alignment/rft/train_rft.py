# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# RFT (Reinforcement Fine-tuning) Training Script for Risk Grounding Task
# Adapted from Visual-RFT for safety hazard detection and localization

import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

import PIL
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Import GRPO trainer and utilities from Visual-RFT
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Visual-RFT", "src", "virft", "src"))

from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Import reward functions
from rewards import (
    reward_funcs_registry,
    ENVIRONMENTAL_EVAL_TEMPLATE,
    ACTION_TRIGGER_EVAL_TEMPLATE,
)


@dataclass
class RFTScriptArguments(ScriptArguments):
    """
    Script arguments for the Risk Grounding RFT training.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'safe_accuracy', 'risk_match', 'iou', 'format'
        hazard_type (`str`):
            Type of hazard: 'environmental' or 'action_triggered'
        dataset_path (`str`):
            Path to the dataset directory or JSON file
        embedding_model_path (`str`):
            Path to the sentence embedding model for risk matching
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["safe_accuracy", "risk_match", "iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'safe_accuracy', 'risk_match', 'iou', 'format'"},
    )
    hazard_type: str = field(
        default="environmental",
        metadata={"help": "Type of hazard: 'environmental' or 'action_triggered'"},
    )
    dataset_path: str = field(
        default="risk_grounding/data_pipeline/data/environmental/success_list.json",
        metadata={"help": "Path to the dataset JSON file"},
    )
    embedding_model_path: str = field(
        default="risk_grounding/checkpoints/all-MiniLM-L6-v2",
        metadata={"help": "Path to the sentence embedding model"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    # Reward weights
    reward_weight_safe_accuracy: float = field(
        default=1.0,
        metadata={"help": "Weight for safe_accuracy reward"},
    )
    reward_weight_risk_match: float = field(
        default=1.0,
        metadata={"help": "Weight for risk_match reward"},
    )
    reward_weight_iou: float = field(
        default=1.0,
        metadata={"help": "Weight for combined iou reward"},
    )
    reward_weight_iou_target_object: float = field(
        default=1.0,
        metadata={"help": "Weight for iou_target_object reward"},
    )
    reward_weight_iou_constraint_object: float = field(
        default=1.0,
        metadata={"help": "Weight for iou_constraint_object reward"},
    )
    reward_weight_format: float = field(
        default=0.5,
        metadata={"help": "Weight for format reward"},
    )


def load_risk_grounding_dataset(dataset_path: str, hazard_type: str, max_samples: Optional[int] = None):
    """
    Load the risk grounding dataset from JSON file.
    Directly processes all samples into conversation format for training.

    Args:
        dataset_path: Path to the JSON dataset file
        hazard_type: Type of hazard (environmental or action_triggered)
        max_samples: Maximum number of samples to load (for debugging)

    Returns:
        Dataset object with image, prompt, and solution fields (ready for training)
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Filter and process valid samples
    processed_data = []
    for item in tqdm(raw_data, desc="Loading dataset"):
        # Skip failed samples
        if "state" in item and item["state"] == "failed":
            continue

        safety_risk = item.get("safety_risk")
        if not safety_risk:
            continue

        # Determine if safe based on safety_hazard field
        # If safety_hazard is null or None, the scene is safe (no risk)
        # If safety_hazard exists, the scene is unsafe
        safety_hazard = safety_risk.get("safety_hazard")
        is_gt_safe = (safety_hazard is None or safety_hazard == "null" or
                      (isinstance(safety_hazard, str) and safety_hazard.strip() == ""))

        # Build ground truth data
        gt_data = {
            "is_gt_safe": is_gt_safe,
            "safety_principle": safety_risk.get("safety_principle", ""),
        }

        # Add bbox list if not safe
        if not is_gt_safe:
            bbox_annotation = safety_risk.get("bbox_annotation", {})
            bbox_list = []
            if hazard_type == "environmental":
                # Environmental format: {"label1": [x1,y1,x2,y2], "label2": [...]}
                for label, bbox in bbox_annotation.items():
                    bbox_list.append({
                        "label": label,
                        "bounding_box": bbox  # Pixel coordinates
                    })
            else:  # action_triggered
                # Action triggered format: {"category1": {"label1": [x1,y1,x2,y2], ...}, "category2": {...}}
                # Flatten all bboxes from all categories into a single list
                for category_name, category_bboxes in bbox_annotation.items():
                    for label, bbox in category_bboxes.items():
                        bbox_list.append({
                            "label": label,
                            "bounding_box": bbox
                        })
            gt_data["bbox_list"] = bbox_list

        # Get image path
        image_path = os.path.join("data_pipeline", safety_risk.get("edit_image_path", ""))
        # Make image path absolute
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(__file__), "..", "..", image_path)

        # Check if image exists and get image size
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        # Get image size for bbox coordinate conversion
        try:
            img = PIL.Image.open(image_path)
            img_width, img_height = img.size
            gt_data["image_width"] = img_width
            gt_data["image_height"] = img_height
        except Exception as e:
            print(f"Warning: Could not get image size for {image_path}: {e}")
            gt_data["image_width"] = None
            gt_data["image_height"] = None

        # Get instruction for action_triggered
        instruction = safety_risk.get("instruction", "")

        # ====================================================================
        # Build conversation format directly (no need for dataset.map())
        # ====================================================================

        # Select prompt template
        if hazard_type == "action_triggered":
            prompt_text = ACTION_TRIGGER_EVAL_TEMPLATE.format(instruction=instruction)
        else:
            prompt_text = ENVIRONMENTAL_EVAL_TEMPLATE

        # Build prompt in Qwen3-VL format
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # Add to processed data - store image path, not PIL Image
        processed_data.append({
            "image": image_path,  # Store path as string, Image feature will handle loading
            "prompt": prompt,
            "solution": gt_data,
        })

        if max_samples and len(processed_data) >= max_samples:
            break

    print(f"Loaded and processed {len(processed_data)} samples from {dataset_path}")

    # Create dataset and shuffle to mix safe and unsafe scenes
    dataset = Dataset.from_list(processed_data)
    dataset = dataset.shuffle(seed=42)

    # Print statistics
    safe_count = sum(1 for item in dataset if item["solution"]["is_gt_safe"])
    unsafe_count = len(dataset) - safe_count
    print(f"Dataset statistics: {safe_count} safe samples, {unsafe_count} unsafe samples ({safe_count/len(dataset)*100:.1f}% safe)")

    return dataset


def make_conversation_image(example, hazard_type):
    """
    Convert dataset example to conversation format for Qwen3-VL.

    Args:
        example: Dataset example with image, instruction, safety_risk
        hazard_type: Type of hazard

    Returns:
        Dictionary with image, prompt, and solution
    """
    # Load image
    image_path = example["image"]
    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Create a dummy black image
        image = PIL.Image.new("RGB", (224, 224), (0, 0, 0))

    # Select prompt template
    if hazard_type == "action_triggered":
        instruction = example.get("instruction", "")
        prompt_text = ACTION_TRIGGER_EVAL_TEMPLATE.format(instruction=instruction)
    else:
        prompt_text = ENVIRONMENTAL_EVAL_TEMPLATE

    # Build prompt in Qwen3-VL format
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]

    # Get ground truth solution
    solution = example["safety_risk"]

    return {
        "image": image,
        "prompt": prompt,
        "solution": solution,
    }


def main(script_args, training_args, model_args):
    """Main training function."""

    # Build reward weights from script arguments
    reward_weights = {
        "safe_accuracy": script_args.reward_weight_safe_accuracy,
        "risk_match": script_args.reward_weight_risk_match,
        "iou": script_args.reward_weight_iou,
        "iou_target_object": script_args.reward_weight_iou_target_object,
        "iou_constraint_object": script_args.reward_weight_iou_constraint_object,
        "format": script_args.reward_weight_format,
    }

    # Update reward functions with custom embedding model path and weights
    from weighted_rewards import get_weighted_reward_registry

    # Get weighted reward functions
    custom_registry = get_weighted_reward_registry(
        embedding_model_path=script_args.embedding_model_path,
        weights=reward_weights
    )

    # Get reward functions - use split IoU for action_triggered
    if script_args.hazard_type == "action_triggered":
        script_args.reward_funcs = ['safe_accuracy', 'risk_match', 'iou_target_object', 'iou_constraint_object', 'format']
    else:
        script_args.reward_funcs = ['safe_accuracy', 'risk_match', 'iou', 'format']

    reward_funcs = [custom_registry[func] for func in script_args.reward_funcs]

    print(f"Using reward functions: {script_args.reward_funcs}")
    print(f"Reward weights: {reward_weights}")

    # Load dataset
    print(f"Loading dataset from: {script_args.dataset_path}")
    dataset = load_risk_grounding_dataset(
        dataset_path=script_args.dataset_path,
        hazard_type=script_args.hazard_type,
        max_samples=None,  # Set to a small number for debugging
    )
    # Data is already processed into conversation format with columns: image, prompt, solution

    # Split into train and validation
    if training_args.eval_strategy != "no":
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        dataset = {
            "train": split_dataset["train"],
            "test": split_dataset["test"],
        }
    else:
        dataset = {"train": dataset}

    print(f"Training samples: {len(dataset['train'])}")
    if "test" in dataset:
        print(f"Validation samples: {len(dataset['test'])}")

    # Select trainer class
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print(f"Using trainer: {trainer_cls.__name__}")

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test"),
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and save the model
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_path)

    print("Training completed!")


if __name__ == "__main__":
    parser = TrlParser((RFTScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)