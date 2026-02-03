"""
Main Evaluation Script for Risk Grounding.

This script orchestrates the three-phase evaluation pipeline:
1. Inference: Run model inference on the dataset
2. Judgement: Evaluate predictions against ground truth
3. Visualization: Generate comparison images

Usage:
    # Run from risk_grounding directory (no __init__.py needed)
    python -m evaluation.evaluation \\
        --target_model checkpoints/Qwen3-VL-8B-Instruct \\
        --hazard_type action_triggered \\
        --data_type test

    # Or run as module (requires __init__.py)
    python -m evaluation.evaluation \\
        --target_model gemini-2.0-flash-exp \\
        --hazard_type environmental \\
        --skip_inference \\
        --skip_viz
"""

import argparse
import json
import os
import sys

# Add parent directory to path for direct execution (no __init__.py needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.inference import SafetyAgent, run_inference_phase
from evaluation.judgement import SafetyEvaluator, run_evaluation_phase
from evaluation.visualization import run_visualization_phase


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate risk grounding models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument('--target_model', type=str, required=True,
                        help='Path to local model or name of API model (e.g., gemini-2.0-flash-exp)')
    parser.add_argument('--hazard_type', type=str, required=True,
                        choices=['action_triggered', 'environmental'],
                        help='Type of hazard to evaluate')

    # Optional arguments
    parser.add_argument('--evaluation_model', type=str, default='Qwen/Qwen3-VL-235B-A22B-Thinking',
                        help='Model for risk matching judgment')
    parser.add_argument('--data_type', type=str, default='test',
                        choices=['test', 'train'],
                        help='Dataset type to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for local model inference')
    parser.add_argument('--max_workers', type=int, default=24,
                        help='Max workers for parallel API calls (inference/evaluation)')
    parser.add_argument('--viz_workers', type=int, default=8,
                        help='Max workers for visualization')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference phase, use existing predictions file')
    parser.add_argument('--skip_viz', action='store_true',
                        help='Skip visualization phase')

    args = parser.parse_args()

    # Setup paths
    if args.data_type == "test":
        DATASET_PATH = os.path.join("data_pipeline", "data", "test", args.hazard_type, "annotation_info.json")
    else:
        DATASET_PATH = os.path.join("data_pipeline", "data", args.hazard_type, "success_list.json")

    save_folder = os.path.join("results", args.data_type, args.hazard_type, os.path.basename(args.target_model))
    os.makedirs(save_folder, exist_ok=True)

    predictions_file = os.path.join(save_folder, "predictions.json")
    output_file = os.path.join(save_folder, "evaluation_results.json")

    # Load dataset
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        gt_dataset = json.load(f)
    if args.data_type == "train":
        gt_dataset = gt_dataset[:200]

    print(f"Dataset: {len(gt_dataset)} samples")

    # ======================================================================
    # Phase 1: Inference
    # ======================================================================
    if not args.skip_inference:
        print("\n" + "="*60)
        print("PHASE 1: INFERENCE")
        print("="*60)
        agent = SafetyAgent(model_name=args.target_model, batch_size=args.batch_size)
        eval_items = run_inference_phase(agent, gt_dataset, args.hazard_type, predictions_file)
    else:
        print("\n" + "="*60)
        print("SKIPPING INFERENCE - USING EXISTING PREDICTIONS")
        print("="*60)

        # Load existing predictions
        with open(predictions_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)

        # Reconstruct eval_items
        valid_items = []
        for i, gt_data in enumerate(gt_dataset):
            if gt_data.get('safety_risk') is None:
                continue
            if gt_data.get("state") == "failed":
                continue
            valid_items.append({
                "id": i,
                "gt_data": gt_data,
                "image_path": os.path.join("data_pipeline", gt_data['safety_risk']['edit_image_path'])
            })

        # Merge with predictions
        pred_map = {p["id"]: p for p in predictions}
        eval_items = []
        for item in valid_items:
            if item["id"] in pred_map:
                pred = pred_map[item["id"]]
                eval_items.append({
                    "id": item["id"],
                    "image_path": item["image_path"],
                    "prediction": pred["prediction"],
                    "raw_output": pred["raw_output"],
                    "gt_data": item["gt_data"]
                })

        print(f"Loaded {len(eval_items)} predictions from {predictions_file}")

    # ======================================================================
    # Phase 2: Evaluation
    # ======================================================================
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION")
    print("="*60)
    evaluator = SafetyEvaluator(model_name=args.evaluation_model, target_model_name=args.target_model)
    detailed_logs, final_metrics = run_evaluation_phase(evaluator, eval_items, args.hazard_type, args.max_workers)

    # Save results
    final_output_data = {
        "summary_metrics": final_metrics,
        "details": detailed_logs
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output_data, f, indent=4, ensure_ascii=False)

    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    if final_metrics:
        print(f"1. Avg Safe Accuracy: {final_metrics.get('avg_safe_accuracy', 0):.4f}")
        print(f"2. Avg Risk GPT Match: {final_metrics.get('avg_risk_match', 0):.4f}")
        print(f"3. Avg IoU (target_object) - ALL unsafe: {final_metrics.get('avg_iou_target_object', 0):.4f}")
        print(f"4. Avg IoU (constraint_object) - ALL unsafe: {final_metrics.get('avg_iou_constraint_object', 0):.4f}")
        print(f"   (unsafe samples: {final_metrics.get('unsafe_sample_count', 0)})")
        print(f"   (correct target IoU: {final_metrics.get('avg_iou_target_object_correct_only', 0):.4f} on {final_metrics.get('correct_target_sample_count', 0)} samples)")
        print(f"   (correct constraint IoU: {final_metrics.get('avg_iou_constraint_object_correct_only', 0):.4f} on {final_metrics.get('correct_constraint_sample_count', 0)} samples)")
    print(f"\nResults saved to: {output_file}")

    # ======================================================================
    # Phase 3: Visualization
    # ======================================================================
    if not args.skip_viz:
        print("\n" + "="*60)
        print("PHASE 3: VISUALIZATION")
        print("="*60)
        run_visualization_phase(eval_items, args.hazard_type, args.target_model, save_folder, args.viz_workers)
    else:
        print("\n" + "="*60)
        print("SKIPPING VISUALIZATION")
        print("="*60)

    print("\n" + "="*60)
    print("ALL PHASES COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
