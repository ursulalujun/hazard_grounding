"""
Visualization Module for Risk Grounding Evaluation.

This module contains functions for visualizing ground truth vs predictions
with colored bounding boxes.
"""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from data_pipeline.utils import bbox_norm_to_pixel


def convert_yx_first_to_xy_first(bbox_yx, width, height):
    """Convert bounding box from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max]."""
    from data_pipeline.utils import bbox_norm_to_pixel
    y_min, x_min, y_max, x_max = bbox_yx
    bbox_x_first = [x_min, y_min, x_max, y_max]
    return bbox_norm_to_pixel(bbox_x_first, width, height)


def visualize_comparison(item: Dict, target_model_name: str, save_folder: str):
    """
    Visualize GT (green) vs Prediction (red) on the same image.

    Args:
        item: Dict containing image_path, prediction, gt_data
        target_model_name: Name of target model (for bbox format detection)
        save_folder: Path to save visualization

    Returns:
        Path to saved image, or None if error
    """
    try:
        # Load image
        image_path = item["image_path"]
        img = Image.open(image_path)
        width, height = img.size

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)

        # Parse GT bboxes (action_triggered format)
        gt_risks = item["gt_data"]["safety_risk"]
        gt_target_bboxes = []
        gt_constraint_bboxes = []

        if "bbox_annotation" in gt_risks:
            bbox_annotation = gt_risks["bbox_annotation"]
            if "target_object" in bbox_annotation:
                for label, bbox in bbox_annotation["target_object"].items():
                    gt_target_bboxes.append({
                        "label": label,
                        "bbox": bbox  # Already in pixel coordinates
                    })
            if "constraint_object" in bbox_annotation:
                for label, bbox in bbox_annotation["constraint_object"].items():
                    gt_constraint_bboxes.append({
                        "label": label,
                        "bbox": bbox  # Already in pixel coordinates
                    })

        # Parse Prediction bboxes (action_triggered format)
        prediction = item["prediction"]
        is_gemini_gpt = ("gemini" in target_model_name.lower() or
                        "gpt" in target_model_name.lower())

        pred_target_bboxes = []
        pred_constraint_bboxes = []

        # Target object
        for bbox in prediction.get("target_object", []):
            if is_gemini_gpt:
                converted = convert_yx_first_to_xy_first(bbox, width, height)
            else:
                converted = bbox_norm_to_pixel(bbox, width, height)
            pred_target_bboxes.append({
                "label": "target",
                "bbox": converted
            })
        # Constraint object
        for bbox in prediction.get("constraint_object", []):
            if is_gemini_gpt:
                converted = convert_yx_first_to_xy_first(bbox, width, height)
            else:
                converted = bbox_norm_to_pixel(bbox, width, height)
            pred_constraint_bboxes.append({
                "label": "constraint",
                "bbox": converted
            })

        # Draw GT bboxes (green, solid line)
        for bbox_item in gt_target_bboxes:
            x1, y1, x2, y2 = bbox_item["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='green', facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"GT: {bbox_item['label']} (target)",
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        for bbox_item in gt_constraint_bboxes:
            x1, y1, x2, y2 = bbox_item["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='green', facecolor='none', linestyle='-'
            )
            ax.add_patch(rect)
            label = bbox_item['label'] if bbox_item['label'] else 'constraint'
            ax.text(x1, y1 - 5, f"GT: {label}",
                   color='green', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Draw Prediction bboxes (red, dashed line)
        for bbox_item in pred_target_bboxes:
            x1, y1, x2, y2 = bbox_item["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(x2, y1 - 5, f"Pred: target",
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        for bbox_item in pred_constraint_bboxes:
            x1, y1, x2, y2 = bbox_item["bbox"]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(x2, y1 - 5, f"Pred: constraint",
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='GT (Ground Truth)'),
            Line2D([0], [0], color='red', lw=3, linestyle='--', label='Prediction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        ax.set_title(f"GT (Green) vs Prediction (Red)", fontsize=14, fontweight='bold')
        ax.axis('off')

        # Save with directory structure matching edit_image_path
        # Extract relative path from edit_image_path (everything after "edit_image/")
        gt_risks = item["gt_data"]["safety_risk"]
        edit_image_path = gt_risks.get("edit_image_path", "")

        if edit_image_path and "edit_image/" in edit_image_path:
            # Get path after "edit_image/": e.g., "living_room/NYU0580__0.png"
            edit_image_idx = edit_image_path.find("edit_image/")
            relative_path = edit_image_path[edit_image_idx + len("edit_image/"):]
            save_path = os.path.join(save_folder, relative_path)
            # Create parent directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            # Fallback to original naming scheme
            sample_id = item.get("id", 0)
            save_path = os.path.join(save_folder, f"vis_{sample_id:06d}.jpg")

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

        return save_path

    except Exception as e:
        print(f"Error visualizing item {item.get('id', 'unknown')}: {e}")
        return None


def run_visualization_phase(eval_items: List[Dict],
                             target_model_name: str, save_folder: str,
                             max_workers: int = 8) -> int:
    """
    Run visualization phase in parallel.

    Args:
        eval_items: List of items containing predictions and ground truth
        target_model_name: Name of target model (for bbox format detection)
        save_folder: Path to save visualizations
        max_workers: Number of parallel workers

    Returns:
        Number of successfully visualized samples
    """
    vis_folder = os.path.join(save_folder, "visualizations")
    os.makedirs(vis_folder, exist_ok=True)

    print(f"Running visualization on {len(eval_items)} samples with {max_workers} workers...")

    success_count = 0

    def visualize_one(item):
        return visualize_comparison(item, target_model_name, vis_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(visualize_one, item): item["id"] for item in eval_items}

        with tqdm(total=len(eval_items), desc="Visualizing") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        success_count += 1
                except Exception as e:
                    print(f"Error in visualization: {e}")
                finally:
                    pbar.update(1)

    print(f"Visualization complete! Saved {success_count}/{len(eval_items)} images to: {vis_folder}")
    return success_count
