"""
Reward functions for Risk Grounding RFT training.
Implements four reward components:
1. Safe accuracy reward - whether the safety judgment is correct
2. Safety hazard match reward - whether the safety hazard description matches (using embedding cos_sim)
3. Principle accuracy reward - whether the safety principle classification is correct
4. IoU reward - bounding box localization accuracy
"""

import re
import os
import torch
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the reward calculator. 
embedding_model_path="checkpoints/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(embedding_model_path, device="cpu")

class RiskGroundingRewards:
    """Reward functions for risk grounding task."""

    def __init__(self):
        pass

    # ========================================================================
    # 1. Safe Accuracy Reward
    # ========================================================================
    def safe_accuracy_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on whether the safety judgment is correct.

        Args:
            completions: Model generated responses (list of messages)
            solution: Ground truth data containing is_gt_safe

        Returns:
            List of rewards (1.0 if correct, 0.0 if incorrect)
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            pred_safe = self._parse_safe(content)
            gt_safe = gt_data.get("is_gt_safe", False)

            # Reward is 1.0 if prediction matches ground truth
            if pred_safe is None:
                # Failed to parse, give partial reward
                reward = 0.0
            else:
                reward = 1.0 if pred_safe == gt_safe else 0.0

            rewards.append(reward)

        return rewards

    def _parse_safe(self, content: str) -> Optional[bool]:
        """
        Parse the 'safe' field from model output.

        Args:
            content: Model generated text

        Returns:
            True if safe, False if unsafe, None if parsing fails
        """
        # Try to match JSON format "safe": true/false
        patterns = [
            r'"safe"\s*:\s*(true|false)',
            r'"safe"\s*:\s*(True|False)',
            r'"safe"\s*:\s*(1|0)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                if value in ['true', '1']:
                    return True
                elif value in ['false', '0']:
                    return False

        # Try to parse as JSON
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'safe' in data:
                    return bool(data['safe'])
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # ========================================================================
    # 2. Safety Hazard Match Reward (using embedding cosine similarity)
    # ========================================================================
    def safety_hazard_match_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on semantic similarity of safety hazard descriptions.
        Uses sentence embedding cosine similarity instead of judge model for efficiency.
        Returns continuous cosine similarity as reward (0.0 to 1.0).

        Args:
            completions: Model generated responses
            solution: Ground truth data containing safety_principle

        Returns:
            List of rewards (cosine similarity scores, range 0.0 to 1.0)
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            pred_hazard = self._parse_safety_hazard(content)
            gt_hazard = gt_data.get("safety_hazard", "")
            gt_safe = gt_data.get("is_gt_safe", False)

            # If both are safe, perfect match
            if gt_safe:
                pred_safe = self._parse_safe(content)
                if pred_safe:
                    reward = 1.0
                else:
                    reward = 0.0
            # If GT is unsafe, check safety hazard description
            elif not gt_safe:
                if not pred_hazard:
                    reward = 0.0
                else:
                    # Compute embedding similarity
                    try:
                        # Use numpy encoding and sklearn cosine similarity to avoid torch device conflicts
                        pred_emb = embedding_model.encode(
                            pred_hazard,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        gt_emb = embedding_model.encode(
                            gt_hazard,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        # Compute cosine similarity using sklearn
                        similarity = cosine_similarity(
                            pred_emb.reshape(1, -1),
                            gt_emb.reshape(1, -1)
                        )[0, 0]
                        # Continuous reward: direct cosine similarity
                        reward = float(similarity)
                    except Exception as e:
                        reward = 0.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards

    def _parse_safety_hazard(self, content: str) -> Optional[str]:
        """
        Parse the 'safety_hazard' field from model output.

        Args:
            content: Model generated text

        Returns:
            Safety hazard description string, or None if parsing fails
        """
        # Try to match JSON format "safety_hazard": "description"
        patterns = [
            r'"safety_hazard"\s*:\s*"([^"]*)"',
            r'"safety_hazard"\s*:\s*\'([^\']*)\'',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        # Try to parse as JSON
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'safety_hazard' in data and data['safety_hazard']:
                    return str(data['safety_hazard'])
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # ========================================================================
    # 3. Principle Accuracy Reward
    # ========================================================================
    def principle_accuracy_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on whether the predicted principle_id matches ground truth.
        This follows the same logic as principle_acc in judgement.py.

        Args:
            completions: Model generated responses
            solution: Ground truth data containing safety_principle

        Returns:
            List of rewards (1.0 if correct, 0.0 if incorrect)
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            gt_safe = gt_data.get("is_gt_safe", False)
            pred_safe = self._parse_safe(content)

            # Extract GT principle ID from safety_principle text
            gt_principle_text = gt_data.get("safety_principle", "")
            gt_principle_id = self._extract_principle_id(gt_principle_text)

            # Get predicted principle ID
            pred_principle_id = self._parse_principle_id(content)

            # Compare
            if gt_principle_id is not None and pred_principle_id is not None:
                reward = 1.0 if gt_principle_id == pred_principle_id else 0.0
            elif gt_safe and pred_principle_id is None:
                # Both safe, principle_id should be null
                reward = 1.0
            else:
                # Otherwise (unsafe but no principle_id provided): 0
                reward = 0.0

            rewards.append(reward)

        return rewards

    def _extract_principle_id(self, safety_principle_text: str) -> Optional[int]:
        """
        Extract principle ID from safety principle text.
        Adapted from judgement.py:_extract_principle_id

        Args:
            safety_principle_text: Text like "4. Power Off Before Cleaning/Moving: ..."

        Returns:
            The principle ID as integer, or None if not found
        """
        if not safety_principle_text:
            return None
        # Match pattern: "number. title: description"
        # Examples: "4. Power Off Before Cleaning/Moving: ..."
        match = re.match(r'(\d+)\.\s*', safety_principle_text.strip())
        return int(match.group(1)) if match else None

    def _parse_principle_id(self, content: str) -> Optional[int]:
        """
        Parse the 'principle_id' field from model output.

        Args:
            content: Model generated text

        Returns:
            Principle ID as integer, or None if parsing fails or value is null
        """
        # Try to match JSON format "principle_id": number
        match = re.search(r'"principle_id"\s*:\s*(\d+)', content)
        if match:
            return int(match.group(1))

        # Try to parse as JSON (handles null case and other formats)
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'principle_id' in data:
                    principle_id = data['principle_id']
                    if principle_id is None:
                        return None
                    return int(principle_id)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    # ========================================================================
    # 4. IoU Reward (bounding box localization) - Split into target and constraint
    # ========================================================================
    def iou_target_object_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on IoU of predicted and ground truth target_object bounding boxes.
        Computes the IoU of the union of all target_object bboxes.

        NOTE: target_object is the object the user needs to interact with for the task.
        This should ALWAYS be identified and localized, regardless of whether the scene is safe or unsafe.
        The safety judgment is about constraint_object (hazards), not target_object.

        Args:
            completions: Model generated responses
            solution: Ground truth data containing bbox_annotation and image dimensions

        Returns:
            List of IoU rewards for target_object
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            # Parse predicted target_object bboxes (list of [x_min, y_min, x_max, y_max])
            pred_target_bboxes = self._parse_target_object_bboxes(content)

            # Get GT target_object bboxes
            gt_target_bboxes = self._get_gt_target_bboxes(gt_data)

            # Get image dimensions for coordinate conversion
            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            # Convert predicted bboxes from normalized to pixel coordinates
            if img_width and img_height:
                pred_target_bboxes = self._normalized_to_pixel_bbox_list(pred_target_bboxes, img_width, img_height)

            # Calculate IoU: target_object should ALWAYS be localized, regardless of scene safety
            if gt_target_bboxes:
                # GT has target_object bboxes, compute IoU
                iou = self.compute_list_iou(gt_target_bboxes, pred_target_bboxes)
                reward = iou
            else:
                # GT has NO target_object bboxes
                # This should ideally not happen for action_triggered tasks
                # If no GT target bbox, model should also NOT predict any
                reward = 1.0 if not pred_target_bboxes else 0.0

            rewards.append(reward)

        return rewards

    def iou_constraint_object_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on IoU of predicted and ground truth constraint_object bounding boxes.
        Computes the IoU of the union of all constraint_object bboxes.

        Args:
            completions: Model generated responses
            solution: Ground truth data containing bbox_annotation and image dimensions

        Returns:
            List of IoU rewards for constraint_object
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            gt_safe = gt_data.get("is_gt_safe", False)
            pred_safe = self._parse_safe(content)

            # Parse predicted constraint_object bboxes (list of [x_min, y_min, x_max, y_max])
            pred_constraint_bboxes = self._parse_constraint_object_bboxes(content)

            # Get GT constraint_object bboxes
            gt_constraint_bboxes = self._get_gt_constraint_bboxes(gt_data)

            # Get image dimensions for coordinate conversion
            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            # Convert predicted bboxes from normalized to pixel coordinates
            if img_width and img_height:
                pred_constraint_bboxes = self._normalized_to_pixel_bbox_list(pred_constraint_bboxes, img_width, img_height)

            # If GT is safe, constraint_object should be empty
            if gt_safe:
                if not pred_constraint_bboxes:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                # GT is unsafe
                if pred_safe:
                    # Model predicted safe, IoU = 0
                    reward = 0.0
                else:
                    # Compute IoU for constraint_object
                    if gt_constraint_bboxes:
                        iou = self.compute_list_iou(gt_constraint_bboxes, pred_constraint_bboxes)
                        reward = iou
                    else:
                        # No GT constraint bboxes (hazard from target's own state)
                        # Model should also predict empty constraint_object
                        reward = 1.0 if not pred_constraint_bboxes else 0.0

            rewards.append(reward)

        return rewards

    # Legacy iou_reward for backward compatibility
    # Handles both environmental (bbox_list format) and action_triggered (target/constraint format)
    def iou_reward(self, completions, solution, **kwargs):
        """
        Combined IoU reward.
        - For environmental: uses bbox_list format (single set of bboxes)
        - For action_triggered: average of target_object and constraint_object
        """
        # Detect the format by checking the first solution's bbox_annotation structure
        if not solution or len(solution) == 0:
            return [0.0] * len(completions)

        first_solution = solution[0]
        bbox_annotation = first_solution.get("bbox_annotation", {})

        # Check if action_triggered format (has target_object or constraint_object keys)
        is_action_triggered = (
            "target_object" in bbox_annotation or
            "constraint_object" in bbox_annotation
        )

        if is_action_triggered:
            # Action triggered: average of target and constraint IoU
            target_rewards = self.iou_target_object_reward(completions, solution, **kwargs)
            constraint_rewards = self.iou_constraint_object_reward(completions, solution, **kwargs)
            return [(t + c) / 2 for t, c in zip(target_rewards, constraint_rewards)]
        else:
            # Environmental: use bbox_list format directly
            return self._iou_reward_bbox_list(completions, solution, **kwargs)

    def _iou_reward_bbox_list(self, completions, solution, **kwargs):
        """
        IoU reward for environmental hazard (bbox_list format).
        This is the original iou_reward logic before splitting into target/constraint.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            pred_bboxes = self._parse_bboxes_list(content)
            gt_safe = gt_data.get("is_gt_safe", False)
            gt_bboxes = gt_data.get("bbox_list", [])

            # Get image dimensions for coordinate conversion
            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            # Convert predicted bboxes from normalized to pixel coordinates
            if img_width and img_height:
                pred_bboxes = self._normalized_to_pixel_bboxes_dict(pred_bboxes, img_width, img_height)

            # If GT is safe, no bbox expected
            if gt_safe:
                pred_safe = self._parse_safe(content)
                if pred_safe and not pred_bboxes:
                    reward = 1.0
                elif not pred_safe:
                    reward = 0.0
                else:
                    reward = 0.0
            else:
                # GT is unsafe
                pred_safe = self._parse_safe(content)
                if pred_safe:
                    reward = 0.0
                else:
                    # Compute IoU
                    iou = self.compute_list_iou(gt_bboxes, pred_bboxes)
                    reward = iou

            rewards.append(reward)

        return rewards

    def _parse_bboxes_list(self, content: str) -> List[Dict]:
        """
        Parse bbox_list from model output (environmental format).
        Expected format: "bbox_list": [{"label": str, "bounding_box": [x1, y1, x2, y2]}, ...]
        """
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'bbox_list' in data:
                    bboxes = data['bbox_list']
                    if isinstance(bboxes, list) and len(bboxes) > 0:
                        return bboxes
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _normalized_to_pixel_bboxes_dict(self, bboxes: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        """
        Convert bboxes from normalized coordinates (0-1000) to pixel coordinates.
        For environmental format with label and bounding_box dict structure.
        """
        converted = []
        for bbox_item in bboxes:
            bbox = bbox_item.get("bounding_box", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                pixel_bbox = [
                    int(x1 / 1000 * img_width),
                    int(y1 / 1000 * img_height),
                    int(x2 / 1000 * img_width),
                    int(y2 / 1000 * img_height)
                ]
                converted.append({
                    "label": bbox_item.get("label", ""),
                    "bounding_box": pixel_bbox
                })
        return converted

    def _get_gt_target_bboxes(self, gt_data: Dict) -> List[Dict]:
        """
        Extract ground truth target_object bboxes from solution data.

        Args:
            gt_data: Ground truth data

        Returns:
            List of bbox dicts with label and bounding_box
        """
        bbox_annotation = gt_data.get("bbox_annotation", {})
        target_bboxes = []

        if "target_object" in bbox_annotation:
            for label, bbox in bbox_annotation["target_object"].items():
                target_bboxes.append({
                    "label": label,
                    "bounding_box": bbox
                })

        return target_bboxes

    def _get_gt_constraint_bboxes(self, gt_data: Dict) -> List[Dict]:
        """
        Extract ground truth constraint_object bboxes from solution data.

        Args:
            gt_data: Ground truth data

        Returns:
            List of bbox dicts with label and bounding_box
        """
        bbox_annotation = gt_data.get("bbox_annotation", {})
        constraint_bboxes = []

        if "constraint_object" in bbox_annotation:
            for label, bbox in bbox_annotation["constraint_object"].items():
                constraint_bboxes.append({
                    "label": label,
                    "bounding_box": bbox
                })

        return constraint_bboxes

    def _parse_target_object_bboxes(self, content: str) -> List[List[int]]:
        """
        Parse target_object bboxes from model output.
        Expected format: "target_object": [[x1, y1, x2, y2], ...]

        Args:
            content: Model generated text

        Returns:
            List of bboxes as [x_min, y_min, x_max, y_max]
        """
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'target_object' in data:
                    bboxes = data['target_object']
                    if isinstance(bboxes, list) and len(bboxes) > 0:
                        # Validate bbox format
                        valid_bboxes = []
                        for bbox in bboxes:
                            if isinstance(bbox, list) and len(bbox) == 4:
                                valid_bboxes.append(bbox)
                        return valid_bboxes
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _parse_constraint_object_bboxes(self, content: str) -> List[List[int]]:
        """
        Parse constraint_object bboxes from model output.
        Expected format: "constraint_object": [[x1, y1, x2, y2], ...]

        Args:
            content: Model generated text

        Returns:
            List of bboxes as [x_min, y_min, x_max, y_max]
        """
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'constraint_object' in data:
                    bboxes = data['constraint_object']
                    if isinstance(bboxes, list) and len(bboxes) > 0:
                        # Validate bbox format
                        valid_bboxes = []
                        for bbox in bboxes:
                            if isinstance(bbox, list) and len(bbox) == 4:
                                valid_bboxes.append(bbox)
                        return valid_bboxes
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    def _normalized_to_pixel_bbox_list(self, bboxes: List[List[int]], img_width: int, img_height: int) -> List[Dict]:
        """
        Convert bboxes from normalized coordinates (0-1000) to pixel coordinates.
        Qwen3-VL outputs bboxes in the range [0, 1000].

        Args:
            bboxes: List of bbox lists [x1, y1, x2, y2]
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of bbox dicts with pixel coordinates
        """
        converted = []
        for bbox in bboxes:
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                pixel_bbox = [
                    int(x1 / 1000 * img_width),
                    int(y1 / 1000 * img_height),
                    int(x2 / 1000 * img_width),
                    int(y2 / 1000 * img_height)
                ]
                converted.append({
                    "label": f"bbox_{len(converted)}",
                    "bounding_box": pixel_bbox
                })
        return converted

    # ========================================================================
    # IoU Computation Functions
    # ========================================================================
    def compute_list_iou(self, gt_bbox_list: List[Dict], pred_bbox_list: List[Dict]) -> float:
        """
        Calculate the IoU of the area covered by two bbox lists.
        That is: IoU(Union(box_list1), Union(box_list2))

        This is adapted from evaluation.py:compute_list_iou

        Args:
            gt_bbox_list: Ground truth bounding boxes
            pred_bbox_list: Predicted bounding boxes

        Returns:
            IoU score between 0 and 1
        """
        if not pred_bbox_list:
            return 0.0
        if not gt_bbox_list:
            return 0.0

        box_list1 = []
        box_list2 = []
        for item in gt_bbox_list:
            bbox = item.get("bounding_box", [])
            if bbox:
                box_list1.append(bbox)
        for item in pred_bbox_list:
            bbox = item.get("bounding_box", [])
            if bbox:
                box_list2.append(bbox)

        # Boundary check
        if not box_list1 or not box_list2:
            return 0.0

        # Convert lists to numpy arrays for fast processing
        arr1 = np.array(box_list1)
        arr2 = np.array(box_list2)

        # Merge all boxes to find the canvas boundaries
        all_boxes = np.vstack((arr1, arr2))

        # Determine canvas size and offset
        min_x = np.floor(np.min(all_boxes[:, 0])).astype(int)
        min_y = np.floor(np.min(all_boxes[:, 1])).astype(int)
        max_x = np.ceil(np.max(all_boxes[:, 2])).astype(int)
        max_y = np.ceil(np.max(all_boxes[:, 3])).astype(int)

        # Calculate width and height
        width = max_x - min_x
        height = max_y - min_y

        if width <= 0 or height <= 0:
            return 0.0

        # Create masks (Canvas)
        mask1 = np.zeros((height, width), dtype=bool)
        mask2 = np.zeros((height, width), dtype=bool)

        # Fill masks (draw union)
        for box in box_list1:
            x1 = int(np.floor(box[0])) - min_x
            y1 = int(np.floor(box[1])) - min_y
            x2 = int(np.ceil(box[2])) - min_x
            y2 = int(np.ceil(box[3])) - min_y

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            mask1[y1:y2, x1:x2] = True

        for box in box_list2:
            x1 = int(np.floor(box[0])) - min_x
            y1 = int(np.floor(box[1])) - min_y
            x2 = int(np.ceil(box[2])) - min_x
            y2 = int(np.ceil(box[3])) - min_y

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            mask2[y1:y2, x1:x2] = True

        # Calculate IoU
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union


# ========================================================================
# Format Reward (optional, for checking JSON format validity)
# ========================================================================
def format_reward(completions, **kwargs):
    """
    Check if the completion follows the expected JSON format.
    This rewards properly formatted outputs.

    Expected format for action_triggered:
    {
        "safe": bool,
        "safety_hazard": str,
        "target_object": [[x1, y1, x2, y2], ...],
        "constraint_object": [[x1, y1, x2, y2], ...],
        "principle_id": int
    }

    Expected format for environmental:
    {
        "safe": bool,
        "safety_hazard": str,
        "bbox_list": [{"label": str, "bounding_box": [x1, y1, x2, y2]}, ...]
    }
    """
    # Check for required fields regardless of order
    # action_triggered: safe, safety_hazard, target_object, constraint_object, principle_id
    required_fields_action = ["safe", "safety_hazard", "target_object", "constraint_object", "principle_id"]
    # environmental: safe, safety_hazard, bbox_list
    required_fields_env = ["safe", "safety_hazard", "bbox_list"]

    completion_contents = [completion[0]["content"] for completion in completions]
    results = []

    for content in completion_contents:
        # Try to parse as JSON
        try:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Check which format based on fields present
                if all(field in data for field in required_fields_action):
                    results.append(1.0)
                elif all(field in data for field in required_fields_env):
                    results.append(1.0)
                else:
                    results.append(0.0)
            else:
                results.append(0.0)
        except (json.JSONDecodeError, ValueError):
            results.append(0.0)

    return results


# ========================================================================
# Reward Registry
# ========================================================================
reward_funcs_registry = {
    "safe_accuracy": RiskGroundingRewards().safe_accuracy_reward,
    "safety_hazard_match": RiskGroundingRewards().safety_hazard_match_reward,
    "principle_accuracy": RiskGroundingRewards().principle_accuracy_reward,
    "iou": RiskGroundingRewards().iou_reward,  # Combined IoU (average of target and constraint)
    "iou_target_object": RiskGroundingRewards().iou_target_object_reward,
    "iou_constraint_object": RiskGroundingRewards().iou_constraint_object_reward,
    "format": format_reward,
}


# ========================================================================
# Prompt Templates
# ========================================================================

ACTION_TRIGGER_EVAL_TEMPLATE = """
You are a smart home safety assistant. Your role is to warn users about actions that may create safety hazards in the current scene.

User Instruction: "{instruction}"

Based on the scene shown in the input image, analyze whether executing the requested action would lead to a safety hazard. Explain your reasoning clearly.
"""
