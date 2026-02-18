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

    def _parse_v1_output(self, raw_output: str, width: int = None, height: int = None) -> Dict:
        """
        Parse v1 format output from model.

        V1 format:
        ... [target_object][object_name][x1,y1,x2,y2][object_state]*n
        ... [constraint_object][object_name][x1,y1,x2,y2][object_state]*n ...

        [safety_hazard][...], violating[safety_principle][int....]

        Args:
            raw_output: Raw model output containing thinking process and answer
            width: Image width for bbox conversion (not used for now)
            height: Image height for bbox conversion (not used for now)

        Returns:
            Dict with keys: safe, safety_hazard, principle_id, target_object, constraint_object
        """
        result = {
            "safe": False,
            "safety_hazard": None,
            "principle_id": None,
            "target_object": [],
            "constraint_object": []
        }

        # Validate input
        if raw_output is None or not isinstance(raw_output, str):
            return result

        try:
            # Split thinking and answer by </think>
            if '</think>' in raw_output and raw_output.count('</think>') == 1:
                thinking, answer = raw_output.split('</think>')
            else:
                return result

            # Validate thinking and answer are strings
            if not isinstance(thinking, str) or not isinstance(answer, str):
                return result

            # Parse target_object from thinking process
            # Pattern: [target_object][object_name][x1,y1,x2,y2][object_state]
            target_pattern = r'\[target_object\]\[([^\]]+)\]\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\](?:\[([^\]]+)\])?'
            for match in re.finditer(target_pattern, thinking):
                try:
                    x1, y1, x2, y2 = map(int, match.groups()[1:5])
                    result["target_object"].append([x1, y1, x2, y2])
                except (ValueError, IndexError) as e:
                    # Skip invalid bbox coordinates
                    continue

            # Parse constraint_object from thinking process
            # Pattern: [constraint_object][object_name][x1,y1,x2,y2][object_state]
            constraint_pattern = r'\[constraint_object\]\[([^\]]+)\]\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\](?:\[([^\]]+)\])?'
            for match in re.finditer(constraint_pattern, thinking):
                try:
                    x1, y1, x2, y2 = map(int, match.groups()[1:5])
                    result["constraint_object"].append([x1, y1, x2, y2])
                except (ValueError, IndexError) as e:
                    # Skip invalid bbox coordinates
                    continue

            # Parse safety_hazard from answer
            # Pattern: [safety_hazard][...], violating[safety_principle][id(int)....]
            hazard_match = re.search(r'\[safety_hazard\]\[([^\]]*(?:\[[^\]]*\][^\]]*)*)\]', answer)
            if hazard_match:
                hazard_text = hazard_match.group(1)
                # Clean up nested brackets
                hazard_text = re.sub(r'\[([^\]]+)\]', r'\1', hazard_text)
                result["safety_hazard"] = hazard_text.strip()
            else:
                # Fallback: extract text before "violating"
                hazard_match = re.search(r'(.*?)\s*,?\s*violating', answer)
                if hazard_match:
                    result["safety_hazard"] = hazard_match.group(1).strip()

            # Parse principle_id from answer
            # Pattern: [safety_principle][int] or just id(...). Example: violating safety principle 4.
            principle_patterns = [
                r'\[safety_principle\]\[(\d+)',  # [safety_principle][4]
                r'violating\s+(?:safety\s+)?principle\s+(\d+)',  # violating principle 4
                r'principle\s+(\d+)',  # principle 4
            ]
            for pattern in principle_patterns:
                principle_match = re.search(pattern, answer, re.IGNORECASE)
                if principle_match:
                    try:
                        result["principle_id"] = int(principle_match.group(1))
                        break
                    except (ValueError, AttributeError):
                        continue

            # Determine if safe (if no hazard detected)
            if result["safety_hazard"] is None:
                result["safe"] = True
            elif isinstance(result["safety_hazard"], str):
                if "no safety hazard" in result["safety_hazard"].lower():
                    result["safe"] = True
                    result["safety_hazard"] = None

        except (AttributeError, TypeError, re.error) as e:
            # Return default result on any regex or string operation error
            pass

        return result

    def _parse_completions_once(self, completions) -> List[Dict]:
        """
        Parse all completions once and return parsed results.
        This avoids repeated parsing in each reward function.

        Args:
            completions: Model generated responses

        Returns:
            List of parsed result dicts
        """
        contents = [completion[0]["content"] for completion in completions]
        return [self._parse_v1_output(content) for content in contents]

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
            # Parse v1 format output
            parsed = self._parse_v1_output(content)
            pred_safe = parsed["safe"]
            gt_safe = gt_data.get("is_gt_safe", False)

            # Reward is 1.0 if prediction matches ground truth
            reward = 1.0 if pred_safe == gt_safe else 0.0

            rewards.append(reward)

        return rewards

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
            # Parse v1 format output
            parsed = self._parse_v1_output(content)
            pred_safe = parsed["safe"]
            pred_hazard = parsed["safety_hazard"]

            gt_hazard = gt_data.get("safety_hazard", "")
            gt_safe = gt_data.get("is_gt_safe", False)

            # If both are safe, perfect match
            if gt_safe:
                reward = 1.0 if pred_safe else 0.0
            # If GT is unsafe, check safety hazard description
            elif not gt_safe:
                if pred_safe or not pred_hazard:
                    reward = 0.0
                else:
                    # Compute embedding similarity
                    try:
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
                        similarity = cosine_similarity(
                            pred_emb.reshape(1, -1),
                            gt_emb.reshape(1, -1)
                        )[0, 0]
                        reward = float(similarity)
                    except Exception as e:
                        reward = 0.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards

    # ========================================================================
    # 3. Principle Accuracy Reward
    # ========================================================================
    def principle_accuracy_reward(self, completions, solution, **kwargs):
        """
        Compute reward based on whether the predicted principle_id matches ground truth.
        Only applies when the scene is unsafe (GT is not safe).

        Args:
            completions: Model generated responses
            solution: Ground truth data containing safety_principle

        Returns:
            List of rewards (1.0 if correct, 0.0 if incorrect)
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, gt_data in zip(contents, solution):
            # Parse v1 format output
            parsed = self._parse_v1_output(content)
            pred_safe = parsed["safe"]
            pred_principle_id = parsed["principle_id"]

            gt_safe = gt_data.get("is_gt_safe", False)

            # If GT is safe, principle_id should be None
            if gt_safe:
                reward = 1.0 if pred_principle_id is None else 0.0
            else:
                # GT is unsafe, check if prediction is correct
                # If model predicted safe, principle accuracy is 0
                if pred_safe:
                    reward = 0.0
                else:
                    # Model predicted unsafe, check principle_id
                    gt_principle_text = gt_data.get("safety_principle", "")
                    gt_principle_id = self._extract_principle_id(gt_principle_text)

                    if gt_principle_id is not None and pred_principle_id is not None:
                        reward = 1.0 if gt_principle_id == pred_principle_id else 0.0
                    else:
                        # Missing principle_id in either GT or prediction
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
            # Parse v1 format output
            parsed = self._parse_v1_output(content)
            pred_target_bboxes = parsed["target_object"]  # Normalized coordinates (0-1000)

            # Get GT target_object bboxes
            gt_target_bboxes = self._get_gt_target_bboxes(gt_data)

            # Get image dimensions for coordinate conversion
            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            # Convert predicted bboxes from normalized (0-1000) to pixel coordinates
            if img_width and img_height:
                pred_target_bboxes = self._normalized_to_pixel_bbox_list(pred_target_bboxes, img_width, img_height)

            # Convert to format expected by compute_list_iou (list of dicts with label and bounding_box)
            pred_target_bboxes_formatted = [
                {"label": f"bbox_{i}", "bounding_box": bbox}
                for i, bbox in enumerate(pred_target_bboxes)
            ] if pred_target_bboxes else []

            # Calculate IoU: target_object should ALWAYS be localized, regardless of scene safety
            if gt_target_bboxes:
                # GT has target_object bboxes, compute IoU
                iou = self.compute_list_iou(gt_target_bboxes, pred_target_bboxes_formatted)
                reward = iou
            else:
                # GT has NO target_object bboxes
                # Model should also NOT predict any
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
            # Parse v1 format output
            parsed = self._parse_v1_output(content)
            pred_safe = parsed["safe"]
            pred_constraint_bboxes = parsed["constraint_object"]  # Normalized coordinates (0-1000)

            gt_safe = gt_data.get("is_gt_safe", False)

            # Get GT constraint_object bboxes
            gt_constraint_bboxes = self._get_gt_constraint_bboxes(gt_data)

            # Get image dimensions for coordinate conversion
            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            # Convert predicted bboxes from normalized (0-1000) to pixel coordinates
            if img_width and img_height:
                pred_constraint_bboxes = self._normalized_to_pixel_bbox_list(pred_constraint_bboxes, img_width, img_height)

            # Convert to format expected by compute_list_iou (list of dicts with label and bounding_box)
            pred_constraint_bboxes_formatted = [
                {"label": f"bbox_{i}", "bounding_box": bbox}
                for i, bbox in enumerate(pred_constraint_bboxes)
            ] if pred_constraint_bboxes else []

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
                        iou = self.compute_list_iou(gt_constraint_bboxes, pred_constraint_bboxes_formatted)
                        reward = iou
                    else:
                        # No GT constraint bboxes (hazard from target's own state)
                        # Model should also predict empty constraint_object
                        reward = 1.0 if not pred_constraint_bboxes else 0.0

            rewards.append(reward)

        return rewards

    # ========================================================================
    # IoU Computation Functions
    # ========================================================================
    def _normalized_to_pixel_bbox_list(self, bboxes: List[List[int]], img_width: int, img_height: int) -> List[Dict]:
        """
        Convert bboxes from normalized coordinates (0-1000) to pixel coordinates.
        For action_triggered (v1) format.

        Args:
            bboxes: List of bbox lists [x1, y1, x2, y2] from parsed v1 output
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            List of bbox dicts with pixel coordinates
        """
        converted = []
        for i, bbox in enumerate(bboxes):
            if len(bbox) == 4:
                pixel_bbox = self._convert_normalized_bbox_to_pixel(bbox, img_width, img_height)
                converted.append({
                    "label": f"bbox_{i}",
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

    # ========================================================================
    # IoU Computation Functions
    # ========================================================================
    def _convert_normalized_bbox_to_pixel(self, bbox: List[int], img_width: int, img_height: int) -> List[int]:
        """
        Convert a single normalized bbox (0-1000) to pixel coordinates.
        Matches the logic in data_pipeline/utils.py:bbox_norm_to_pixel

        Args:
            bbox: Normalized bbox [x1, y1, x2, y2] in range [0, 1000]
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Pixel bbox [x_min, y_min, x_max, y_max] with coordinates corrected
        """
        abs_y1 = int(bbox[1] / 1000 * img_height)
        abs_x1 = int(bbox[0] / 1000 * img_width)
        abs_y2 = int(bbox[3] / 1000 * img_height)
        abs_x2 = int(bbox[2] / 1000 * img_width)

        # Ensure x1 <= x2 and y1 <= y2
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        return [abs_x1, abs_y1, abs_x2, abs_y2]

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
    Check if the completion follows the expected v1 format.
    This rewards properly formatted outputs.

    V1 format (thinking process with structured output):
    - Contains </think> section with target_object and constraint_object tags
    - Contains answer section with safety_hazard and principle_id
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    results = []

    for content in completion_contents:
        # Check for v1 format markers
        has_thinking = '</think>' in content and content.count('</think>') == 1
        has_safety_hazard = '[safety_hazard]' in content

        if has_thinking and has_safety_hazard:
            results.append(1.0)
        else:
            results.append(0.0)

    return results


# ========================================================================
# Batch Rewards Wrapper (caches parsed results to avoid repeated parsing)
# ========================================================================
class BatchRewardsWrapper:
    """
    Wrapper that caches parsed completions to avoid repeated parsing.
    This is useful when multiple reward functions are called on the same completions.
    Only caches the current batch - automatically clears when a new batch arrives.
    """

    def __init__(self):
        """Initialize the wrapper."""
        self.calculator = RiskGroundingRewards()
        self._current_batch_key = None
        self._current_parsed_results = None

    def _get_cached_parsed(self, completions):
        """
        Get cached parsed results or parse and cache them.
        Only caches the current batch - clears cache when a new batch arrives.
        """
        # Create a hashable key from completions
        contents_key = tuple(c[0]["content"] for c in completions)

        # Check if this is a new batch
        if contents_key != self._current_batch_key:
            # New batch - clear old cache and parse new batch
            self._current_batch_key = contents_key
            self._current_parsed_results = self.calculator._parse_completions_once(completions)

        return self._current_parsed_results

    def clear_cache(self):
        """Manually clear the parsing cache."""
        self._current_batch_key = None
        self._current_parsed_results = None

    def safe_accuracy_reward(self, completions, solution, **kwargs):
        """Safe accuracy reward using cached parsed results."""
        parsed_list = self._get_cached_parsed(completions)
        rewards = []
        for parsed, gt_data in zip(parsed_list, solution):
            pred_safe = parsed["safe"]
            gt_safe = gt_data.get("is_gt_safe", False)
            reward = 1.0 if pred_safe == gt_safe else 0.0
            rewards.append(reward)
        return rewards

    def safety_hazard_match_reward(self, completions, solution, **kwargs):
        """Safety hazard match reward using cached parsed results."""
        parsed_list = self._get_cached_parsed(completions)
        rewards = []

        for parsed, gt_data in zip(parsed_list, solution):
            pred_safe = parsed["safe"]
            pred_hazard = parsed["safety_hazard"]
            gt_hazard = gt_data.get("safety_hazard", "")
            gt_safe = gt_data.get("is_gt_safe", False)

            if gt_safe:
                reward = 1.0 if pred_safe else 0.0
            elif not gt_safe:
                if pred_safe or not pred_hazard:
                    reward = 0.0
                else:
                    try:
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
                        similarity = cosine_similarity(
                            pred_emb.reshape(1, -1),
                            gt_emb.reshape(1, -1)
                        )[0, 0]
                        reward = float(similarity)
                    except Exception as e:
                        reward = 0.0
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards

    def principle_accuracy_reward(self, completions, solution, **kwargs):
        """Principle accuracy reward using cached parsed results."""
        parsed_list = self._get_cached_parsed(completions)
        rewards = []

        for parsed, gt_data in zip(parsed_list, solution):
            pred_safe = parsed["safe"]
            pred_principle_id = parsed["principle_id"]
            gt_safe = gt_data.get("is_gt_safe", False)

            if gt_safe:
                reward = 1.0 if pred_principle_id is None else 0.0
            else:
                if pred_safe:
                    reward = 0.0
                else:
                    # Try to get principle_id directly from gt_data first (new format)
                    gt_principle_id = gt_data.get("principle_id")

                    # Fallback to parsing from safety_principle text (old format)
                    if gt_principle_id is None:
                        gt_principle_text = gt_data.get("safety_principle", "")
                        gt_principle_id = self.calculator._extract_principle_id(gt_principle_text)

                    if gt_principle_id is not None and pred_principle_id is not None:
                        reward = 1.0 if gt_principle_id == pred_principle_id else 0.0
                    else:
                        reward = 0.0

            rewards.append(reward)

        return rewards

    def iou_target_object_reward(self, completions, solution, **kwargs):
        """Target object IoU reward using cached parsed results."""
        parsed_list = self._get_cached_parsed(completions)
        rewards = []

        for parsed, gt_data in zip(parsed_list, solution):
            pred_target_bboxes = parsed["target_object"]
            gt_target_bboxes = self.calculator._get_gt_target_bboxes(gt_data)

            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            if img_width and img_height:
                pred_target_bboxes = self.calculator._normalized_to_pixel_bbox_list(
                    pred_target_bboxes, img_width, img_height
                )

            pred_target_bboxes_formatted = [
                {"label": f"bbox_{i}", "bounding_box": bbox["bounding_box"]}
                for i, bbox in enumerate(pred_target_bboxes)
            ] if pred_target_bboxes else []

            if gt_target_bboxes:
                iou = self.calculator.compute_list_iou(gt_target_bboxes, pred_target_bboxes_formatted)
                reward = iou
            else:
                reward = 1.0 if not pred_target_bboxes else 0.0

            rewards.append(reward)

        return rewards

    def iou_constraint_object_reward(self, completions, solution, **kwargs):
        """Constraint object IoU reward using cached parsed results."""
        parsed_list = self._get_cached_parsed(completions)
        rewards = []

        for parsed, gt_data in zip(parsed_list, solution):
            pred_safe = parsed["safe"]
            pred_constraint_bboxes = parsed["constraint_object"]
            gt_safe = gt_data.get("is_gt_safe", False)
            gt_constraint_bboxes = self.calculator._get_gt_constraint_bboxes(gt_data)

            img_width = gt_data.get("image_width")
            img_height = gt_data.get("image_height")

            if img_width and img_height:
                pred_constraint_bboxes = self.calculator._normalized_to_pixel_bbox_list(
                    pred_constraint_bboxes, img_width, img_height
                )

            pred_constraint_bboxes_formatted = [
                {"label": f"bbox_{i}", "bounding_box": bbox["bounding_box"]}
                for i, bbox in enumerate(pred_constraint_bboxes)
            ] if pred_constraint_bboxes else []

            if gt_safe:
                reward = 1.0 if not pred_constraint_bboxes else 0.0
            else:
                if pred_safe:
                    reward = 0.0
                else:
                    if gt_constraint_bboxes:
                        iou = self.calculator.compute_list_iou(gt_constraint_bboxes, pred_constraint_bboxes_formatted)
                        reward = iou
                    else:
                        reward = 1.0 if not pred_constraint_bboxes else 0.0

            rewards.append(reward)

        return rewards


# Global batch rewards instance
_batch_rewards = BatchRewardsWrapper()


def get_batch_reward_registry():
    """
    Get reward functions that use cached parsing to avoid repeated work.
    Use this instead of reward_funcs_registry for better performance.
    """
    return {
        "safe_accuracy": _batch_rewards.safe_accuracy_reward,
        "safety_hazard_match": _batch_rewards.safety_hazard_match_reward,
        "principle_accuracy": _batch_rewards.principle_accuracy_reward,
        "iou_target_object": _batch_rewards.iou_target_object_reward,
        "iou_constraint_object": _batch_rewards.iou_constraint_object_reward,
        "format": format_reward,
    }


# ========================================================================
# Prompt Templates
# ========================================================================

ACTION_TRIGGER_EVAL_TEMPLATE = """
You are a smart home safety assistant. Your role is to warn users about actions that may create safety hazards in the current scene.

User Instruction: "{action}"

Based on the scene shown in the input image, analyze whether executing the requested action would lead to a safety hazard. Explain your reasoning clearly.
"""
