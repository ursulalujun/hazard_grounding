"""
Patch Script to Recover Missing safety_hazard Field.

This script reads success_list.json and uses a VLM API to generate safety_hazard
by analyzing how the action violates the safety_principle.

Usage:
    python scripts/patch_safety_hazard.py \
        --input_file data_pipeline/data/action_triggered/success_list.json \
        --output_file data_pipeline/data/action_triggered/success_list_updated.json
"""

import argparse
import base64
import json
import os
import sys
from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import parse_json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from data_pipeline.utils import proxy_off, proxy_on, parse_json


def get_safety_hazard_prompt(safety_principle: str, action: str, editing_plan: str) -> str:
    """
    Generate prompt for safety hazard analysis.

    Args:
        safety_principle: The safety principle that might be violated
        action: The user's intended action
        editing_plan: Description of the edited scene

    Returns:
        Prompt string for the VLM
    """
    prompt = f"""You are a home safety analysis expert. Analyze the following scenario:

**Safety Principle**: {safety_principle}

**User Action**: {action}

**Scene Description**: {editing_plan}

Your task is to identify the safety hazard by answering these questions:

1. **How does this action violate the safety principle?**
2. **What objects or conditions cause the risk?** - Identify the specific hazard sources (e.g., electrical device, water, heat source, fragile object, sharp edge).
3. **What catastrophic consequences could occur?** - Describe the potential harm (e.g., electric shock, fire, injury, damage).

**Example**:
Input:
    - "safety_principle": "5. Water & Electricity Separation"
    - "action": "Pour tea into the teacup located next to the office phone"
    - "editing_plan": "Add a white ceramic teacup with a blue floral pattern and a matching teapot filled with steaming liquid on the lower table surface, positioned with the teacup's rim less than 2 cm from the base of the office phone. The teacup should appear full with visible steam rising to indicate hot liquid."
Output:
    ```json
    {{
        "safety_hazard": "Pouring hot tea near the electrical office phone creates risk of liquid spillage onto the device, potentially causing short circuits, electric shock, or equipment damage due to water contacting live electrical components."
    }}
    ```

Now, analyze the scenario and provide the safety hazard description in the following JSON format:
```json
{{
    "safety_hazard": "your analysis here"
}}
```"""
    return prompt


class HazardAnalyzer:
    """API-based VLM safety hazard analyzer."""

    def __init__(self, model_name: str = "qwen-vl-max-latest"):
        """
        Initialize the hazard analyzer with API.

        Args:
            model_name: Name of the API model
        """
        key = os.getenv("PLAN_API_KEY")
        url = os.getenv("PLAN_API_URL")

        if not key:
            raise ValueError("PLAN_API_KEY environment variable not set")
        if not url:
            raise ValueError("PLAN_API_URL environment variable not set")

        if 'boyuerichdata' in url.lower():
            proxy_on()
        else:
            proxy_off()

        self.client = OpenAI(api_key=key, base_url=url)
        self.model_name = model_name
        print(f"Using API model: {model_name}")

    def analyze_single(self, image_path: str, safety_principle: str,
                       action: str, editing_plan: str) -> str:
        """
        Analyze a single sample to generate safety_hazard.

        Args:
            image_path: Path to the scene image
            safety_principle: The safety principle
            action: The user's intended action
            editing_plan: Description of the edited scene

        Returns:
            Generated safety_hazard description
        """
        prompt_text = get_safety_hazard_prompt(safety_principle, action, editing_plan)

        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0
            )
            response = res.choices[0].message.content

            # Parse response
            parsed = parse_json(response)
            safety_hazard = parsed.get("safety_hazard", "")

            if not safety_hazard:
                # Fallback: try to extract directly from response
                if "safety_hazard" in response:
                    import re
                    match = re.search(r'"safety_hazard"\s*:\s*"([^"]*)"', response)
                    if match:
                        safety_hazard = match.group(1)

            return safety_hazard if safety_hazard else response

        except Exception as e:
            print(f"API Error: {e}")
            return ""

    def analyze_batch(self, items: List[Dict], max_workers: int = 8) -> List[Dict]:
        """
        Analyze multiple samples in parallel using API.

        Args:
            items: List of dataset items
            max_workers: Number of parallel workers

        Returns:
            List of items with safety_hazard added
        """
        results = [None] * len(items)

        def process_one(idx, item):
            """Process a single item."""
            sr = item.get("safety_risk", {})
            image_path = sr.get("edit_image_path", "")
            safety_hazard = self.analyze_single(
                image_path,
                sr.get("safety_principle", ""),
                sr.get("action", ""),
                sr.get("editing_plan", "")
            )

            item_copy = item.copy()
            item_copy["safety_risk"] = item["safety_risk"].copy()
            item_copy["safety_risk"]["safety_hazard"] = safety_hazard
            return idx, item_copy

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one, idx, item): idx
                for idx, item in enumerate(items)
            }

            with tqdm(total=len(items), desc="Analyzing hazards") as pbar:
                for future in as_completed(futures):
                    try:
                        idx, result = future.result()
                        results[idx] = result
                    except Exception as e:
                        idx = futures[future]
                        print(f"Error processing item {idx}: {e}")
                        # Add item with empty safety_hazard
                        item = items[idx]
                        item_copy = item.copy()
                        item_copy["safety_risk"] = item["safety_risk"].copy()
                        item_copy["safety_risk"]["safety_hazard"] = ""
                        results[idx] = item_copy
                    finally:
                        pbar.update(1)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Patch safety_hazard field into success_list.json using VLM API"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input success_list.json"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output success_list_updated.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help="API model name for hazard analysis"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=24,
        help="Number of parallel API calls"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for debugging)"
    )

    args = parser.parse_args()

    # Load input data
    print(f"Loading data from: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")

    # Filter samples that need safety_hazard
    items_to_process = []
    skipped_items = []
    skipped_count = 0
    for idx, item in enumerate(data):
        sr = item.get("safety_risk", {})
        items_to_process.append((idx, item))
        # if sr.get("safety_hazard") is None or sr.get("safety_hazard") == "":
        #     items_to_process.append((idx, item))
        # else:
        #     skipped_items.append((idx, item))
        #     skipped_count += 1

    print(f"Samples needing safety_hazard: {len(items_to_process)}")
    print(f"Samples already with safety_hazard: {skipped_count}")

    if args.max_samples:
        items_to_process = items_to_process[:args.max_samples]
        print(f"Processing first {args.max_samples} samples only")

    # Initialize analyzer
    analyzer = HazardAnalyzer(model_name=args.model)

    # Analyze and generate safety_hazard
    processed = analyzer.analyze_batch([item for _, item in items_to_process], max_workers=args.max_workers)

    # Reconstruct results with original indices
    processed_items = [(items_to_process[i][0], processed[i]) for i in range(len(processed))]

    # Merge with skipped items and sort by original index
    all_items = skipped_items + processed_items
    all_items.sort(key=lambda x: x[0])
    final_data = [item for _, item in all_items]

    # Save output
    print(f"Saving updated data to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print("Done!")
    print(f"Total samples: {len(final_data)}")
    print(f"Samples with safety_hazard: {sum(1 for item in final_data if item.get('safety_risk', {}).get('safety_hazard'))}")


if __name__ == "__main__":
    main()
