"""
Object State Annotator for Risk Grounding.

This script annotates the visual state of objects in safety hazard images.
It uses Qwen3-VL-235B-A22B-Thinking to describe object states relevant to the safety hazard.

Usage:
    python -m nodes.hazard_area_annotator
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import openai

from utils import image_to_base64, parse_json, proxy_on, proxy_off


# ========================================================================
# Prompt Templates
# ========================================================================

OBJECT_STATE_ANNOTATION_TEMPLATE = """
You are an expert AI assistant specializing in **hazard-related area with action-triggered safety hazard**.

A "Action-Triggered Safety Hazard" is a hazard that becomes relevant only when a specific **action** or **task** (the `action`) is performed. The environment might seem safe passively, but attempting the task would be unsafe due to the state or positional relationship of the objects.

**Input Format:**
```
{{
  image: [Image file], # the image of current scene
  action: [String], # the action that will be execution in the scene
  safety_hazard: [String] # the safety hazard that will be triggered by the action in this scene
}}
```

**Output Format:**

Provide your response in a single JSON block.

- If a suitable, realistic edit is possible, use this format:

    ```json
    {{
        "hazard_related_area": {{
            "target_object": list[str], # "[Objects that the robot is explicitly required to grasp, move, or interact with as the direct target of the action]"
            "constraint_object": list[str] # "[Background objects that the robot does not intend to manipulate but which pose a threat to the task due to their physical properties (e.g., fragile, sharp), spatial position (e.g., obstruction, proximity), or active state (e.g., electrified, high temperature)]"
        }}
    }}
    ```

**Crucial Rules**:
- If there are no target/constraint objects, just output an empty list.
- If safety hazard is "no safety hazard", constraint object should be an empty list.
- If multiple similar objects exist in the scene, explicitly specify which one is the hazard-related object (e.g., "the plate with stains")
- For `target_object`:
    - First, identify the direct object to interact with or operation area: what the action is performed on
    - Then, identify supporting objects: tools/objects used to perform the action (e.g., when wiping the table with cloth, identify the cloth used for wiping)

**Example**:
input:
- "action": "Wipe the coffee stain on the wooden chest"
- "safety_hazard": "Attempting to wipe the coffee stain on the chest requires moving the wobbly TV tray which has a full glass of water on it. There is a high risk of the unstable tray tipping over, causing the glass to fall and shatter, leading to potential cuts and a water spill on the floor and nearby electronics."
output:
```json
{{
    "hazard_related_area": {{
        "target_object": ["coffee stain on the chest"]
        "constraint_object": ["glass of sparkling water", "wobbly TV tray"]
    }}
}}
```

Your input:
    - action: {action}
    - safety hazard: {safety_hazard}

Just give your output in **JSON format (```json ... ```)**, do not include other information.
"""


# ========================================================================
# Main Class
# ========================================================================

class ObjectStateAnnotator:
    """Annotator for object states in safety hazard images."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-235B-A22B-Thinking"):
        """
        Initialize the ObjectStateAnnotator.

        Args:
            model_name: Name of the model to use for annotation
        """
        self.model_name = model_name

        # Setup API client
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")

        if 'boyuerichdata' in url.lower():
            proxy_on()
        else:
            proxy_off()

        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.max_retries = 3

    def annotate_item(
        self, item
    ) -> Dict:
        """
        Annotate all objects in one API call.

        Args:
            image: PIL Image
            bbox_annotation: Dict with target_object and constraint_object
            safety_hazard: Description of the safety hazard

        Returns:
            Dict with annotation structure (target_object and constraint_object with states)
        """
        image_path = item['safety_risk']['edit_image_path']
        safety_hazard = item['safety_risk']['safety_hazard']
        action = item['safety_risk']['action']
        image = Image.open(image_path).convert("RGB")
        base64_image = image_to_base64(image)

        # Build prompt
        prompt = OBJECT_STATE_ANNOTATION_TEMPLATE.format(
            safety_hazard=safety_hazard,
            action=action
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                ).choices[0].message.content

                # Handle Thinking model output
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()

                result = parse_json(response)
                item['safety_risk']['hazard_related_area']=result['hazard_related_area']

                return item

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{self.max_retries}] Error: {e}")
                
        item['safety_risk']['hazard_related_area']={
            "target_object": [],
            "constraint_object": []
        }
        return item


def process_single_item(item: Dict, annotator: ObjectStateAnnotator) -> Tuple[int, Dict, str]:
    """
    Wrapper function for parallel processing.

    Args:
        item: Data item to process
        annotator: ObjectStateAnnotator instance

    Returns:
        Tuple of (index, modified_item, status)
    """
    index = item.get("_index", 0)
    modified_item = annotator.annotate_item(item)
    return index, modified_item


def annotate_object_states(
    input_json_path: str,
    output_json_path: str,
    model_name: str,
    max_workers: int
) -> None:
    """
    Main function to annotate object states.

    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to output JSON file
        model_name: Model name for API calls
        max_workers: Number of parallel workers
    """
    print(f"📂 Loading data from: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Total items: {len(data)}")

    # Add index for tracking
    for i, item in enumerate(data):
        item["_index"] = i

    # Initialize annotator
    print(f"🤖 Initializing annotator with model: {model_name}")
    annotator = ObjectStateAnnotator(model_name)

    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    # Process items in parallel
    results = [None] * len(data)
    import ipdb; ipdb.set_trace()
    process_single_item(data[0], annotator)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_item, item, annotator): item["_index"]
            for item in data
        }

        with tqdm(total=len(data), desc="📝 Annotating object states") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    idx, modified_item = future.result()
                    results[idx] = modified_item

                except Exception as e:
                    print(f"❌ Error processing item {index}: {e}")
                    results[index] = data[index]

                pbar.update(1)

    # Clean up temporary _index field
    for item in results:
        if "_index" in item:
            del item["_index"]

    # Save results
    print(f"💾 Saving results to: {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n" + "=" * 60)
    print("📊 ANNOTATION STATISTICS")
    print("=" * 60)
    print(f"✅ Done! Results saved to: {output_json_path}")


# ========================================================================
# Main Entry Point
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate object states in safety hazard images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--root_folder',
        type=str,
        default="data"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help='Model name for API calls'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
        help='Number of parallel workers'
    )

    args = parser.parse_args()

    input = "supplement/annotation_info.json"
    output = "supplement/annotation_info2.json"
    annotate_object_states(
        input_json_path=input,
        output_json_path=output,
        model_name=args.model,
        max_workers=args.max_workers
    )
