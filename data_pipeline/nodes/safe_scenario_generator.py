"""
Object Requirement Analyzer: Analyzes task instructions to identify required objects
and generates editing plans to add any missing objects to the scene.

This script reads base images with action instructions, analyzes what objects are
required to perform the task, checks if they exist in the image, and generates
editing plans to add any missing objects.
"""

import argparse
import base64
import json
import openai
import os
from tqdm import tqdm
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import parse_json
from typing import Optional, Dict, Any, List

from nodes.principle_tracker import (
    ACTION_TRIGGERED_PRINCIPLES
)


SAFE_SCENARIO_TEMPLATE = """
You are an expert AI assistant specializing in domestic scene understanding and object requirement analysis. Your task is to analyze a task instruction and identify what objects are required to perform it, then check if those objects exist in the given image.

**Context:**
We have a base image and a task instruction. Your goal is to:
1. Analyze what objects are required to perform the task
2. Check if those objects exist in the image
3. Generate an editing plan to add any missing objects

**Input Information:**
- Scene Type: {scene_type}
- Task Instruction: {action}

**Process:**

1. **Analyze Required Objects:** Identify ALL objects that are required to perform the task, including:
   - Direct target objects (what the action is performed on)
   - Supporting objects (tools, containers, materials needed)
   - Contextual objects (furniture, fixtures that are part of the task)

2. **Check Object Presence:** Examine the image to determine which required objects are:
   - **Present:** The object exists in the image and is accessible
   - **Missing:** The object does not exist or is not visible in the image

3. **Generate Editing Plan:** For each missing object, provide detailed instructions to add it to the scene. If all required objects are present, output `null` for editing_plan.

**Examples for Reference:**

- **Task: "Wipe the dust off the lamp on the nightstand"**
  - Required objects: lamp, nightstand, cloth/wipe
  - If lamp is missing: Add a table lamp on the nightstand
  - If nightstand is missing: Add a nightstand beside the bed
  - If cloth is missing (optional): Can be ignored or added

- **Task: "Replace the lightbulb on the ceiling"**
  - Required objects: lightbulb, ladder/chair, replacement bulb
  - If lightbulb/fixture is missing: Add a ceiling light fixture
  - If climbing support is missing: Add a ladder or chair

- **Task: "Clean the toilet using cleaning agents"**
  - Required objects: toilet, cleaning agents (bleach, cleaner)
  - If toilet is missing: Add a toilet
  - If cleaning agents are missing: Add cleaning supplies near toilet

**Output Format:**

Provide your response in a single JSON block.

- **If missing objects need to be added:**

    ```json
    {{
        "action": "{action}",
        "editing_plan": str, # "[A clear, concise description of objects to ADD to the scene]",
        "hazard_related_area": {{
            "target_object": list[str], # "[Objects that are required as direct targets of the action]"
            "constraint_object": []  # "please output an empty list"
        }}
    }}
    ```

- **If ALL required objects are present in the image:**
    Output `null`.

**Critical Rules for `editing_plan`:**

1. **Scene Realism:** Any added objects must match `scene_type` and fit the logic of a standard home.

2. **Minimal Editing:** Only add objects that are genuinely missing and necessary for the task.

3. **Detailed Visual Descriptions:** The `editing_plan` must be extremely detailed with:
   - **Attributes:** Size, Material, Color, Texture, State
   - **Spatial Relationships:** Exact positioning relative to existing objects
   - **Integration:** How the object fits naturally into the scene

4. **Explicit Visual Evidence:** Describe concrete visual cues, not abstract concepts.
   - Bad: "Add a lamp"
   - Good: "Add a white ceramic table lamp (15cm height) with a fabric shade on the wooden nightstand (40cm height) positioned to the right of the bed. The lamp has a cylindrical base (8cm diameter) and is placed 10cm from the edge of the nightstand."

5. **Multiple Objects:** If multiple objects are missing, describe ALL of them in the editing_plan with their individual positions.

6. **Do Not Duplicate Existing Objects:** If a target object already exists in the image, do NOT add it again in the editing_plan. Only list it in the `target_object` output list.

For `target_object`:
1. List ALL objects that the action directly interacts with, including both:
   - The primary target object (what is being acted upon)
   - Tools/objects used to perform the action (e.g., cleaning cloth for wiping, spray bottle for cleaning)
2. Examples:
   - Action: "Wipe the table", editing_plan adds a cloth and ink stain → target_object: ["ink stain in the desk", "white microfiber cleaning cloth"]
   - Action: "Clean the toilet using cleaning agents", editing_plan adds spray bottle → target_object: ["toilet", "white plastic spray bottle"]

**Your input:**
- scene_type: {scene_type}
- action: {action}

Just give your output in **JSON format (```json ... ```)**, do not include other information. If all required objects are present, output `null`. DO NOT add objects that do not match the `scene_type`.
"""


class ObjectRequirementAnalyzer:
    def __init__(self, model: str):
        """
        Initialize the ObjectRequirementAnalyzer.

        Args:
            model: Name of the VLM model to use for analysis
        """
        # Setup API client
        key = os.getenv("PLAN_API_KEY")
        url = os.getenv("PLAN_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.model = model

    def analyze_object_requirements(
        self,
        original_plan: Dict[str, Any],
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze task instruction to identify required objects and generate
        editing plans to add any missing objects.

        Args:
            original_plan: Dictionary containing the image_path, scene_type, and action
            min_pixels: Minimum pixels for image encoding
            max_pixels: Maximum pixels for image encoding
            max_retries: Maximum number of retries for API calls

        Returns:
            Dictionary containing the object requirement analysis result,
            or None if analysis failed
        """
        # Extract information from original plan
        image_path = original_plan.get("image_path")
        scene_type = original_plan.get("scene_type")
        safety_risk = original_plan.get("safety_risk", {})

        # Get action and safety_principle from safety_risk
        action = safety_risk.get("action", "")
        safety_principle = safety_risk.get("safety_principle", "")

        if not action:
            return {
                "image_path": image_path,
                "scene_type": scene_type,
                "safety_risk": None,
                "state": "no_action_provided"
            }

        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load and encode the base image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Format the prompt with task information
        prompt = SAFE_SCENARIO_TEMPLATE.format(
            action=action,
            scene_type=scene_type
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                ).choices[0].message.content

                if "</think>" in response:
                    response = response.split("</think>")[-1]
                try:
                    result = parse_json(response)
                except Exception as e:
                    print(f"Parse Error: {e}")
                    result = None
                # If VLM returned null, all required objects are present
                if result is None:
                    return {
                        "image_path": image_path,
                        "scene_type": scene_type,
                        "safety_risk": None,
                        "state": "all_objects_present"
                    }


                result['safety_principle'] = safety_principle
                result['safety_hazard'] = None
                result['pre_image_path'] = image_path
                return {
                    "image_path": image_path,
                    "scene_type": scene_type,
                    "safety_risk": result
                }

            except json.JSONDecodeError as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] JSON parsing failed for {os.path.basename(image_path)} | Error: {e}")
                print(f"   Response snippet: {response[:200]}...")

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(image_path)}")
                    return None

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] Object requirement analysis failed for {os.path.basename(image_path)} | Error: {e}")
                traceback.print_exc()

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(image_path)}")
                    return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze task instructions to identify and add missing objects"
    )
    parser.add_argument(
        '--root_folder',
        type=str,
        default="data"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-235B-A22B-Thinking',
        help='VLM model to use for analysis'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=24,
        help='Maximum number of concurrent workers'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of samples to process (for testing)'
    )

    args = parser.parse_args()

    input = os.path.join(args.root_folder, "success_list.json")
    output = os.path.join(args.root_folder, "safepair", "editing_plan.json")

    # Load input data
    print(f"📂 Loading data from {input}...")
    with open(input, 'r') as f:
        editing_plans = json.load(f)

    # Apply limit if specified
    if args.limit:
        editing_plans = editing_plans[:args.limit]
        print(f"⚠️ Processing limited to {args.limit} samples")

    print(f"✅ Loaded {len(editing_plans)} editing plans")

    # Initialize analyzer
    analyzer = ObjectRequirementAnalyzer(
        model=args.model
    )

    # Process plans
    print(f"🚀 Analyzing object requirements using {args.model}...")

    results = []
    failed = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(analyzer.analyze_object_requirements, plan): i
            for i, plan in enumerate(editing_plans)
        }

        with tqdm(total=len(editing_plans), desc="🖼️ Analyzing object requirements") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]

                try:
                    result = future.result()

                    if result is not None:
                        if result.get("safety_risk") is not None:
                            results.append(result)
                        else:
                            skipped += 1
                            # Still save skipped items for tracking
                            results.append(result)
                    else:
                        failed.append(idx)

                except Exception as e:
                    print(f"\n❌ Error processing sample {idx}: {e}")
                    traceback.print_exc()
                    failed.append(idx)

                finally:
                    pbar.update(1)

    # Save results
    print(f"\n💾 Saving results to {output}...")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else '.', exist_ok=True)
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    valid_with_edits = sum(1 for r in results if r.get("safety_risk") is not None)
    print(f"\n{'='*50}")
    print(f"📊 Summary:")
    print(f"  Total processed: {len(editing_plans)}")
    print(f"  Requires adding objects: {valid_with_edits}")
    print(f"  All objects present (skipped): {skipped}")
    print(f"  Failed: {len(failed)}")
    print(f"{'='*50}")
    print(f"✅ Done! Results saved to {output}")


if __name__ == "__main__":
    main()
