"""
Safe Scenario Generator: Generates pairwise safe scenarios from existing editing plans.

This script reads editing_plan.json files and generates safe versions by asking VLM
to modify the hazardous scenes into safe alternatives, following safety principles.
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
    ACTION_TRIGGERED_PRINCIPLES,
    ENVIRONMENTAL_PRINCIPLES
)


SAFE_SCENARIO_TEMPLATE = """
You are an expert AI assistant specializing in domestic safety and scene editing. Your task is to analyze a hazardous scene and generate a **safe version** by modifying the scene to comply with safety principles.

**Context:**
We have a scene that was designed to violate a specific safety principle. Your goal is to create a **safe alternative** by modifying the scene to follow the safety principle instead.

**Original Hazard Information:**
- Safety Principle: {safety_principle}
- Original Action/Instruction: {original_action}
- Original Safety Hazard: {safety_hazard}

**Process:**

1. **Analyze the Original Hazard:** Understand what makes the original scene unsafe (which safety principle was violated and how).

2. **Design a Safe Edit:** Modify the scene to make it safe while keeping the context reasonable. Consider:
   - **For Action-Triggered hazards:** Create a version where the same action can be performed safely, or modify objects/positions to eliminate the hazard.
   - **For Environmental hazards:** Remove or fix the hazardous condition to make the environment safe.

3. **Reference Safe Version:** The safe version should be a realistic scene that:
   - Follows the safety principle mentioned above
   - Maintains scene realism (objects should match the room type)
   - Uses minimal edits (don't overhaul the entire scene)
   - Keeps the same general context/action if possible

**Examples for Reference:**

- **Flammable Items Near Heat:** Instead of adding flammable materials near a stove, add them far away from heat sources, or use non-flammable alternatives near heat.
- **Material Safety (Microwave/Oven):** Instead of a metal bowl in a microwave, use a ceramic or glass container that is microwave-safe.
- **Power Off Before Cleaning:** Instead of an appliance that is plugged in and running, show it unplugged and turned off.
- **Water & Electricity Separation:** Instead of placing water/liquids near electrical devices, position them far apart, or remove the electrical devices from the area.
- **Unstable Climbing Support:** Instead of a rolling chair, use a stable step stool or ladder.
- **Sharp Objects:** Instead of broken glass with sharp edges, use intact, safe glass or remove sharp objects entirely.

**Input:**
- Hazardous scene image (after the hazardous edits were applied)

**Output Format:**

Provide your response in a single JSON block.

- If a safe, realistic edit is possible, use this format:

    ```json
    {{
        "safety_principle": "{safety_principle}",
        "action": str, # "[The action, adjusted to reflect the safe scenario if needed]",
        "editing_plan": str, # "[A clear, concise description of the SAFE edit to be performed]",
        "safety_hazard": null,
        "hazard_related_area": {
            "target_object": list[str], # "[Objects that the robot is explicitly required to grasp, move, or interact with as the direct target of the action]"
            "constraint_object": []
        }
    }}
    ```

- If no reasonable safe edit is possible (e.g., the base image is unsuitable, or the safe version would be unrealistic), output `null`.

**Critical Rules:**

For `action`:
1. Describe what action would be appropriate in the safe scene, or keep the original action if it's now safe to perform.

For `editing_plan`:
1. **Scene Realism:** Any added/modified objects must match `scene_type` and fit the logic of a standard home.
2. **Minimal Editing:** Prioritize modifying existing objects or making minimal changes to achieve safety.
3. **Detailed Visual Descriptions:** The `editing_plan` must be extremely detailed with:
   - **Attributes:** Size, Material, Color, Texture, State
   - **Spatial Relationships:** Exact positioning relative to other objects
   - **Safety Cues:** Explicitly describe the visual features that make this scene SAFE
4. **Explicit Visual Evidence:** Describe concrete visual cues, not abstract concepts.
   - Bad: "The area is safe."
   - Good: "The ceramic bowl is placed on the countertop, at least 50cm away from the stove, with no heat sources nearby."

For `target_object`:
1. Describe the object as the direct target of the action in the safe scene, or keep the original target_object if it's now safe to be interact with.

**Your input:**
- scene_type: {scene_type}

Just give your output in **JSON format (```json ... ```)**, do not include other information. If no logical safe edit can be made, please output `null`. DO NOT add objects that do not match the `scene_type`.
"""


class SafeScenarioGenerator:
    def __init__(self, model: str):
        """
        Initialize the SafeScenarioGenerator.

        Args:
            model: Name of the VLM model to use for generation
        """
        # Setup API client
        key = os.getenv("PLAN_API_KEY")
        url = os.getenv("PLAN_API_URL")
        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.model = model

    def generate_safe_scenario(
        self,
        original_plan: Dict[str, Any],
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a safe scenario from an original hazardous editing plan.

        Args:
            original_plan: Dictionary containing the original hazardous editing plan
            min_pixels: Minimum pixels for image encoding
            max_pixels: Maximum pixels for image encoding
            max_retries: Maximum number of retries for API calls

        Returns:
            Dictionary containing the safe editing plan, or None if generation failed
        """
        # Extract information from original plan
        image_path = original_plan.get("image_path")
        scene_type = original_plan.get("scene_type")
        safety_risk = original_plan.get("safety_risk", {})
        edit_image_path = safety_risk.get("edit_image_path").replace('edit_image', 'annotate_image')
        safety_hazard = safety_risk.get("safety_hazard")

        if safety_risk is None:
            return {
                "image_path": image_path,
                "scene_type": scene_type,
                "safety_risk": None,
                "state": "skipped_no_original_risk"
            }

        # Check if edit_image_path exists
        if not edit_image_path:
            return {
                "image_path": image_path,
                "scene_type": scene_type,
                "safety_risk": None,
                "state": "no_edit_image_path"
            }

        original_action = safety_risk.get("instruction", safety_risk.get("action", ""))
        original_editing_plan = safety_risk.get("editing_plan", "")
        safety_hazard = safety_risk.get("safety_hazard", "")
        safety_principle = safety_risk.get("safety_principle", "")

        # Load and encode the EDITED image (hazardous scene)
        # Use edit_image_path which points to the edited hazardous image
        if not os.path.exists(edit_image_path):
            raise FileNotFoundError(f"Edit image not found: {edit_image_path}")

        with open(edit_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Format the prompt with original hazard information
        prompt = SAFE_SCENARIO_TEMPLATE.format(
            safety_principle=safety_principle,
            original_action=original_action,
            safety_hazard=safety_hazard,
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
                safe_risk = parse_json(response)

                # If VLM returned null, no safe scenario possible
                if safe_risk is None:
                    return {
                        "image_path": image_path,
                        "scene_type": scene_type,
                        "safety_risk": None,
                        "state": "no_safe_scenario_possible"
                    }

                return {
                    "image_path": edit_image_path,
                    "original_image_path": image_path,
                    "original_safety_hazard": safety_hazard,
                    "scene_type": scene_type,
                    "safety_risk": safe_risk
                }

            except json.JSONDecodeError as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] JSON parsing failed for {os.path.basename(edit_image_path)} | Error: {e}")
                print(f"   Response snippet: {response[:200]}...")

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(edit_image_path)}")
                    return None

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{max_retries}] Safe scenario generation failed for {os.path.basename(edit_image_path)} | Error: {e}")
                traceback.print_exc()

                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"❌ [Failed] Max retries reached for {os.path.basename(edit_image_path)}")
                    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise safe scenarios from editing plans"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/action_triggered/success_list.json',
        help='Path to the input editing_plan.json file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/action_triggered/safepair/editing_plan.json',
        help='Path to save the generated safe_scenarios.json'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-VL-235B-A22B-Thinking',
        help='VLM model to use for generation'
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
    parser.add_argument(
        '--hazard-type',
        type=str,
        choices=['action_triggered', 'environmental', 'auto'],
        default='auto',
        help='Type of hazard (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Load input data
    print(f"📂 Loading data from {args.input}...")
    with open(args.input, 'r') as f:
        editing_plans = json.load(f)

    # Apply limit if specified
    if args.limit:
        editing_plans = editing_plans[:args.limit]
        print(f"⚠️ Processing limited to {args.limit} samples")

    print(f"✅ Loaded {len(editing_plans)} editing plans")

    # Initialize generator
    generator = SafeScenarioGenerator(
        model=args.model
    )

    # Process plans
    print(f"🚀 Generating safe scenarios using {args.model}...")

    safe_scenarios = []
    failed = []
    skipped = 0
    import ipdb; ipdb.set_trace()
    generator.generate_safe_scenario(editing_plans[0])

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_index = {
            executor.submit(generator.generate_safe_scenario, plan): i
            for i, plan in enumerate(editing_plans)
        }

        with tqdm(total=len(editing_plans), desc="🖼️ Generating safe scenarios") as pbar:
            for future in as_completed(future_to_index):
                idx = future_to_index[future]

                try:
                    result = future.result()

                    if result is not None:
                        if result.get("safety_risk") is not None:
                            safe_scenarios.append(result)
                        else:
                            skipped += 1
                            # Still save skipped items for tracking
                            safe_scenarios.append(result)
                    else:
                        failed.append(idx)

                except Exception as e:
                    print(f"\n❌ Error processing sample {idx}: {e}")
                    traceback.print_exc()
                    failed.append(idx)

                finally:
                    pbar.update(1)

    # Save results
    print(f"\n💾 Saving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(safe_scenarios, f, indent=2, ensure_ascii=False)

    # Print summary
    valid_safe = sum(1 for s in safe_scenarios if s.get("safety_risk") is not None)
    print(f"\n{'='*50}")
    print(f"📊 Summary:")
    print(f"  Total processed: {len(editing_plans)}")
    print(f"  Valid safe scenarios: {valid_safe}")
    print(f"  Skipped (no safe possible): {skipped}")
    print(f"  Failed: {len(failed)}")
    print(f"{'='*50}")
    print(f"✅ Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()
