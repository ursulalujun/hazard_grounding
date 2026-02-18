"""
CoT Generator for Safety Hazard Recognition.

This script generates Chain of Thought (CoT) reasoning for safety hazard identification,
which can be used as Ground Truth for SFT training.

Usage:
    python -m nodes.cot_generator \\
        --input data/success_list_annotated.json \\
        --output data/success_list_with_cot.json \\
        --max_workers 24
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import openai

from utils import image_to_base64, proxy_on, proxy_off, parse_json


# ========================================================================
# Prompt Templates
# ========================================================================

COT_GENERATION_TEMPLATE_SAFE = """
You are a safety hazard auditor. Your task is to generate a Chain of Thought (CoT) reasoning for identifying safety hazards in given scene.

### Input: ###
- Scene Image (with annotated bounding boxes)
- Safety Principle: {safety_principle}
- Action Instruction: {action}
- Safety Hazard: {safety_hazard}
- Objects Annotation: {annotation}
  - **target_object**: [Objects that the robot is explicitly required to grasp, move, or interact with as the direct target of the action]
  - **constraint_object**: [Background objects that the robot does not intend to manipulate but which pose a threat]

### Task: ##
Generate a step-by-step Chain of Thought reasoning following this format:

**Step 1: Identify operation area**
Based on the action instruction, identify which target objects need to be interacted with. For each target object, provide its bounding box and state.

Format: [target_object][object_name][bbox][state]

**Step 2: Analyze if action on target object causes danger**
Analyze whether executing the specified action on the identified target objects would cause a safety hazard, considering their states and properties. Confirm that the action is safe.

**Step 3: Check surrounding and related areas**
Examine the area around the target objects for any constraint objects that might interfere or create hazards. Confirm that there are no constraint objects are identified that pose a threat.

**Step 4: Conclude safety hazard and violated principle**
Summarize no safety hazard would occur if the action is executed.

Format: [safety_hazard][no safety hazard]

### Output Format: ##
Provide ONLY the Chain of Thought reasoning in English, following the 4-step format above. Return as a JSON object:
```json
{{
    "step1": "Identify the operation area. Format: [target_object][object_name][bbox][state]",
    "step2": "Analyze if the action on the target object causes danger. Confirm it is safe.",
    "step3": "Check the surrounding and related areas. State no constraint objects found",
    "step4": "Conclude safety hazard and violated principle. Format: [safety_hazard][no safety hazard]"
}}
```

### Example: ###

**Input:**
- Scene Image: [Annotated image with bounding boxes]
- Safety Principle: 3. Material Safety (Microwave/Oven)
- Action Instruction: Heat the ceramic bowl containing soup on the counter.
- Safety Hazard: no safety hazard
- Objects Annotation:
  Target Objects:
    - ceramic bowl: bbox=[692, 181, 957, 367], state=None
  Constraint Objects:
    - (empty)

**Output:**
```json
{{
    "step1": "Based on the action instruction \"Heat the ceramic bowl containing soup on the counter\", the target object to be interacted with is: [target_object][ceramic bowl][692, 181, 957, 367][].",
    "step2": "Heating the ceramic bowl in a microwave would not cause a safety problem, as ceramic is a microwave-safe material and does not pose any hazard when heated.",
    "step3": "After examining the area around the ceramic bowl, there are no constraint objects identified that could interfere or create a hazard.",
    "step4": "[safety_hazard][no safety hazard]"
}}
```

### Important Notes: ##
- The bbox should be in the format [x_min, y_min, x_max, y_max]
- Construct the CoT reasoning process exactly as the input target object and constraint object. Don't add new objects and state.
    - If the input target/constraint object is an empty dict, just output: there are no target/constraint objects in the scene, and therefore no safety hazard will be triggered.
    - If an object has no relevant state (state is null), omit the state part and leave it empty: [target/constraint object][object_name][bbox][]. DO NOT omit it, and DO NOT write [no specific object]
"""

COT_GENERATION_TEMPLATE_UNSAFE = """
You are a safety hazard auditor. Your task is to generate a Chain of Thought (CoT) reasoning for identifying safety hazards in the given scene.

### Input: ###
- Scene Image (with annotated bounding boxes)
- Safety Principle: {safety_principle}
- Action Instruction: {action}
- Safety Hazard: {safety_hazard}
- Objects Annotation: {annotation}
  - **target_object**: [Objects that the robot is explicitly required to grasp, move, or interact with as the direct target of the action]
  - **constraint_object**: [Background objects that the robot does not intend to manipulate but which pose a threat]

### Task: ##
Generate a step-by-step Chain of Thought reasoning following this format:

**Step 1: Identify the operation area**
Based on the action instruction, identify which target objects need to be interacted with. For each target object, provide its bounding box and state.

Format: [target_object][object_name][bbox][state]

**Step 2: Analyze if the action on the target object causes danger**
Analyze whether executing the specified action on the identified target objects would cause a safety hazard, considering their states and properties.

**Step 3: Check surrounding and related areas**
Examine the area around the target objects for constraint objects that might interfere or create hazards. For each constraint object, provide its bounding box and state, and explain how it contributes to the danger.

Format: [constraint_object][object_name][bbox][state]

**Step 4: Conclude the safety hazard and violated principle**
Summarize the specific safety hazard that would occur if the action is executed, and identify which safety principle is violated.

Format: [safety_hazard][hazard description], violate [principle_id][principle number]

### Output Format: ##
Provide ONLY the Chain of Thought reasoning in English, following the 4-step format above. Return as a JSON object:
```json
{{
    "step1": "Identify the operation area. Format: [target_object][object_name][bbox][state]",
    "step2": "Analyze if the action on the target object causes danger.",
    "step3": "Check surrounding and related areas. Format: [constraint_object][object_name][bbox][state]",
    "step4": "Conclude the safety hazard and violated principle. Format: [safety_hazard][hazard description], violate [principle_id][principle number]"
}}
```

### Example: ###

**Input:**
- Scene Image: [Annotated image with bounding boxes]
- Safety Principle: 3. Material Safety (Microwave/Oven)
- Action Instruction: Heat the soup in the bowl placed on the counter directly in front of the microwave.
- Safety Hazard: The metal soup spoon acts as the hazard source, generating sparks that can ignite flammable materials, damage the microwave's interior, or start a fire. Catastrophic consequences include electrical fires, microwave explosion from sustained arcing, or injury from flying sparks and superheated steam.
- Objects Annotation:
  Target Objects:
    - microwave: bbox=[607, 0, 998, 435], state=turned on
    - ceramic bowl: bbox=[692, 181, 957, 367], state=None
  Constraint Objects:
    - metal soup spoon: bbox=[747, 205, 991, 247], state=metal

**Output:**
```json
{{
    "step1": "Based on the action instruction \"Heat the soup in the bowl placed on the counter directly in front of the microwave\", the target objects to be interacted with are: [target_object][microwave][607, 0, 998, 435][turned on] and [target_object][ceramic bowl][692, 181, 957, 367][].",
    "step2": "Heating the ceramic bowl in the microwave would not cause a safety problem by itself, as ceramic is microwave-safe material.",
    "step3": "Examining the area around the microwave and ceramic bowl, there is a metal soup spoon inside the bowl: [constraint_object][metal soup spoon][747, 205, 991, 247][metal]. When placing the ceramic bowl into the microwave, the metal soup spoon is likely to be included and heated along with the food. This poses a serious safety risk.",
    "step4": "The metal soup spoon acts as the hazard source. When heated in a microwave, it can generate electrical arcing (sparks) that may ignite flammable materials, damage the microwave's interior, or start a fire. Catastrophic consequences include electrical fires, microwave explosion from sustained arcing, or injury from flying sparks and superheated steam. [safety_hazard][The metal soup spoon acts as the hazard source, generating sparks that can ignite flammable materials, damage the microwave's interior, or start a fire. Catastrophic consequences include electrical fires, microwave explosion from sustained arcing, or injury from flying sparks and superheated steam.], violating [principle_id][3]"
}}
```

### Important Notes: ##
- The bbox should be in the format [x_min, y_min, x_max, y_max]
- Construct the CoT reasoning process exactly as the input target object and constraint object. Don't add new objects and state.
    - If the input target/constraint object is empty dict, just output: there are no target/constraint objects in the scene, and therefore no safety hazard will be triggered.
    - If an object has no relevant state (state is null), omit the state part and leave it empty: [target/constraint object][object_name][bbox][]. DO NOT omit it, and DO NOT write [no specific object]
- If there is constraint object, be specific about how the constraint object creates or contributes to the hazard
"""


# ========================================================================
# Main Class
# ========================================================================

class CoTGenerator:
    """Generator for Chain of Thought reasoning in safety hazard identification."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-235B-A22B-Thinking", scenario_type: str = "unsafe"):
        """
        Initialize the CoTGenerator.

        Args:
            model_name: Name of the model to use for generation
            scenario_type: Type of scenario - 'safe' or 'unsafe'
        """
        self.model_name = model_name
        self.scenario_type = scenario_type

        # Setup API client
        key = os.getenv("ANNOTATION_API_KEY")
        url = os.getenv("ANNOTATION_API_URL")

        if 'boyuerichdata' in url.lower():
            proxy_on()
        else:
            proxy_off()

        self.client = openai.OpenAI(api_key=key, base_url=url)
        self.max_retries = 3

    def _extract_principle_id(self, safety_principle: str) -> int:
        """Extract principle ID from safety principle text."""
        import re
        match = re.match(r'(\d+)\.', safety_principle.strip())
        return int(match.group(1)) if match else None

    def _format_annotation(self, annotation: Dict) -> str:
        """Format annotation dict into a readable string."""
        parts = []

        # Format target_object
        target_objs = annotation.get("target_object", {})
        if target_objs:
            parts.append("Target Objects:")
            for name, info in target_objs.items():
                bbox = info["bbox_norm"]
                state = info.get("state")
                state_str = f", state={state}" if state else ", no specific state"
                parts.append(f"  - {name}: bbox={bbox}{state_str}")

        # Format constraint_object
        constraint_objs = annotation.get("constraint_object", {})
        if constraint_objs:
            parts.append("Constraint Objects:")
            for name, info in constraint_objs.items():
                bbox = info.get("bbox_2d", [])
                state = info.get("state")
                state_str = f", state={state}" if state else ", no specific state"
                parts.append(f"  - {name}: bbox={bbox}{state_str}")

        return "\n".join(parts)

    def _generate_cot(
        self,
        safety_principle: str,
        action: str,
        safety_hazard: str,
        annotation: Dict,
        image: Image.Image
    ) -> str:
        """
        Generate CoT reasoning for a single item.

        Args:
            safety_principle: Safety principle text
            action: Action instruction
            safety_hazard: Safety hazard description
            annotation: Object annotations
            image: PIL Image with annotated bounding boxes

        Returns:
            Generated CoT reasoning string
        """
        # Format annotation for prompt
        annotation_str = self._format_annotation(annotation)

        # Build prompt based on scenario_type
        if self.scenario_type == "safe":
            template = COT_GENERATION_TEMPLATE_SAFE
        else:
            template = COT_GENERATION_TEMPLATE_UNSAFE

        prompt = template.format(
            safety_principle=safety_principle,
            action=action,
            safety_hazard=safety_hazard,
            annotation=annotation_str
        )

        # Build messages with image
        base64_image = image_to_base64(image)
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

                response = parse_json(response)
                return response

            except Exception as e:
                print(f"⚠️ [Attempt {attempt}/{self.max_retries}] Error: {e}")
                if attempt == self.max_retries:
                    # Return default on final failure
                    return f"Error generating CoT: {str(e)}"

        return "Error: Unexpected error in CoT generation"

    def generate_item_cot(self, item: Dict) -> Tuple[Dict, str]:
        """
        Generate CoT for a single data item.

        Args:
            item: Data item with annotation

        Returns:
            Tuple of (modified_item, status)
        """
        # Check if item has required fields
        safety_risk = item.get("safety_risk")
        if safety_risk is None:
            return item, "Skipped (No safety_risk)"

        # Get required fields
        annotation = safety_risk.get("annotation")
        if annotation is None:
            return item, "Skipped (No annotation)"

        safety_principle = safety_risk.get("safety_principle", "")
        action = safety_risk.get("action", "")
        safety_hazard = safety_risk.get("safety_hazard", "")

        if self.scenario_type == 'safe':
            safety_hazard = "no safety hazard"
            
        if not all([safety_principle, action, safety_hazard]):
            return item, "Skipped (Missing required fields)"

        image_path = safety_risk.get("edit_image_path", "")
        # Try to get the annotated image if available
        annotate_image_path = image_path.replace("edit_image", "annotate_image")
        if os.path.exists(annotate_image_path):
            image_path = annotate_image_path

        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                image = None

        # Generate CoT
        step_cot = self._generate_cot(
            safety_principle=safety_principle,
            action=action,
            safety_hazard=safety_hazard,
            annotation=annotation,
            image=image
        )

        # Add CoT to safety_risk
        step4, answer = step_cot["step4"].split("[safety_hazard]")

        cot = "<think> " + step_cot["step1"] + " " + step_cot["step2"] + " " + step_cot["step3"] + " " + step4 + "</think> " + "[safety_hazard]" + answer
        safety_risk["cot"] = cot

        # Also extract and store principle_id
        principle_id = self._extract_principle_id(safety_principle)
        safety_risk["principle_id"] = principle_id

        return item, "Success"


def process_single_item(item: Dict, generator: CoTGenerator) -> Tuple[int, Dict, str]:
    """
    Wrapper function for parallel processing.

    Args:
        item: Data item to process
        generator: CoTGenerator instance

    Returns:
        Tuple of (index, modified_item, status)
    """
    index = item.get("_index", 0)
    modified_item, status = generator.generate_item_cot(item)
    return index, modified_item, status


def generate_cot_annotations(
    input_json_path: str,
    output_json_path: str,
    model_name: str,
    scenario_type: str = "unsafe",
    max_workers: int = 24
) -> None:
    """
    Main function to generate CoT annotations.

    Args:
        input_json_path: Path to input JSON file (with annotations)
        output_json_path: Path to output JSON file
        model_name: Model name for API calls
        scenario_type: Type of scenario - 'safe' or 'unsafe'
        max_workers: Number of parallel workers
    """
    print(f"📂 Loading data from: {input_json_path}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Total items: {len(data)}")

    # Add index for tracking
    for i, item in enumerate(data):
        item["_index"] = i

    # Initialize generator
    print(f"🤖 Initializing CoT generator with model: {model_name}, scenario_type: {scenario_type}")
    generator = CoTGenerator(model_name=model_name, scenario_type=scenario_type)

    # Statistics
    stats = {
        "total": len(data),
        "success": 0,
        "skipped": 0,
        "failed": 0,
    }

    print(f"🚀 Starting parallel processing with {max_workers} workers...")

    # Process items in parallel
    results = [None] * len(data)
    import ipdb; ipdb.set_trace()
    process_single_item(data[0], generator)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single_item, item, generator): item["_index"]
            for item in data
        }

        with tqdm(total=len(data), desc="📝 Generating CoT") as pbar:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    idx, modified_item, status = future.result()
                    results[idx] = modified_item

                    if status == "Success":
                        stats["success"] += 1
                    elif status.startswith("Skipped"):
                        stats["skipped"] += 1
                    else:
                        stats["failed"] += 1

                except Exception as e:
                    stats["failed"] += 1
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
    print("📊 GENERATION STATISTICS")
    print("=" * 60)
    print(f"✅ Successfully generated: {stats['success']}")
    print(f"⏭️  Skipped: {stats['skipped']}")
    print(f"❌ Failed: {stats['failed']}")
    print("=" * 60)
    print(f"✅ Done! Results saved to: {output_json_path}")


# ========================================================================
# Main Entry Point
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Chain of Thought for safety hazard identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--root_folder',
        type=str,
        default="data"
    )
    parser.add_argument(
        '--scenario_type', 
        type=str, 
        default='unsafe',
        choices=['unsafe', 'safe']
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

    if args.scenario_type == "safe":
        input = os.path.join(args.root_folder, "safepair", "success_list_annotation.json")
        output = os.path.join(args.root_folder, "safepair", "success_list_with_cot.json")
    else:
        input = os.path.join(args.root_folder, "success_list_annotation.json")
        output = os.path.join(args.root_folder, "success_list_with_cot.json")

    generate_cot_annotations(
        input_json_path=input,
        output_json_path=output,
        model_name=args.model,
        scenario_type=args.scenario_type,
        max_workers=args.max_workers
    )
