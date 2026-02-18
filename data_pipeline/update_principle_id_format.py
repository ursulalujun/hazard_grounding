#!/usr/bin/env python3
"""
Script to update [principle_id][X] format to [principle_id][X. Title: Description]
in the cot field of success_list_with_cot2.json
"""

import json
import re
import os
from pathlib import Path


# Import ACTION_TRIGGERED_PRINCIPLES from principle_tracker.py
# We'll read it directly from the file to avoid import issues
PRINCIPLE_TRACKER_PATH = Path(__file__).parent / "nodes/principle_tracker.py"

INPUT_JSON_PATH = Path(__file__).parent / "data_sun/unsafe_scenario_info.json"
OUTPUT_JSON_PATH = Path(__file__).parent / "data_sun/unsafe_scenario_info.json"


def load_principles():
    """Load ACTION_TRIGGERED_PRINCIPLES from principle_tracker.py"""
    principles = {}

    with open(PRINCIPLE_TRACKER_PATH, 'r') as f:
        content = f.read()

        # Use regex to extract the ACTION_TRIGGERED_PRINCIPLES dictionary
        # Match patterns like: 1: {\n        "title": "...",\n        "description": "..."
        pattern = r'(\d+):\s*\{\s*"title":\s*"([^"]+)",\s*"description":\s*"([^"]+)"'
        for match in re.finditer(pattern, content):
            pid = int(match.group(1))
            title = match.group(2)
            description = match.group(3)
            principles[pid] = {
                "title": title,
                "description": description
            }

    return principles


def format_principle_ref(pid: int, principles: dict) -> str:
    """Format principle reference as [principle_id][X. Title: Description]"""
    if pid not in principles:
        return f"[principle_id][{pid}]"

    principle = principles[pid]
    return f"[safety_principle][{pid}. {principle['title']}: {principle['description']}]"


def update_cot_text(cot_text: str, principles: dict) -> str:
    """Replace [principle_id][X] with [principle_id][X. Title: Description]"""
    # Pattern to match [principle_id][(int)]
    pattern = r'\[principle_id\]\[(\d+)\]'

    def replace_func(match):
        pid = int(match.group(1))
        return format_principle_ref(pid, principles)

    return re.sub(pattern, replace_func, cot_text)


def process_json_file(input_path: Path, output_path: Path, principles: dict) -> dict:
    """Process the JSON file and update cot fields"""
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Total records: {len(data)}")

    updated_count = 0
    for i, item in enumerate(data):
        if 'safety_risk' in item and 'cot' in item['safety_risk']:
            original_cot = item['safety_risk']['cot']
            updated_cot = update_cot_text(original_cot, principles)

            if original_cot != updated_cot:
                item['safety_risk']['cot'] = updated_cot
                updated_count += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} records...")

    print(f"Updated {updated_count} records")

    # Write to output file
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done!")
    return data


def main():
    """Main function"""
    # Check if input file exists
    if not INPUT_JSON_PATH.exists():
        print(f"Error: Input file not found: {INPUT_JSON_PATH}")
        return

    # Load principles
    print("Loading safety principles...")
    principles = load_principles()
    print(f"Loaded {len(principles)} principles")

    # Process JSON file
    process_json_file(INPUT_JSON_PATH, OUTPUT_JSON_PATH, principles)


if __name__ == "__main__":
    main()
