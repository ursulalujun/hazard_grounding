"""
Image Rewrite Script for PaSBench.

This script rewrites images from PaSBench dataset, condensing image sequence content
into single images that express the same safety risk with different appearances.

Usage:
    python -m data_pipeline.supplement.rewrite_pasbench_image
"""

import argparse
import base64
import json
import os
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm

from data_pipeline.utils import proxy_on


# Template for image rewriting - condense sequence content, same safety risk, different appearance
IMAGE_REWRITE_PROMPT = """You are an expert image editor specializing in scene variation while preserving safety context.

### Context Information ###
- Instruction: {instruction}
- Risk Reason: {risk_reason}

### Requirements ###
1. **Same safety risk**: The rewritten image must express the SAME safety risk as described in the risk_reason
2. **Condense sequence content**: The original may represent an action sequence - condense this into a single compelling image that captures the key safety moment
3. **Different item styles**: Change the styles/appearance of objects (furniture, appliances, utensils, etc.) - different colors, designs, brands
4. **Different environment layout**: Significantly alter the room layout, furniture arrangement, and spatial organization
5. **Different person appearance**: If there's a person, change their pose, clothing, hair, and overall appearance completely
6. **Different camera perspective**: Change the viewing angle - you can use a different camera perspective (e.g., from different height, angle, or distance)
7. **Expanded environment**: Extend and expand the surrounding environment - show more of the room, add background elements, create a fuller, more complete scene
8. **Realistic style**: The output must be photorealistic, NOT manga, anime, cartoon, simulator, or game style

Edit the entire image to create a new variant that illustrates the same safety risk in a completely different visual style, with changed perspective and expanded surroundings."""


class PaSBenchImageRewriter:
    """Handler for image rewriting operations on PaSBench using API."""

    def __init__(self, model_name: str = "gpt-image-1-mini"):
        """
        Initialize the image rewriting pipeline with API client.

        Args:
            model_name: Name of the model to use (default: gpt-image-1-mini)
        """
        self.model_name = model_name
        
        # Setup API client (only once)
        proxy_on()
        key = os.getenv("EDIT_API_KEY")
        url = os.getenv("EDIT_API_URL")
        
        self.client = openai.OpenAI(api_key=key, base_url=url)
        print(f"API client initialized for model: {model_name}")

    def rewrite_single_image(self, image_path: str, instruction: str, risk_reason: str, 
                            output_path: str) -> dict:
        """
        Rewrite a single image (one variant).

        Args:
            image_path: Path to the input image
            instruction: The instruction text
            risk_reason: The risk reason text
            output_path: Path to save the output image

        Returns:
            Dict with status and output_path
        """
        # Skip if already exists
        if os.path.exists(output_path):
            return {
                'status': 'skipped',
                'output_path': output_path,
                'message': 'Already exists'
            }

        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'output_path': output_path,
                'message': f'Image not found: {image_path}'
            }

        # Format prompt with context information
        prompt = IMAGE_REWRITE_PROMPT.format(
            instruction=instruction,
            risk_reason=risk_reason
        )

        try:
            # Call GPT-image-1-mini API
            result = self.client.images.edit(
                model=self.model_name,
                image=open(image_path, "rb"),
                prompt=prompt
            )
            
            # Get base64 encoded image from response
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Save the output image
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            return {
                'status': 'success',
                'output_path': output_path,
                'message': 'Generated successfully'
            }

        except Exception as e:
            return {
                'status': 'error',
                'output_path': output_path,
                'message': str(e)
            }

    def process_item_variants(self, item: dict, output_dir: str,
                             num_variants: int, max_workers: int = 10) -> dict:
        """
        Process all variants for a single item.

        Args:
            item: Data item with image_path, instruction, risk_reason
            output_dir: Directory to save output images
            num_variants: Number of variants to generate
            max_workers: Number of parallel workers for API calls

        Returns:
            Dict with processing results for this item
        """
        image_path = item.get('image_path', '')
        instruction = item.get('instruction', '')
        risk_reason = item.get('risk_reason', '')
        item_id = item.get('id', 'unknown')

        # Skip if no image path or image doesn't exist
        if not image_path or not os.path.exists(image_path):
            return {
                'id': item_id,
                'status': 'error',
                'message': 'Image file not found',
                'image_path': image_path,
                'outputs': []
            }

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create all tasks for this item
        tasks = []
        for i in range(num_variants):
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_dir, output_filename)

            tasks.append({
                'image_path': image_path,
                'instruction': instruction,
                'risk_reason': risk_reason,
                'output_path': output_path,
                'variant_index': i
            })

        # Process variants in parallel
        outputs = []
        success_count = 0
        skipped_count = 0
        error_count = 0

        def process_single_task(task):
            return self.rewrite_single_image(
                task['image_path'],
                task['instruction'],
                task['risk_reason'],
                task['output_path']
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_task, task): task for task in tasks}

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    outputs.append(result)

                    if result['status'] == 'success':
                        success_count += 1
                        print(f"  [{item_id}] Variant {task['variant_index']+1}: Saved")
                    elif result['status'] == 'skipped':
                        skipped_count += 1
                        print(f"  [{item_id}] Variant {task['variant_index']+1}: Skipped (already exists)")
                    else:
                        error_count += 1
                        print(f"  [{item_id}] Variant {task['variant_index']+1}: Error - {result['message']}")
                except Exception as e:
                    error_count += 1
                    print(f"  [{item_id}] Variant {task['variant_index']+1}: Exception - {str(e)}")

        return {
            'id': item_id,
            'status': 'completed',
            'input': image_path,
            'outputs': [r['output_path'] for r in outputs if r['status'] in ['success', 'skipped']],
            'count': success_count + skipped_count,
            'success_count': success_count,
            'skipped_count': skipped_count,
            'error_count': error_count
        }

    def process_json_data(self, json_path: str, output_dir: str, num_variants: int = 5,
                         max_items: int = None, max_workers: int = 10) -> dict:
        """
        Process all items in the PaSBench contextual JSON with parallel API calls.

        Args:
            json_path: Path to the pasbench_contextual.json file
            output_dir: Directory to save output images
            num_variants: Number of variants per image
            max_items: Maximum number of items to process (None for all)
            max_workers: Number of parallel workers for API calls

        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[4:]
        
        if max_items is not None:
            data = data[:max_items]
        
        print(f"Found {len(data)} items to process")
        print(f"Generating {num_variants} variants per image")
        print(f"Total API calls: {len(data) * num_variants}")
        print(f"Using {max_workers} parallel workers")

        results = {
            'json_path': json_path,
            'output_dir': output_dir,
            'num_variants_per_image': num_variants,
            'processed': [],
            'failed': [],
            'total_input_images': 0,
            'total_output_images': 0,
            'total_success': 0,
            'total_skipped': 0,
            'total_errors': 0
        }

        # Process each item with parallel variants
        for item in tqdm(data, desc="Processing items"):
            item_id = item.get('id', 'unknown')
            results['total_input_images'] += 1
            
            try:
                item_result = self.process_item_variants(item, output_dir, num_variants, max_workers)
                
                if item_result['status'] == 'completed':
                    results['processed'].append(item_result)
                    results['total_output_images'] += item_result['count']
                    results['total_success'] += item_result['success_count']
                    results['total_skipped'] += item_result['skipped_count']
                    results['total_errors'] += item_result['error_count']
                else:
                    results['failed'].append(item_result)

            except Exception as e:
                print(f"\n[ERROR] Failed to process item {item_id}: {e}")
                results['failed'].append({
                    'id': item_id,
                    'image_path': item.get('image_path', ''),
                    'error': str(e)
                })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Image rewrite for PaSBench using GPT-image-1-mini API with parallel calls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--json_path',
        type=str,
        default='data_pipeline/supplement/pasbench_contextual.json',
        help='Path to the pasbench_contextual.json file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data_pipeline/supplement/pasbench_rewritten_images',
        help='Output directory for rewritten images'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-image-1-mini',
        help='Model name for image editing API'
    )
    parser.add_argument(
        '--num_variants',
        type=int,
        default=1,
        help='Number of style variants to generate per input image'
    )
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help='Maximum number of items to process (None for all)'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=24,
        help='Number of parallel API workers'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PaSBench Image Rewrite (using API with parallel calls)")
    print("=" * 60)
    print(f"JSON input: {args.json_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Variants per image: {args.num_variants}")
    print(f"Parallel workers: {args.max_workers}")
    if args.max_items:
        print(f"Max items to process: {args.max_items}")
    print("=" * 60)

    # Initialize the image rewriting pipeline
    rewriter = PaSBenchImageRewriter(model_name=args.model)

    # Process all items from JSON
    results = rewriter.process_json_data(
        json_path=args.json_path,
        output_dir=args.output_dir,
        num_variants=args.num_variants,
        max_items=args.max_items,
        max_workers=args.max_workers
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total input images: {results['total_input_images']}")
    print(f"Successfully processed items: {len(results['processed'])}")
    print(f"Failed items: {len(results['failed'])}")
    print(f"Total output images: {results['total_output_images']}")
    print(f"  - Newly generated: {results['total_success']}")
    print(f"  - Skipped (existing): {results['total_skipped']}")
    print(f"  - Errors: {results['total_errors']}")
    print(f"Output directory: {results['output_dir']}")
    print("=" * 60)

    # Save results summary
    summary_path = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
