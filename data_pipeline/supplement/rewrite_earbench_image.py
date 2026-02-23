"""
Image Rewrite Script for EARBench.

This script rewrites images from EARBench dataset, creating variants that
express the same safety content but with different styles, layouts, and appearances.

Usage:
    python -m data_pipeline.supplement.rewrite_earbench_image
"""

import argparse
import json
import os
from io import BytesIO
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import QwenImageEditPlusPipeline


# Template for image rewriting - same safety content, different appearance
IMAGE_REWRITE_PROMPT = """You are an expert image editor specializing in scene variation while preserving safety context.

### Context Information ###
- Instruction: {instruction}
- Safety Tip: {safety_tip}
- Explanation: {explanation}

### Requirements ###
1. **Same safety content**: The rewritten image must express the SAME safety tip and explanation as the original
2. **Different item styles**: Change the styles/appearance of objects (furniture, appliances, utensils, etc.) - different colors, designs, brands
3. **Different environment layout**: Significantly alter the room layout, furniture arrangement, and spatial organization
4. **Different person appearance**: If there's a person, change their pose, clothing, hair, and overall appearance completely
5. **Different camera perspective**: Change the viewing angle - you can use a different camera perspective (e.g., from different height, angle, or distance)
6. **Expanded environment**: Extend and expand the surrounding environment - show more of the room, add background elements, create a fuller, more complete scene
7. **Realistic style**: The output must be photorealistic, NOT manga, anime, cartoon, simulator, or game style

### What to KEEP the same ###
- The core safety hazard/tip being illustrated
- The type of room/scene (kitchen, bathroom, etc.)
- The general category of objects (e.g., if there's a stove, keep a stove, but change its style)
- The safety concept being demonstrated

### What to CHANGE significantly ###
- Object styles, colors, and designs
- **Room layout and furniture arrangement**
- **Camera perspective and viewing angle** (e.g., bird's eye view, eye-level, low angle)
- Person's pose, clothing, hair, and appearance (if applicable)
- Background elements and decorations
- Lighting direction and atmosphere
- **Surrounding environment extent and composition**

### Style characteristics to achieve ###
- Photorealistic quality like real photography
- Natural lighting conditions
- Authentic home environment
- Real-world materials and textures
- High-quality photography aesthetic
- Expanded, immersive scene composition

Edit the entire image to create a new variant that illustrates the same safety concept in a completely different visual style, with changed perspective and expanded surroundings."""


class EARbenchImageRewriter:
    """Handler for image rewriting operations on EARBench."""

    def __init__(self, model_path: str = "checkpoints/Qwen-Image-Edit-2511"):
        """
        Initialize the image rewriting pipeline.

        Args:
            model_path: Path to the Qwen Image Edit model
        """
        print(f"Loading model from: {model_path}")
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path)
        self.pipeline.to(torch.bfloat16)
        self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=None)
        print("Model loaded successfully!")

    def rewrite_image(self, image_path: str, instruction: str, safety_tip: str, 
                     explanation: str, output_folder: str, num_variants: int = 5) -> list:
        """
        Rewrite a single image, generating multiple variants.

        Args:
            image_path: Path to the input image
            instruction: The instruction text
            safety_tip: The safety tip text
            explanation: The explanation text
            output_folder: Folder to save the output images
            num_variants: Number of output images to generate (default: 5)

        Returns:
            List of paths to the generated images
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        output_paths = []
        
        # Format prompt with context information
        prompt = IMAGE_REWRITE_PROMPT.format(
            instruction=instruction,
            safety_tip=safety_tip,
            explanation=explanation
        )

        # Load and process image
        image = Image.open(image_path).convert("RGB")

        print(f"Processing {base_name} -> generating {num_variants} variants...")

        for i in range(num_variants):
            output_filename = f"{base_name}.png"
            output_path = os.path.join(output_folder, output_filename)

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"  [Skip] {output_filename} already exists")
                output_paths.append(output_path)
                continue

            try:
                # Set different seed for each variant
                seed = 42 + i * 100

                # Prepare inputs
                inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": torch.manual_seed(seed),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": "cartoon, anime, manga, simulator, game, low quality, blurry, distorted, unrealistic, artificial",
                    "num_inference_steps": 50,
                    "num_images_per_prompt": 1  # Generate one at a time with different seeds
                }

                # Run the pipeline
                with torch.inference_mode():
                    output = self.pipeline(**inputs)
                    output_image = output.images[0]

                # Save the output image
                output_image.save(output_path)
                print(f"  [Saved] {output_filename}")
                output_paths.append(output_path)

            except Exception as e:
                print(f"  [Error] Failed to generate variant {i+1}: {e}")
                continue

        return output_paths

    def process_json_data(self, json_path: str, output_dir: str, num_variants: int = 5, start = 0, end = -1) -> dict:
        """
        Process all items in the EARBench contextual JSON.

        Args:
            json_path: Path to the earbench_contextual.json file
            output_dir: Directory to save output images
            num_variants: Number of variants per image
            max_items: Maximum number of items to process (None for all)

        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if end != -1:
            data = data[start:end]
        else:
            data = data[start:]   
        print(f"Found {len(data)} items to process")
        print(f"Generating {num_variants} variants per image")
        print(f"Total output images: {len(data) * num_variants}")

        results = {
            'json_path': json_path,
            'output_dir': output_dir,
            'num_variants_per_image': num_variants,
            'processed': [],
            'failed': [],
            'total_input_images': 0,
            'total_output_images': 0
        }

        # Process each item
        for item in tqdm(data, desc="Processing images"):
            image_path = item.get('image_path', '')
            instruction = item.get('instruction', '')
            safety_tip = item.get('safety_tip', '')
            explanation = item.get('explanation', '')
            item_id = item.get('id', 'unknown')
            
            # Skip if no image path
            if not image_path or not os.path.exists(image_path):
                print(f"\n[WARNING] Image not found for item {item_id}: {image_path}")
                results['failed'].append({
                    'id': item_id,
                    'image_path': image_path,
                    'error': 'Image file not found'
                })
                continue
            
            results['total_input_images'] += 1
            
            try:
                # Create scene-specific output folder
                scene = item.get('scene', 'unknown')
                scene_output_dir = os.path.join(output_dir, scene)
                
                output_paths = self.rewrite_image(
                    image_path, instruction, safety_tip, explanation,
                    scene_output_dir, num_variants
                )
                
                if output_paths:
                    results['processed'].append({
                        'id': item_id,
                        'scene': scene,
                        'input': image_path,
                        'outputs': output_paths,
                        'count': len(output_paths)
                    })
                    results['total_output_images'] += len(output_paths)
                else:
                    results['failed'].append({
                        'id': item_id,
                        'image_path': image_path,
                        'error': 'No outputs generated'
                    })

            except Exception as e:
                print(f"\n[ERROR] Failed to process item {item_id}: {e}")
                results['failed'].append({
                    'id': item_id,
                    'image_path': image_path,
                    'error': str(e)
                })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Image rewrite for EARBench - same safety content, different appearance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--json_path',
        type=str,
        default='data_pipeline/supplement/earbench_contextual.json',
        help='Path to the earbench_contextual.json file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data_pipeline/supplement/earbench_rewritten_images',
        help='Output directory for rewritten images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/Qwen-Image-Edit-2511',
        help='Path to the Qwen Image Edit model'
    )
    parser.add_argument(
        '--num_variants',
        type=int,
        default=1,
        help='Number of style variants to generate per input image'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Maximum number of items to process (None for all)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=-1,
        help='Maximum number of items to process (None for all)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EARBench Image Rewrite")
    print("=" * 60)
    print(f"JSON input: {args.json_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Variants per image: {args.num_variants}")
    print(f"Process image from {args.start} to {args.end}")
    print("=" * 60)

    # Initialize the image rewriting pipeline
    rewriter = EARbenchImageRewriter(model_path=args.model_path)

    # Process all items from JSON
    results = rewriter.process_json_data(
        json_path=args.json_path,
        output_dir=args.output_dir,
        num_variants=args.num_variants,
        start=args.start,
        end=args.end
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total input images: {results['total_input_images']}")
    print(f"Successfully processed: {len(results['processed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Total output images generated: {results['total_output_images']}")
    print(f"Output directory: {results['output_dir']}")
    print("=" * 60)

    # Save results summary
    summary_path = os.path.join(args.output_dir, 'processing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
