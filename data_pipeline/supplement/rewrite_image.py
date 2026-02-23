"""
Image Style Transfer Script for MSSBench.

This script performs style transfer on images from MSSBench embodied dataset,
converting simulator-style images to realistic home scene style while
preserving object types and layout.

Usage:
    python -m data_pipeline.supplement.rewrite_image
"""

import argparse
import base64
import os
from io import BytesIO
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import QwenImageEditPlusPipeline


# Template for style transfer - converting simulator style to realistic home style
STYLE_TRANSFER_PROMPT = """You are an expert image editor specializing in style transfer. Your task is to transform the given image from a simulator/game style to a realistic home scene style.

### Requirements ###
1. **Preserve all objects**: Keep the exact same types of objects in the scene (furniture, appliances, utensils, etc.)
2. **Preserve layout**: Maintain the same spatial arrangement and positioning of all objects
3. **Style transfer only**: Change ONLY the visual style from simulator/game graphics to photorealistic home photography
4. **Realistic appearance**: The output should look like a real photograph taken in a home environment
5. **Natural lighting**: Use natural lighting conditions found in real homes
6. **Realistic materials**: Replace synthetic/textured materials with realistic ones (wood, metal, fabric, etc.)

### What NOT to change ###
- Do NOT add or remove any objects
- Do NOT change the position or arrangement of objects
- Do NOT alter the scene composition
- Do NOT add text, labels, or UI elements

### Style characteristics to achieve ###
- Photorealistic textures and materials
- Natural lighting and shadows
- Realistic depth of field
- Authentic home environment atmosphere
- High-quality photography aesthetic

Edit the entire image to achieve this style transformation."""


class ImageStyleTransfer:
    """Handler for style transfer operations on images."""

    def __init__(self, model_path: str = "checkpoints/Qwen-Image-Edit-2511"):
        """
        Initialize the style transfer pipeline.

        Args:
            model_path: Path to the Qwen Image Edit model
        """
        print(f"Loading model from: {model_path}")
        self.pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path)
        self.pipeline.to(torch.bfloat16)
        self.pipeline.to("cuda")
        self.pipeline.set_progress_bar_config(disable=None)
        print("Model loaded successfully!")

    def transfer_style(self, image_path: str, output_folder: str, num_images: int = 3) -> list:
        """
        Transfer style of a single image, generating multiple variants.

        Args:
            image_path: Path to the input image
            output_folder: Folder to save the output images
            num_images: Number of output images to generate (default: 5)

        Returns:
            List of paths to the generated images
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            return []

        if 'unsafe' not in image_path:
            print(f"Skipping safe image")
            return []

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[1]

        # Create output folder
        os.makedirs(output_folder, exist_ok=True)

        output_paths = []
        prompt = STYLE_TRANSFER_PROMPT

        # Load and process image
        image = Image.open(image_path).convert("RGB")

        print(f"Processing {base_name}{ext} -> generating {num_images} variants...")

        for i in range(num_images):
            output_filename = f"{base_name}.png" # f"{base_name}_variant_{i+1}.png"
            output_path = os.path.join(output_folder, output_filename)

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"  [Skip] {output_filename} already exists")
                output_paths.append(output_path)
                continue

            try:
                # Set different seed for each variant
                seed = 42 + i*50

                # Prepare inputs
                inputs = {
                    "image": image,
                    "prompt": prompt,
                    "generator": torch.manual_seed(seed),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": "simulator, game, cartoon, anime, low quality, blurry, distorted",
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

    def process_directory(self, input_dir: str, output_dir: str, num_images: int = 3, 
                        max_images: int = None) -> dict:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            num_images: Number of variants per image
            max_images: Maximum number of input images to process (None for all)

        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for f in os.listdir(input_dir):
            if os.path.splitext(f)[1].lower() in image_extensions:
                image_files.append(os.path.join(input_dir, f))

        image_files.sort()

        if max_images is not None:
            image_files = image_files[:max_images]

        print(f"Found {len(image_files)} images to process")
        print(f"Generating {num_images} variants per image")
        print(f"Total output images: {len(image_files) * num_images}")

        results = {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'num_variants_per_image': num_images,
            'processed': [],
            'failed': [],
            'total_input_images': len(image_files),
            'total_output_images': 0
        }

        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            base_name = os.path.basename(image_path)
            
            try:
                output_paths = self.transfer_style(image_path, output_dir, num_images)
                
                if output_paths:
                    results['processed'].append({
                        'input': image_path,
                        'outputs': output_paths,
                        'count': len(output_paths)
                    })
                    results['total_output_images'] += len(output_paths)
                else:
                    results['failed'].append({
                        'input': image_path,
                        'error': 'No outputs generated'
                    })

            except Exception as e:
                print(f"\n[ERROR] Failed to process {base_name}: {e}")
                results['failed'].append({
                    'input': image_path,
                    'error': str(e)
                })

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Style transfer for MSSBench images - simulator to realistic home style",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default='third_party/data/safe_agent_bench/processed/images', # 'third_party/data/mssbench/MSSBench/mssbench/embodied',
        help='Input directory containing simulator-style images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data_pipeline/supplement/sabench_rewritten_images',
        help='Output directory for style-transferred images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='checkpoints/Qwen-Image-Edit-2511',
        help='Path to the Qwen Image Edit model'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=1,
        help='Number of style variants to generate per input image'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=None,
        help='Maximum number of input images to process (None for all)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MSSBench Image Style Transfer")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_path}")
    print(f"Variants per image: {args.num_images}")
    if args.max_images:
        print(f"Max input images: {args.max_images}")
    print("=" * 60)

    # Initialize the style transfer pipeline
    transfer = ImageStyleTransfer(model_path=args.model_path)

    # Process all images
    results = transfer.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        max_images=args.max_images
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
        import json
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
