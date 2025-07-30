import rawpy
import cv2
import numpy as np
import scipy
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import os
import glob


def process_raw_file(filepath):
    """Process a single RAW file and return hot/dead pixel masks"""
    with rawpy.imread(filepath) as raw:
        img = raw.raw_image.copy().astype(np.float32)

    # Normalize image
    norm_img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Define thresholds
    hot_threshold = 0.98  # pixels brighter than this (near-white)
    dead_threshold = 0.02  # pixels darker than this (near-black)

    # Apply median filter for better detection
    median_filtered = median_filter(norm_img, size=3)

    # Create masks for hot and dead pixels
    hot_mask = (norm_img > hot_threshold) & ((norm_img - median_filtered) > 0.2)
    dead_mask = (norm_img < dead_threshold) & ((median_filtered - norm_img) > 0.2)

    return hot_mask, dead_mask, norm_img


def find_consistent_defective_pixels(raw_files):
    """Find pixels that are consistently defective across all images"""
    if not raw_files:
        raise ValueError("No RAW files provided")

    print(f"Processing {len(raw_files)} RAW files...")

    # Process first file to get dimensions and initialize masks
    hot_mask_combined, dead_mask_combined, first_img = process_raw_file(raw_files[0])
    print(f"Processed: {os.path.basename(raw_files[0])}")

    # Process remaining files and combine masks using AND operation
    for i, filepath in enumerate(raw_files[1:], 1):
        hot_mask, dead_mask, _ = process_raw_file(filepath)

        # Only keep pixels that are defective in ALL images
        hot_mask_combined = hot_mask_combined & hot_mask
        dead_mask_combined = dead_mask_combined & dead_mask

        print(f"Processed: {os.path.basename(filepath)} ({i + 1}/{len(raw_files)})")

    return hot_mask_combined, dead_mask_combined, first_img


# Main execution
if __name__ == "__main__":
    # Method 1: Specify files manually
    raw_files = [
        '_DSC3624.ARW',
        '_DSC3625.ARW',
        '_DSC3626.ARW',
        '_DSC3627.ARW',
        # Add more files as needed
    ]

    # Method 2: Automatically find all ARW files in current directory
    # raw_files = glob.glob('*.ARW')

    # Method 3: Find all ARW files in a specific directory
    # raw_files = glob.glob('/path/to/your/raw/files/*.ARW')

    # Filter out files that don't exist
    existing_files = [f for f in raw_files if os.path.exists(f)]

    if not existing_files:
        print("No RAW files found! Please check your file paths.")
        exit(1)

    print(f"Found {len(existing_files)} RAW files")

    try:
        # Find consistently defective pixels
        consistent_hot_mask, consistent_dead_mask, reference_img = find_consistent_defective_pixels(existing_files)

        # Get pixel coordinates
        consistent_hot_pixels = np.where(consistent_hot_mask)
        consistent_dead_pixels = np.where(consistent_dead_mask)

        # Print results
        print(f"\n=== RESULTS ===")
        print(f"Consistent hot pixels (present in ALL {len(existing_files)} images): {np.sum(consistent_hot_mask)}")
        print(f"Consistent dead pixels (present in ALL {len(existing_files)} images): {np.sum(consistent_dead_mask)}")
        print(f"Total consistent defective pixels: {np.sum(consistent_hot_mask) + np.sum(consistent_dead_mask)}")

        # Save pixel coordinates to file for future reference
        if np.sum(consistent_hot_mask) > 0 or np.sum(consistent_dead_mask) > 0:
            defective_pixels = {
                'hot_pixels': {
                    'y_coords': consistent_hot_pixels[0].tolist(),
                    'x_coords': consistent_hot_pixels[1].tolist(),
                    'count': int(np.sum(consistent_hot_mask))
                },
                'dead_pixels': {
                    'y_coords': consistent_dead_pixels[0].tolist(),
                    'x_coords': consistent_dead_pixels[1].tolist(),
                    'count': int(np.sum(consistent_dead_mask))
                },
                'processed_files': existing_files
            }

            import json

            with open('defective_pixels_map.json', 'w') as f:
                json.dump(defective_pixels, f, indent=2)
            print(f"Defective pixel map saved to 'defective_pixels_map.json'")

        # Visualize results
        plt.figure(figsize=(12, 10))
        plt.imshow(reference_img, cmap='gray')

        if len(consistent_hot_pixels[0]) > 0:
            plt.scatter(consistent_hot_pixels[1], consistent_hot_pixels[0],
                        color='red', s=3, label=f'Consistent Hot Pixels ({np.sum(consistent_hot_mask)})', alpha=0.8)

        if len(consistent_dead_pixels[0]) > 0:
            plt.scatter(consistent_dead_pixels[1], consistent_dead_pixels[0],
                        color='blue', s=3, label=f'Consistent Dead Pixels ({np.sum(consistent_dead_mask)})', alpha=0.8)

        plt.legend()
        plt.title(f"Consistently Defective Pixels Across {len(existing_files)} RAW Images")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Save the plot
        plt.savefig('consistent_defective_pixels.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'consistent_defective_pixels.png'")

        plt.show()

    except Exception as e:
        print(f"Error processing files: {e}")
