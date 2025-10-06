#!/usr/bin/env python3
"""
Image Directory Difference Comparison Script

This script compares two directories with the same structure containing images
and saves the differences to a new output directory with the same structure.

Usage:
    python image_diff_comparison.py --dir1 path/to/first/directory --dir2 path/to/second/directory --output path/to/output/directory

Features:
- Supports common image formats (jpg, jpeg, png, bmp, tiff, webp)
- Maintains the same directory structure in the output
- Computes pixel-wise absolute differences
- Saves difference images with optional normalization
- Provides detailed statistics and progress reporting
- Handles missing files gracefully
"""

import os
import argparse
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

class ImageDifferenceComparator:
    def __init__(self, dir1, dir2, output_dir, normalize=True, threshold=0, save_stats=True):
        """
        Initialize the image difference comparator.
        
        Args:
            dir1 (str): Path to first directory
            dir2 (str): Path to second directory  
            output_dir (str): Path to output directory for difference images
            normalize (bool): Whether to normalize difference images for better visibility
            threshold (int): Minimum difference threshold (0-255) to consider as significant
            save_stats (bool): Whether to save comparison statistics
        """
        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)
        self.output_dir = Path(output_dir)
        self.normalize = normalize
        self.threshold = threshold
        self.save_stats = save_stats
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'missing_files': 0,
            'error_files': 0,
            'identical_files': 0,
            'different_files': 0,
            'processing_time': 0,
            'differences': []
        }
        
    def is_image_file(self, filepath):
        """Check if file is a supported image format."""
        return filepath.suffix.lower() in self.image_extensions
    
    def get_all_image_files(self, directory):
        """Get all image files from directory recursively."""
        image_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = Path(root) / file
                if self.is_image_file(filepath):
                    # Get relative path from the base directory
                    rel_path = filepath.relative_to(directory)
                    image_files.append(rel_path)
        return sorted(image_files)
    
    def load_and_resize_images(self, img1_path, img2_path):
        """
        Load two images and resize them to the same dimensions.
        
        Returns:
            tuple: (img1_array, img2_array) or (None, None) if error
        """
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Get dimensions
            w1, h1 = img1.size
            w2, h2 = img2.size
            
            # Resize to the larger dimensions to avoid information loss
            max_w = max(w1, w2)
            max_h = max(h1, h2)
            
            if (w1, h1) != (max_w, max_h):
                img1 = img1.resize((max_w, max_h), Image.Resampling.LANCZOS)
            if (w2, h2) != (max_w, max_h):
                img2 = img2.resize((max_w, max_h), Image.Resampling.LANCZOS)
            
            # Convert to numpy arrays
            img1_array = np.array(img1)
            img2_array = np.array(img2)
            
            return img1_array, img2_array
            
        except Exception as e:
            print(f"Error loading images {img1_path} and {img2_path}: {e}")
            return None, None
    
    def compute_difference(self, img1, img2):
        """
        Compute the absolute difference between two images.
        
        Returns:
            tuple: (difference_image, statistics_dict)
        """
        # Compute absolute difference
        diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
        
        # Apply threshold if specified
        if self.threshold > 0:
            diff = np.where(diff >= self.threshold, diff, 0)
        
        # Compute statistics
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        total_diff_pixels = np.sum(diff > 0)
        total_pixels = diff.shape[0] * diff.shape[1]
        diff_percentage = (total_diff_pixels / total_pixels) * 100
        
        stats = {
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'different_pixels': int(total_diff_pixels),
            'total_pixels': int(total_pixels),
            'difference_percentage': float(diff_percentage)
        }
        
        # Normalize for better visibility if requested
        if self.normalize and max_diff > 0:
            diff_normalized = (diff / max_diff * 255).astype(np.uint8)
        else:
            diff_normalized = np.clip(diff, 0, 255).astype(np.uint8)
        
        return diff_normalized, stats
    
    def save_difference_image(self, diff_image, output_path):
        """Save the difference image to the specified path."""
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PIL Image and save
        if len(diff_image.shape) == 3:
            pil_image = Image.fromarray(diff_image, 'RGB')
        else:
            pil_image = Image.fromarray(diff_image, 'L')
        
        pil_image.save(output_path)
    
    def process_image_pair(self, rel_path):
        """Process a single pair of images."""
        img1_path = self.dir1 / rel_path
        img2_path = self.dir2 / rel_path
        output_path = self.output_dir / rel_path
        
        # Check if both files exist
        if not img1_path.exists():
            print(f"Warning: {img1_path} does not exist")
            self.stats['missing_files'] += 1
            return False
            
        if not img2_path.exists():
            print(f"Warning: {img2_path} does not exist")
            self.stats['missing_files'] += 1
            return False
        
        # Load and resize images
        img1, img2 = self.load_and_resize_images(img1_path, img2_path)
        
        if img1 is None or img2 is None:
            self.stats['error_files'] += 1
            return False
        
        # Compute difference
        diff_image, file_stats = self.compute_difference(img1, img2)
        
        # Save difference image
        try:
            self.save_difference_image(diff_image, output_path)
            
            # Track statistics
            file_stats['file_path'] = str(rel_path)
            file_stats['original_size_1'] = img1.shape
            file_stats['original_size_2'] = img2.shape
            
            if file_stats['max_difference'] == 0:
                self.stats['identical_files'] += 1
            else:
                self.stats['different_files'] += 1
            
            self.stats['differences'].append(file_stats)
            self.stats['processed_files'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error saving difference image for {rel_path}: {e}")
            self.stats['error_files'] += 1
            return False
    
    def save_statistics(self):
        """Save comparison statistics to a JSON file."""
        if not self.save_stats:
            return
            
        stats_file = self.output_dir / 'comparison_statistics.json'
        ordered_stat_file = self.output_dir / 'comparison_statistics_ordered.json'
        # Add summary statistics
        self.stats['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'dir1': str(self.dir1),
            'dir2': str(self.dir2),
            'output_dir': str(self.output_dir),
            'normalize': self.normalize,
            'threshold': self.threshold,
            'success_rate': (self.stats['processed_files'] / max(self.stats['total_files'], 1)) * 100
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"Statistics saved to: {stats_file}")
            with open(ordered_stat_file, 'w') as f:
                temp = self.stats.copy()
                temp['differences'] = sorted(temp['differences'], key=lambda x: x['difference_percentage'], reverse=True)
                json.dump(temp, f, indent=2)
            print(f"Statistics saved to: {ordered_stat_file}")
        except Exception as e:
            print(f"Error saving statistics: {e}")
    
    def compare_directories(self):
        """
        Main method to compare two directories and generate difference images.
        """
        start_time = datetime.now()
        
        print(f"Comparing directories:")
        print(f"  Directory 1: {self.dir1}")
        print(f"  Directory 2: {self.dir2}")
        print(f"  Output: {self.output_dir}")
        print(f"  Normalize: {self.normalize}")
        print(f"  Threshold: {self.threshold}")
        print()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files from the first directory
        image_files = self.get_all_image_files(self.dir1)
        self.stats['total_files'] = len(image_files)
        
        if not image_files:
            print("No image files found in the first directory.")
            return
        
        print(f"Found {len(image_files)} image files to process")
        
        # Process each image pair with progress bar
        with tqdm(image_files, desc="Processing images") as pbar:
            for rel_path in pbar:
                pbar.set_postfix_str(f"Processing: {rel_path.name}")
                self.process_image_pair(rel_path)
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Print summary
        print(f"\nComparison completed!")
        print(f"Total files: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['processed_files']}")
        print(f"Identical files: {self.stats['identical_files']}")
        print(f"Different files: {self.stats['different_files']}")
        print(f"Missing files: {self.stats['missing_files']}")
        print(f"Error files: {self.stats['error_files']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        # Save statistics
        self.save_statistics()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two directories with the same structure and save image differences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_diff_comparison.py --dir1 dataset1/ --dir2 dataset2/ --output differences/
  python image_diff_comparison.py --dir1 results_v1/ --dir2 results_v2/ --output diff_v1_v2/ --normalize --threshold 10
        """
    )
    
    parser.add_argument('--dir1', required=True, help='Path to first directory')
    parser.add_argument('--dir2', required=True, help='Path to second directory')
    parser.add_argument('--output', required=True, help='Path to output directory for differences')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize difference images for better visibility (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Disable normalization of difference images')
    parser.add_argument('--threshold', type=int, default=0,
                        help='Minimum difference threshold (0-255) to consider as significant (default: 0)')
    parser.add_argument('--no-stats', action='store_true',
                        help='Do not save comparison statistics')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.dir1):
        print(f"Error: Directory 1 does not exist: {args.dir1}")
        return 1
    
    if not os.path.exists(args.dir2):
        print(f"Error: Directory 2 does not exist: {args.dir2}")
        return 1
    
    # Create comparator and run comparison
    comparator = ImageDifferenceComparator(
        dir1=args.dir1,
        dir2=args.dir2,
        output_dir=args.output,
        normalize=args.normalize,
        threshold=args.threshold,
        save_stats=not args.no_stats
    )
    
    try:
        comparator.compare_directories()
        return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    exit(main())