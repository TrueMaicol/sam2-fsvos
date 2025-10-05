# Image Directory Difference Comparison Tool

This tool compares two directories with the same structure containing images and saves the pixel-wise differences to a new output directory, maintaining the same folder structure.

## Features

- **Recursive directory comparison**: Processes all images in subdirectories
- **Multiple image format support**: JPG, PNG, BMP, TIFF, WebP
- **Automatic image resizing**: Handles images of different sizes by resizing to larger dimensions
- **Configurable difference threshold**: Filter out minor differences
- **Normalization options**: Enhance visibility of differences
- **Detailed statistics**: JSON report with comparison metrics
- **Progress tracking**: Real-time progress bar during processing
- **Error handling**: Graceful handling of missing files and processing errors

## Installation

Ensure you have the required dependencies installed:

```bash
pip install numpy pillow opencv-python tqdm
```

## Usage

### Command Line Interface

Basic usage:
```bash
python image_diff_comparison.py --dir1 path/to/first/directory --dir2 path/to/second/directory --output path/to/output/directory
```

Advanced usage with options:
```bash
python image_diff_comparison.py \
    --dir1 "dataset_v1/" \
    --dir2 "dataset_v2/" \
    --output "differences/" \
    --normalize \
    --threshold 10 \
    --no-stats
```

### Command Line Options

- `--dir1`: Path to the first directory (required)
- `--dir2`: Path to the second directory (required)  
- `--output`: Path to output directory for difference images (required)
- `--normalize`: Normalize difference images for better visibility (default: True)
- `--no-normalize`: Disable normalization of difference images
- `--threshold`: Minimum difference threshold (0-255) to consider significant (default: 0)
- `--no-stats`: Do not save comparison statistics

### Python API

```python
from image_diff_comparison import ImageDifferenceComparator

# Create comparator instance
comparator = ImageDifferenceComparator(
    dir1="path/to/first/directory",
    dir2="path/to/second/directory",
    output_dir="path/to/output/directory",
    normalize=True,      # Enhance difference visibility
    threshold=10,        # Ignore differences < 10 (0-255 scale)
    save_stats=True     # Save detailed statistics
)

# Run the comparison
comparator.compare_directories()
```

## Examples

### Example 1: Basic Comparison
Compare two model output directories:
```bash
python image_diff_comparison.py \
    --dir1 "model_outputs_v1/" \
    --dir2 "model_outputs_v2/" \
    --output "model_comparison/"
```

### Example 2: Quality Control with Threshold
Filter out minor differences for quality control:
```bash
python image_diff_comparison.py \
    --dir1 "original_dataset/" \
    --dir2 "processed_dataset/" \
    --output "processing_differences/" \
    --threshold 25
```

### Example 3: Dataset Change Analysis
Analyze changes between dataset versions:
```bash
python image_diff_comparison.py \
    --dir1 "dataset_v1.0/" \
    --dir2 "dataset_v2.0/" \
    --output "dataset_changes/" \
    --normalize
```

Run the example script to see the tool in action:
```bash
python example_image_diff.py
```

## Output Structure

The tool maintains the exact directory structure of the input directories:

```
Input Structure:
dir1/
├── images/
│   ├── subfolder1/
│   │   ├── image1.jpg
│   │   └── image2.png
│   └── image3.jpg
└── other_folder/
    └── image4.png

Output Structure:
output/
├── images/
│   ├── subfolder1/
│   │   ├── image1.jpg       # Difference image
│   │   └── image2.png       # Difference image
│   └── image3.jpg           # Difference image
├── other_folder/
│   └── image4.png           # Difference image
└── comparison_statistics.json  # Detailed report
```

## Statistics Report

The tool generates a comprehensive JSON report containing:

- **Processing Summary**: Total files, success rate, processing time
- **File-by-file Statistics**: Difference metrics for each image pair
- **Overall Metrics**: Identical vs different files, error counts
- **Configuration**: Settings used for the comparison

Example statistics:
```json
{
  "total_files": 150,
  "processed_files": 148,
  "identical_files": 45,
  "different_files": 103,
  "missing_files": 2,
  "error_files": 0,
  "processing_time": 23.45,
  "summary": {
    "success_rate": 98.67,
    "timestamp": "2025-10-06T14:30:00",
    "normalize": true,
    "threshold": 10
  }
}
```

## Image Difference Computation

The tool computes pixel-wise absolute differences:

1. **Load and Resize**: Images are loaded and resized to the largest dimensions
2. **Difference Calculation**: `|pixel1 - pixel2|` for each RGB channel
3. **Threshold Application**: Optional filtering of small differences
4. **Normalization**: Optional enhancement for better visibility
5. **Save**: Results saved as images in the same format

## Use Cases

- **Model Evaluation**: Compare outputs from different model versions
- **Dataset Quality Control**: Identify changes in processed datasets  
- **Algorithm Testing**: Evaluate image processing algorithm effects
- **Version Control**: Track changes in image-based projects
- **Regression Testing**: Detect unintended changes in visual outputs
- **A/B Testing**: Compare different image processing approaches

## Error Handling

The tool handles various error conditions gracefully:

- **Missing files**: Reports missing files without stopping processing
- **Format errors**: Skips unsupported or corrupted images
- **Size mismatches**: Automatically resizes images to compatible dimensions
- **Permission errors**: Reports access issues and continues with other files

## Performance Considerations

- **Memory Usage**: Images are processed one at a time to minimize memory usage
- **Large Datasets**: Progress bar provides feedback for long-running operations
- **Parallel Processing**: Currently single-threaded; could be extended for parallel processing
- **Output Size**: Difference images typically compress well due to sparse differences

## Troubleshooting

### Common Issues:

1. **"No image files found"**: Check that the directory contains supported image formats
2. **"Directory does not exist"**: Verify input directory paths are correct
3. **"Permission denied"**: Ensure read access to input directories and write access to output
4. **"Out of memory"**: For very large images, consider preprocessing to reduce size

### Supported Image Formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WebP (.webp)

## Contributing

Feel free to submit issues and enhancement requests!