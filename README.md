# SAM2-FSVOS: Few-Shot Video Object Segmentation with SAM 2

This repository implements **Few-Shot Video Object Segmentation (FSVOS)** using the **SAM 2 (Segment Anything Model 2)** foundation model, evaluated on the **YouTube-VIS 2019 VAL** benchmark following the methodology established by the **DANet** model.

## Overview

Few-Shot Video Object Segmentation is a challenging computer vision task where models must segment objects in video sequences based on only a few examples (support images) of novel object classes. This implementation leverages SAM 2's powerful video segmentation capabilities to perform FSVOS using a support-query paradigm:

- **Support Set**: Static images containing examples of a novel object class with corresponding ground truth masks
- **Query Video**: A video sequence containing the same novel object class that needs to be segmented

The evaluation follows the standard FSVOS protocol where models are tested on novel classes not seen during training, using the YouTube-VIS 2019 validation set.

## Key Features

- **SAM 2 Integration**: Utilizes the state-of-the-art SAM 2 model for robust video object segmentation
- **Few-Shot Learning**: Performs segmentation on novel object classes using only 5 support images
- **YouTube-VIS 2019 Benchmark**: Comprehensive evaluation on the standard FSVOS dataset
- **DANet Protocol**: Follows the evaluation methodology established by DANet for fair comparison
- **Cross-Validation**: 4-fold cross-validation setup (10 novel classes per fold)
- **Comprehensive Metrics**: Evaluation using both J-score (IoU) and F-score (boundary accuracy)

## Method

### Support-Query Framework

1. **Support Set Creation**: For each novel class, 5 static images are randomly selected from different videos, each containing the target object with ground truth masks.

2. **Query Video Processing**: The remaining videos of the same class serve as query videos to be segmented.

3. **SAM 2 Inference**: 
   - Support frames and their masks are added to SAM 2's memory
   - The model propagates segmentation across the entire query video
   - Per-frame masks are extracted and evaluated

### Class Split Strategy

Following the DANet protocol, the 40 YouTube-VIS 2019 classes are split into 4 groups for cross-validation:
- **Group 1**: Classes where `class_id % 4 == 0` are novel (test), others are base (train)
- **Group 2**: Classes where `class_id % 4 == 1` are novel (test), others are base (train)
- **Group 3**: Classes where `class_id % 4 == 2` are novel (test), others are base (train)
- **Group 4**: Classes where `class_id % 4 == 3` are novel (test), others are base (train)

## Installation

### Prerequisites
- SAM2 prerequisites
- <a href="https://github.com/youtubevos/cocoapi">CocoAPI</a> for Youtube-VIS
- <a href="https://youtube-vos.org/dataset/vis/">Youtube VIS 2019 - Validation</a> dataset

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/TrueMaicol/sam2-fsvos.git
cd sam2-fsvos
```

2. **Install dependencies and build the CUDA extension:**
```bash
pip uninstall -y SAM-2 && \
rm -f ./sam2/*.so && \
SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e .
```

3. **Download SAM 2 checkpoints:**
Note: right now the tiny version is hardcoded for testing (to be updated in the future)
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

4. **Prepare YouTube-VIS 2019 dataset:**
```bash
# Download YouTube-VIS 2019 validation set
# Extract to ./datasets/YoutubeVIS-2019/
# Expected structure:
# ./datasets/YoutubeVIS-2019/
#   ├── valid/
#   │   ├── JPEGImages/
#   │   └── instances_val_sub.json
```

## Usage

### Basic Testing

Run FSVOS evaluation on a specific group:

```bash
python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_tiny.pt --config sam2.1/sam2.1_hiera_t.yaml --dataset_path /datasets/Youtube-FSVOS/reprod_dataset_fold_1 --output_dir /output --group 1 --session_name reprod_test_fold_1_run_3
```

### Command Line Arguments

- `--group`: Cross-validation group (1-4) - determines which classes are novel vs base
- `--dataset_path`: Path to YouTube-VIS 2019 dataset directory
- `--session_name`: Unique identifier for the experiment (auto-generated if not specified)
- `--test_query_frame_num`: Limit number of query frames per video (optional)
- `--verbose`: Enable detailed logging during evaluation (to be completed)
- `--output_dir`: Custom path for output directory

## Evaluation Metrics

The implementation uses standard video object segmentation metrics:

- **J-score (Region Similarity)**: Intersection over Union (IoU) between predicted and ground truth masks
- **F-score (Contour Accuracy)**: Boundary-based evaluation measuring segmentation boundary quality
- **Mean Performance**: Average scores across all test classes in the group

Results are computed per-frame and averaged across all frames in each video, then across all videos for each class.

## Output Structure

Results are saved in the following directory structure:

```
{output_dir}/{session_name}/{video_dir_name}
├── frames/           # Combined support + query frames for SAM 2 processing
├── output/           # Generated segmentation masks with overlays
└── support_overlay/  # Visualization of support frames and query frames with ground truth masks
```

## Dataset Structure

Expected YouTube-VIS 2019 dataset organization:

```
{dataset_dir}/Youtube-FSVOS/...
    ├── JPEGImages/
    │   ├── {video_id}/
    │   │   ├── 000000.jpg
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    └── instances.json  # COCO-format annotations
```

## Template command

```bash
python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_tiny.pt --config sam2.1/sam2.1_hiera_t.yaml --dataset_path /datasets/Youtube-FSVOS/temp --output_dir /output --group 1 --session_name new_test_1
```
