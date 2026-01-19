#!/bin/bash

python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_tiny.pt --config sam2.1/sam2.1_hiera_t.yaml --group 1 --dataset_path $WORK/MARS/src/src/Matcher/output/MARS/test_fold_1_run_1/fold_1 --output_dir ./output/MARS_SAM2 --session_name full_test_fold_1
 