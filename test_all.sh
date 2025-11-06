for i in 4
do
    python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_tiny.pt --config sam2.1/sam2.1_hiera_t.yaml --group $i --dataset_path /datasets/Youtube-FSVOS/reprod_dataset_fold_$i --output_dir /output/CHECKPOINT_COMPARE --session_name tiny_fold_$i
    python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_small.pt --config sam2.1/sam2.1_hiera_s.yaml --group $i --dataset_path /datasets/Youtube-FSVOS/reprod_dataset_fold_$i --output_dir /output/CHECKPOINT_COMPARE --session_name small_fold_$i
    python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_base_plus.pt --config sam2.1/sam2.1_hiera_b+.yaml --group $i --dataset_path /datasets/Youtube-FSVOS/reprod_dataset_fold_$i --output_dir /output/CHECKPOINT_COMPARE --session_name base_plus_fold_$i
    python test_SAM2_FSVOS.py --checkpoint sam2.1_hiera_large.pt --config sam2.1/sam2.1_hiera_l.yaml --group $i --dataset_path /datasets/Youtube-FSVOS/reprod_dataset_fold_$i --output_dir /output/CHECKPOINT_COMPARE --session_name large_fold_$i
done
