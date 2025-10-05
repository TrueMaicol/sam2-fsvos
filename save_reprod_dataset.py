from YoutubeVOS import YTVOSDataset
import argparse
import json
import random
from PIL import Image
import numpy as np
import os

def get_arguments():
        parser = argparse.ArgumentParser(description='FSVOS')
        parser.add_argument("--dataset_path", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--group", type=int, default=1)
        parser.add_argument("--test_query_frame_num", type=int, default=None)

        return parser.parse_args()

def save_image(image, path):
        """Save a numpy array or PIL image to the specified path."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)

if __name__ == "__main__":
    args = get_arguments()
    
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    dataset = YTVOSDataset(train=False, set_index=args.group, data_dir=args.dataset_path, test_query_frame_num=args.test_query_frame_num)

    for id in range(len(dataset.get_class_list())):
        print(f"Class ID {dataset.get_class_list()[id]}")
        support_frames, support_masks, query_set = dataset.get_reprod_test_dataset(class_id=id)
        
        for i in range(len(query_set)):
            print(f"Video {i} in query set")
            query_frames, query_masks = query_set[i]
            os.makedirs(f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/frames", exist_ok=True)
            os.makedirs(f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/ground_truth", exist_ok=True)
            os.makedirs(f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/output", exist_ok=True)

            for j, (img, mask) in enumerate(zip(support_frames, support_masks)):
                save_image(img, f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/frames/support_{j:04d}.jpg")
                # print("# non zero elements in mask: ", np.sum(mask))
                mask = (mask > 0).astype(np.uint8) * 255  # Scale to 0-255 range for visibility
                save_image(mask, f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/ground_truth/support_{j:04d}.png")

            for j, (img, mask) in enumerate(zip(query_frames, query_masks)):
                save_image(img, f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/frames/query_{(j+len(support_frames)):04d}.jpg")
                # print("# non zero elements in query mask: ", np.sum(mask))
                mask = (mask > 0).astype(np.uint8) * 255  # Scale to 0-255 range for visibility
                save_image(mask, f"{args.output_dir}/class_{dataset.get_class_list()[id]}_{i}/ground_truth/query_{(j+len(support_frames)):04d}.png")