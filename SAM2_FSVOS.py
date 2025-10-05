import torch 
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import os
from PIL import Image
import cv2

class SAM2_FSVOS:
    def __init__(self, checkpoint, config, session_name, dataset_path, output_dir, verbose, test_query_frame_num):
        # self.args = args
        if checkpoint is None:
            print("No checkpoint path provided. Exiting")
            return

        if config is None:
            print("No config path provided. Exiting")
            return

        self.checkpoint = f"./checkpoints/{checkpoint}"
        self.model_cfg = f"./configs/{config}"
        self.session_name = session_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.test_query_frame_num = test_query_frame_num        

        # checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        print("Successfully loaded SAM2 model")
                

    def save_mask_overlay(self, image, mask, output_path):
        """Save image with mask overlay"""
        # Ensure image is numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
    
        # print(f"Image shape: {image.shape[:2]}, Mask shape: {mask.shape[:2]}")
        
        # Ensure mask is 2D boolean array
        mask = mask.squeeze()  # Remove any extra dimensions
        if mask.ndim > 2:
            mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.max(axis=2)
        
        if mask.shape[:2] != image.shape[:2]:
            # print(f"Warning: Mask shape {mask.shape} doesn't match image shape {image.shape}")
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Ensure mask is boolean
        mask = mask.astype(bool)
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        colored_mask[mask] = [0, 255, 0]  # Green overlay
        
        # Blend with original image
        overlay = cv2.addWeighted(image.astype(np.uint8), 0.7, colored_mask.astype(np.uint8), 0.3, 0)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def save_image(self, image, path):
        """Save a numpy array or PIL image to the specified path."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)

    def create_dirs(self, base_dir, video_query_img, support_set):
        
        frames_dir = os.path.join(base_dir, "frames")
        output_dir = os.path.join(base_dir, "output")
        ground_truth_dir = os.path.join(base_dir, "ground_truth")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        for i, (img, _) in enumerate(support_set):
            self.save_image(img, os.path.join(frames_dir, f"{i:04d}.jpg"))

        for i, img in enumerate(video_query_img):
            self.save_image(img, os.path.join(frames_dir, f"{i + len(support_set):04d}.jpg"))

        return frames_dir, output_dir, ground_truth_dir

    def process_video_sam2(self, video_query_img, dir_name, video_predictor, support_set, device, data_dir="./output"):

        # print(f"Length of query gt: {len(video_query_mask)}")
        # print(f"Shape of gt mask: {video_query_mask[0].shape}")
        # print(f"MASK {np.sum(video_query_mask[0])} non-zero elements")
        # print(f"Starting the segmentation of test {dir_name} with class {idx}")
        print(f"Processing video: {dir_name}")
        base_dir = f"{data_dir}/{dir_name}"

        frames_dir, prediction_dir, ground_truth_dir = self.create_dirs(base_dir, video_query_img, support_set)
        
        # Initialize inference state with all frames directory
        inference_state = video_predictor.init_state(video_path=frames_dir)

        print("Loading support frames and their masks into SAM2")
        
        obj_id = 1  # Use same object ID for all support frames and query frames

        # Add support frame masks to the predictor
        for i, (img, mask) in enumerate(support_set):
            mask_binary = (mask > 0).astype(np.uint8)  # Ensure binary mask
            mask_tensor = torch.tensor(mask_binary, dtype=torch.bool, device=device)
            
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=obj_id,
                mask=mask_tensor
            )
            self.save_mask_overlay(img, mask_binary > 0, os.path.join(ground_truth_dir, f"support_{i:04d}.png"))
            print(f"Added support frame {i}")

        print("Processing query video...")

        # Load query frames in sorted order
        segmented_masks = []
        # Propagate masks to this frame
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[j] > 0.0).cpu().numpy()
                for j, out_obj_id in enumerate(out_obj_ids)
            }

        for i, query_img in enumerate(video_query_img):
            print(f"Processing query frame {i}")
            query_frame = np.array(query_img)
            query_frame_idx = len(support_set) + i
            
            # Extract mask for current query frame
            if query_frame_idx in video_segments and obj_id in video_segments[query_frame_idx]:
                mask = video_segments[query_frame_idx][obj_id]
                segmented_masks.append(mask)
                print(np.sum(mask), "non-zero elements in the predicted mask")
                # Save visualization
                self.save_mask_overlay(query_frame, mask, f"{prediction_dir}/out_{i:04d}.png")
                
                print(f"Successfully processed query frame {i}")
            else:
                # No mask found, append empty mask
                empty_mask = np.zeros((query_frame.shape[0], query_frame.shape[1]), dtype=bool)
                segmented_masks.append(empty_mask)
                print(f"No mask found for query frame {i}")
        
        print(f"Segmentation complete! Generated {len(segmented_masks)} masks")
        
        return segmented_masks

    def load_test_data(self, n_support_frames):
        test_dataset = []
        for i, dir in enumerate(os.listdir(self.dataset_path)):
            if os.path.isdir(os.path.join(self.dataset_path, dir)):
                temp = {}
                support_set = []
                video_query_img = []
                for j in range(len(os.listdir(os.path.join(self.dataset_path, dir, "frames")))):
                    if j < n_support_frames:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"support_{j:04d}.jpg"))
                        img = np.array(img)
                        mask = np.array(Image.open(os.path.join(self.dataset_path, dir, "ground_truth", f"support_{j:04d}.png")))
                        mask = np.array(mask)
                        support_set.append((img, mask))
                    else:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"query_{j:04d}.jpg"))
                        img = np.array(img)
                        video_query_img.append(img)

                temp["dir_name"] = dir
                temp["support_set"] = support_set
                temp["video_query_img"] = video_query_img
                test_dataset.append(temp)
        return test_dataset

    def test(self):
        device = self.device
        video_predictor = self.video_predictor

        output_directory = f"{self.output_dir}/{self.session_name}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}"
        
        os.makedirs(output_directory, exist_ok=True)

        test_dataset = self.load_test_data(5)

        support_set = []
        for index, data in enumerate(test_dataset):
            dir_name = data["dir_name"]
            support_set = data["support_set"]
            video_query_img = data["video_query_img"]

            self.process_video_sam2(video_query_img, dir_name, video_predictor, support_set, device, data_dir=output_directory)


