from datetime import datetime
from utils.Evaluator import Evaluator
from YoutubeVOS import YTVOSDataset
import torch 
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import os
from PIL import Image
import cv2
from vos_inference import vos_separate_inference_per_object
import time

class SAM2_FSVOS:
    def __init__(self, checkpoint, config, session_name, dataset_path, output_dir, verbose, test_query_frame_num, apply_postprocessing,
        
        vos_optimized):
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
        self.apply_postprocessing = apply_postprocessing
        self.vos_optimized = vos_optimized

        # checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.video_predictor = build_sam2_video_predictor(
            self.model_cfg, 
            self.checkpoint, 
            device=self.device,
            apply_postprocessing=self.apply_postprocessing,
            vos_optimized=self.vos_optimized
        )
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

    def create_dirs(self, base_dir, video_query_set, support_set):
        
        frames_dir = os.path.join(base_dir, "frames")
        support_masks_dir = os.path.join(base_dir, "support_masks")
        output_dir = os.path.join(base_dir, "output")
        ground_truth_dir = os.path.join(base_dir, "ground_truth")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(support_masks_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(ground_truth_dir, exist_ok=True)

        for i, (img, mask) in enumerate(support_set):
            self.save_image(img, os.path.join(frames_dir, f"{i:05d}.jpg"))
            self.save_image(mask.astype(np.uint8) * 255, os.path.join(support_masks_dir, f"{i:05d}.png"))

        for i, (img, mask) in enumerate(video_query_set):
            self.save_image(img, os.path.join(frames_dir, f"{i + len(support_set):05d}.jpg"))
            # self.save_image(mask, os.path.join(ground_truth_dir, f"{i + len(support_set):05d}.png"))

        return frames_dir, support_masks_dir, output_dir, ground_truth_dir

    def process_video_sam2(self, video_query_img, video_query_mask, idx, dir_name, video_predictor, evaluator, support_set, data_dir="./output"):
        
        base_dir = f"{data_dir}/{dir_name}"
        video_query_set = [(img, mask) for img, mask in zip(video_query_img, video_query_mask)]
        frames_dir, support_masks_dir, prediction_dir, ground_truth_dir = self.create_dirs(base_dir, video_query_set, support_set)

        video_segments = vos_separate_inference_per_object(video_predictor, frames_dir, support_masks_dir, prediction_dir, dir_name)

        video_segmented_masks = []
        # Assuming single object per video for FSVOS
        frame_list = sorted(video_segments.keys())
        for frame_idx in frame_list:
            obj_id = list(video_segments[frame_idx].keys())[0]
            video_segmented_masks.append(video_segments[frame_idx][obj_id])

        os.makedirs(os.path.join(prediction_dir, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(prediction_dir, "overlays"), exist_ok=True)
        for i, mask in enumerate(video_segmented_masks[5:]):
            mask = mask.squeeze()
            self.save_image(mask.astype(np.uint8)*255, os.path.join(prediction_dir, "annotations", f"{i:05d}.png"))
            self.save_mask_overlay(video_query_img[i], mask, os.path.join(prediction_dir, "overlays", f"{i:05d}.png"))

        print("Updating evaluation metrics...")
        evaluator.update_evl(idx, video_query_mask, video_segmented_masks[5:])
        print(f"Evaluation of video {dir_name} completed. \n")

    def save_evaluation_results(self, output_directory, mean_f, mean_j, score_dict, elapsed_time):
        results_path = os.path.join(output_directory, "evaluation_results.txt")
        with open(results_path, 'w') as f:
            f.write(f"Mean F: {mean_f:.8f}\n")
            f.write(f"Mean J: {mean_j:.8f}\n\n")
            f.write("Detailed Scores:\n")
            for class_id, scores in score_dict.items():
                f.write(f"Class {class_id} - F: {scores['f_score']:.8f}, J: {scores['j_score']:.8f}\n")
            f.write(f"Execution time: {elapsed_time:.4f} minutes\n")
        print(f"Saved evaluation results to {results_path}")

    def load_test_data(self, n_support_frames):
        test_dataset = []
        for i, dir in enumerate(os.listdir(self.dataset_path)):
            if os.path.isdir(os.path.join(self.dataset_path, dir)):
                temp = {}
                support_set = []
                video_query_img = []
                video_query_mask = []
                for j in range(len(os.listdir(os.path.join(self.dataset_path, dir, "frames")))):
                    if j < n_support_frames:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"support_{j:04d}.jpg"))
                        img = np.array(img)
                        mask = np.array(Image.open(os.path.join(self.dataset_path, dir, "ground_truth", f"support_{j:04d}.png")))
                        mask = np.array(mask) * 255
                        support_set.append((img, mask))
                    else:
                        img = Image.open(os.path.join(self.dataset_path, dir, "frames", f"query_{j:04d}.jpg"))
                        img = np.array(img)
                        mask = np.array(Image.open(os.path.join(self.dataset_path, dir, "ground_truth", f"query_{j:04d}.png")))
                        mask = np.array(mask) * 255
                        video_query_mask.append(mask)
                        video_query_img.append(img)

                temp["dir_name"] = dir
                temp["support_set"] = support_set
                temp["video_query_img"] = video_query_img
                temp["video_query_mask"] = video_query_mask
                test_dataset.append(temp)
        return test_dataset

    def test(self, group=1):
        device = self.device
        video_predictor = self.video_predictor

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"{self.output_dir}/{self.session_name}/fold_{group}_{timestamp}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}/fold_{group}_{timestamp}"
        
        os.makedirs(output_directory, exist_ok=True)

        test_dataset = YTVOSDataset(train=False, set_index=group, data_dir=self.dataset_path, test_query_frame_num=self.test_query_frame_num)
        test_list = test_dataset.get_class_list()

        print('test_group:',group, '  test_num:', len(test_dataset), '  class_list:', test_list, ' dataset_path:', self.dataset_path)

        test_evaluations = Evaluator(class_list=test_list, verbose=self.verbose)
        support_set = []
        start_time = time.perf_counter()
        for index, data in enumerate(test_dataset):
            video_query_img, video_query_mask, new_support_img, new_support_mask, idx, dir_name, begin_new = data
            if begin_new:
                support_set = [(img, mask) for img, mask in zip(new_support_img, new_support_mask)]
                print(f"Support set for class {idx} initialized with {len(support_set)} images.")

            self.process_video_sam2(video_query_img, video_query_mask, idx, dir_name, video_predictor, test_evaluations, support_set, data_dir=output_directory)
            
            print(f"F-score list: {test_evaluations.f_score}")
            print(f"J-score list: {test_evaluations.j_score}")
        elapsed_time = time.perf_counter() - start_time
        elapsed_minutes = elapsed_time / 60.0
        print(f"Total processing time: {elapsed_minutes:.4f} minutes")

        mean_f = np.mean(test_evaluations.f_score)
        str_mean_f = 'F: %.8f ' % (mean_f)
        mean_j = np.mean(test_evaluations.j_score)
        str_mean_j = 'J: %.8f ' % (mean_j)

        f_list = ['%.8f' % n for n in test_evaluations.f_score]
        str_f_list = ' '.join(f_list)
        j_list = ['%.8f' % n for n in test_evaluations.j_score]
        str_j_list = ' '.join(j_list)
        # Generate dictionary with class id as key and f_score, j_score as values
        score_dict = {
            class_id: {"f_score": f, "j_score": j}
            for class_id, f, j in zip(test_list, test_evaluations.f_score, test_evaluations.j_score)
        }

        print(str_mean_f, str_f_list + '\n')
        print(str_mean_j, str_j_list + '\n')

        # Save evaluation results
        self.save_evaluation_results(output_directory, mean_f, mean_j, score_dict, elapsed_minutes)

        return mean_f, mean_j, score_dict
    
    def reprod_test(self, group=1):
        device = self.device
        video_predictor = self.video_predictor

        output_directory = f"{self.output_dir}/{self.session_name}"
        if self.output_dir is None:
            output_directory = f"./output/{self.session_name}"

        test_list = [i * 4 + group for i in range(10)]
        os.makedirs(output_directory, exist_ok=True)

        test_evaluations = Evaluator(class_list=test_list, verbose=self.verbose)
        test_dataset = self.load_test_data(5)
        support_set = []
        start_time = time.perf_counter()
        for index, data in enumerate(test_dataset):
            dir_name = data["dir_name"]
            support_set = data["support_set"]
            video_query_img = data["video_query_img"]
            video_query_mask = data["video_query_mask"]
            class_id = int(data["dir_name"].split('_')[1])
            print(f"Processing video {dir_name} with class ID {class_id}")
            self.process_video_sam2(video_query_img, video_query_mask, class_id, dir_name, video_predictor, test_evaluations, support_set, data_dir=output_directory)

        mean_f = np.mean(test_evaluations.f_score)
        str_mean_f = 'F: %.8f ' % (mean_f)
        mean_j = np.mean(test_evaluations.j_score)
        str_mean_j = 'J: %.8f ' % (mean_j)
        
        elapsed_time = time.perf_counter() - start_time
        elapsed_minutes = elapsed_time / 60.0
        print(f"Total processing time: {elapsed_minutes:.4f} minutes")
        
        f_list = ['%.8f' % n for n in test_evaluations.f_score]
        str_f_list = ' '.join(f_list)
        j_list = ['%.8f' % n for n in test_evaluations.j_score]
        str_j_list = ' '.join(j_list)
        # Generate dictionary with class id as key and f_score, j_score as values
        score_dict = {
            class_id: {"f_score": f, "j_score": j}
            for class_id, f, j in zip(test_list, test_evaluations.f_score, test_evaluations.j_score)
        }

        print(str_mean_f, str_f_list + '\n')
        print(str_mean_j, str_j_list + '\n')

        # Save evaluation results
        self.save_evaluation_results(output_directory, mean_f, mean_j, score_dict, elapsed_minutes)


