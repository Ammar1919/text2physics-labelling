from the_well.data import WellDataset
from pathlib import Path
from utils import setup_logging, check_space
import gc
import shutil
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
setup_logging()

class WELLDataPreProcessor:
    def __init__(self, dataset_name : str, timesteps : int):
        self.well_base_path = "hf://datasets/polymathic-ai/"
        self.dataset_name = dataset_name
        self.timesteps = timesteps 
    
    def _extract_and_downsample(self, split_name: str, frame_number: int):
        dataset = WellDataset(
            well_base_path=self.well_base_path,
            well_dataset_name=self.dataset_name,
            well_split_name=split_name
        )

        n_trajectories = len(dataset) // self.timesteps  
        logging.info(f"{split_name} split: {n_trajectories} trajectories ({len(dataset)} samples)")

        frames = []

        # Iterate over chosen frames for each trajectory
        for traj_idx in tqdm(range(n_trajectories), 
                            desc=f"Processing {split_name}",
                            unit="trajectory",
                            ncols=100):
            sample_idx = traj_idx * self.timesteps + frame_number 
            sample = dataset[sample_idx]
            fields = sample["input_fields"][0]
            tracer = fields[:, :, 0].numpy()
            tracer_downsampled = tracer[::2, ::2]
            frames.append(tracer_downsampled)

        logging.info(f"Completed processing {n_trajectories} trajectories from {split_name} split")
        return np.stack(frames, axis=0)
    
    def process_all_samples(self, frame_number: int):
        logging.info(f"Starting processing for frame {frame_number} from dataset '{self.dataset_name}'")
        split_files = []

        # Process and save each split separately
        for split in ['test', 'valid', 'train']:
            split_data = self._extract_and_downsample(split, frame_number)
            
            # Save split data
            split_output_file = f"{self.dataset_name}_frame{frame_number}_{split}.npz"
            logging.info(f"Saving {split} split to {split_output_file}...")
            np.savez_compressed(split_output_file, field=split_data)
            logging.info(f"{split} split saved successfully (shape: {split_data.shape})")
            split_files.append(split_output_file)
            
            # Free memory
            del split_data
            gc.collect()
        
        # Load and concatenate all splits
        logging.info("Loading and concatenating all splits...")
        all_splits_data = []
        for split_file in split_files:
            data = np.load(split_file)['field']
            all_splits_data.append(data)
            logging.info(f"Loaded {split_file} (shape: {data.shape})")
        
        final_data = np.concatenate(all_splits_data, axis=0)
        
        # Save final concatenated file
        output_file = f"{self.dataset_name}_frame{frame_number}.npz"
        logging.info(f"Saving concatenated data to {output_file}...")
        np.savez_compressed(output_file, field=final_data)
        
        logging.info(f"Successfully saved to {output_file} (shape: {final_data.shape})")
            
        
if __name__ == "__main__":
    shear_flow_data_preprocessor = WELLDataPreProcessor("shear_flow", 200)

    shear_flow_data_preprocessor.process_all_samples(60)


