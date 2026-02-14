from the_well.data import WellDataset
from pathlib import Path
from utils import setup_logging, check_space
import gc
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
    
    def _extract_and_process(self, split_name: str, frame_number: int, downsample: bool = False):
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
            if downsample:
                tracer_processed = tracer[::2, ::2]
            else:
                tracer_processed = tracer
            frames.append(tracer_processed)

        logging.info(f"Completed processing {n_trajectories} trajectories from {split_name} split")
        return np.stack(frames, axis=0)

    def process_single_split(self, split_name: str, frame_number: int, downsample: bool = False) -> str:
        split_output_file = f"{self.dataset_name}_frame{frame_number}_{split_name}.npz"
        if Path(split_output_file).exists():
            logging.info(f"File {split_output_file} already exists. Skipping processing.")
            return split_output_file
        
        logging.info(f"Processing {split_name} split for frame {frame_number}")
        split_data = self._extract_and_process(split_name, frame_number, downsample)
        
        # Save split data using the standard naming convention
        logging.info(f"Saving {split_name} split to {split_output_file}...")
        np.savez_compressed(split_output_file, field=split_data)
        logging.info(f"{split_name} split saved successfully (shape: {split_data.shape})")
        
        # Free memory
        del split_data
        gc.collect()
        
        return split_output_file
    
    def concatenate_splits(self, frame_number: int, splits: list = None) -> str:
        output_file = f"{self.dataset_name}_frame{frame_number}.npz"
        if Path(output_file).exists():
            logging.info(f"File {output_file} already exists. Skipping concatenation.")
            return output_file
        
        if splits is None:
            splits = ['test', 'valid', 'train']
        
        # Build file paths using the standard naming convention
        split_files = [f"{self.dataset_name}_frame{frame_number}_{split}.npz" for split in splits]
        
        logging.info(f"Loading and concatenating splits: {splits}")
        all_splits_data = []
        
        for split_file in split_files:
            data = np.load(split_file)['field']
            all_splits_data.append(data)
            logging.info(f"Loaded {split_file} (shape: {data.shape})")
        
        final_data = np.concatenate(all_splits_data, axis=0)
        
        # Save final concatenated file
        logging.info(f"Saving concatenated data to {output_file}...")
        np.savez_compressed(output_file, field=final_data)
        logging.info(f"Successfully saved to {output_file} (shape: {final_data.shape})")
        
        return output_file
    
    def process_split_in_batches(self, split_name: str, frame_number: int, batch_size: int = 100, downsample: bool = False) -> list:
        dataset = WellDataset(
            well_base_path=self.well_base_path,
            well_dataset_name=self.dataset_name,
            well_split_name=split_name
        )

        n_trajectories = len(dataset) // self.timesteps  
        logging.info(f"{split_name} split: {n_trajectories} trajectories ({len(dataset)} samples)")
        
        batch_files = []
        n_batches = (n_trajectories + batch_size - 1) // batch_size  # Ceiling division
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_trajectories)
            
            batch_output_file = f"{self.dataset_name}_frame{frame_number}_{split_name}_batch{batch_idx}.npz"
            
            # Skip if batch file already exists
            if Path(batch_output_file).exists():
                logging.info(f"Batch file {batch_output_file} already exists. Skipping.")
                batch_files.append(batch_output_file)
                continue
            
            logging.info(f"Processing batch {batch_idx + 1}/{n_batches} (trajectories {start_idx} to {end_idx - 1})")
            
            frames = []
            for traj_idx in tqdm(range(start_idx, end_idx), 
                                desc=f"Processing {split_name} batch {batch_idx + 1}/{n_batches}",
                                unit="trajectory",
                                ncols=100):
                sample_idx = traj_idx * self.timesteps + frame_number 
                sample = dataset[sample_idx]
                fields = sample["input_fields"][0]
                tracer = fields[:, :, 0].numpy()
                if downsample:
                    tracer_processed = tracer[::2, ::2]
                else:
                    tracer_processed = tracer
                frames.append(tracer_processed)
            
            # Stack and save batch
            batch_data = np.stack(frames, axis=0)
            logging.info(f"Saving batch to {batch_output_file}...")
            np.savez_compressed(batch_output_file, field=batch_data)
            logging.info(f"Batch saved successfully (shape: {batch_data.shape})")
            batch_files.append(batch_output_file)
            
            # Free memory
            del batch_data, frames
            gc.collect()
        
        logging.info(f"Completed processing {n_batches} batches for {split_name} split")
        return batch_files
    
    def concatenate_batches(self, split_name: str, frame_number: int, batch_files: list = None) -> str:
        output_file = f"{self.dataset_name}_frame{frame_number}_{split_name}.npz"
        
        if Path(output_file).exists():
            logging.info(f"File {output_file} already exists. Skipping concatenation.")
            return output_file
        
        # Auto-detect batch files if not provided
        if batch_files is None:
            batch_files = sorted(Path('.').glob(f"{self.dataset_name}_frame{frame_number}_{split_name}_batch*.npz"))
            batch_files = [str(f) for f in batch_files]
            logging.info(f"Auto-detected {len(batch_files)} batch files")
        
        if not batch_files:
            logging.error(f"No batch files found for {split_name} split")
            raise FileNotFoundError(f"No batch files found for {split_name} split")
        
        logging.info(f"Loading and concatenating {len(batch_files)} batches for {split_name} split")
        all_batch_data = []
        
        for batch_file in batch_files:
            data = np.load(batch_file)['field']
            all_batch_data.append(data)
            logging.info(f"Loaded {batch_file} (shape: {data.shape})")
        
        final_data = np.concatenate(all_batch_data, axis=0)
        
        # Save concatenated file
        logging.info(f"Saving concatenated data to {output_file}...")
        np.savez_compressed(output_file, field=final_data)
        logging.info(f"Successfully saved to {output_file} (shape: {final_data.shape})")
        
        # Free memory
        del all_batch_data, final_data
        gc.collect()
        
        return output_file
    
    def process_all_samples(self, frame_number: int, downsample: bool = False):
        output_file = f"{self.dataset_name}_frame{frame_number}.npz"
        if Path(output_file).exists():
            logging.info(f"File {output_file} already exists. Skipping processing.")
            return output_file
        
        logging.info(f"Starting processing for frame {frame_number} from dataset '{self.dataset_name}'")
        split_files = []

        # Process and save each split separately
        for split in ['test', 'valid', 'train']:
            split_data = self._extract_and_process(split, frame_number, downsample)
            
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
        logging.info(f"Saving concatenated data to {output_file}...")
        np.savez_compressed(output_file, field=final_data)
        
        logging.info(f"Successfully saved to {output_file} (shape: {final_data.shape})")
        
if __name__ == "__main__":

    shear_flow_data_preprocessor = WELLDataPreProcessor("shear_flow", 200)
    shear_flow_data_preprocessor.concatenate_splits(60)

'''
    # Example: Process splits in batches (safer for memory)
    rayleigh_benard_processor = WELLDataPreProcessor("rayleigh_benard", 200)
    for split in ["test", "valid", "train"]:
        # Process split in batches of 100 trajectories each
        batch_files = rayleigh_benard_processor.process_split_in_batches(split, 60, batch_size=100)
        # Concatenate all batches into a single split file
        rayleigh_benard_processor.concatenate_batches(split, 60, batch_files)
    # Finally concatenate all splits
    rayleigh_benard_processor.concatenate_splits(60)
'''

'''
    shear_flow_data_preprocessor = WELLDataPreProcessor("shear_flow", 200)

    for split in ["test", "valid", "train"]:
        shear_flow_data_preprocessor.process_single_split(split, 60, True)
    shear_flow_data_preprocessor.concatenate_splits(60)
'''

    


