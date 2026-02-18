import os
from pathlib import Path
from anthropic import Anthropic
from base_prompt import get_prompt, DATASET_CONFIGS
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
import json
from transformers import AutoTokenizer
from tqdm import tqdm

load_dotenv()

class LabelData:
    def __init__(self, dataset_name: str, file_path: Path, reasoning_mode: bool = False, reasoning_tokens: int = 10000):
        """
        Initialize the Data Labeller

        Args:
            dataset (str): Name of the dataset to be labelled e.g. (shear flow, rayleigh benard)
            file_path (Path): root path of the dataset on the PC
            reasoning_model (bool): True if reasoning mode is enabled, False otherwise. Default value is False
            reasoning_tokens (int): Number of budget_tokens to be provided if model is in reasoning mode. Default value is 10000
        Returns:
            None
        """
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.reasoning = reasoning_mode
        self.reasoning_budget_tokens = reasoning_tokens

        if str(file_path).endswith('.npz'):
            self.data = np.load(file_path)['field']
        elif str(file_path).endswith('.npy'):
            self.data = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file path. Use .npz or .npy file. Got {file_path}")
        self.client = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-5-20250929"
        
        config = DATASET_CONFIGS[self.dataset_name]
        self.prompt = get_prompt(**config)
    
    def _chunk_trajectories(self, batch_size: int = 20):
        """
        Batches trajectories in the dataset to be processed batchwise

        Args:
            batch_size (int): The no. of trajectories per batch (default is 20)
        
        Returns:
            List(numpy.ndarray: Stacked array of processed frames with shape (n_trajectories, height, width))
        """
        batches = []

        for i in range(0, len(self.data), batch_size):
            batches.append(self.data[i:i+batch_size])
        return batches
    
    def _trajectory_to_base64(self, trajectory_data):
        """
        Convert a 2D trajectory array to base64-encoded PNG image.
        
        Args:
            trajectory_data (numpy.ndarray): 2D array of shape (height, width)
        
        Returns:
            str: Base64-encoded PNG image string
        """
        # Standard visualization: compute figure width from data aspect ratio
        if self.dataset_name == "rayleigh_benard":
            plot_data = trajectory_data.T
        else:
            plot_data = trajectory_data

        height_fig = 15
        aspect_ratio = plot_data.shape[1] / plot_data.shape[0]
        width_fig = height_fig * aspect_ratio

        fig, ax = plt.subplots(figsize=(width_fig, height_fig))
        im = ax.imshow(plot_data, cmap='viridis')
        
        # Remove ticks for cleaner appearance
        ax.set_xticks([])
        ax.set_yticks([])
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64 

    def _generate_label(self, batch_data, batch_index, batch_size):
        """
        Create a batch request with separate requests for each trajectory.
        Uses reasoning mode depending on class instance with a budget of 10,000 tokens
        
        Args:
            batch_data (numpy.ndarray): Array of trajectories with shape (n_trajectories, height, width)
            batch_index (int): Index of the current batch
            batch_size (int): Number of trajectories per batch
        
        Returns:
            dict: Batch object containing batch_id and status
        """
        requests = []

        # uses fixed value of budget_tokens at 10000. 
        thinking_config = {"type": "enabled", "budget_tokens": self.reasoning_budget_tokens} if self.reasoning else {"type": "disabled"}
        
        for i, trajectory in enumerate(batch_data):
            traj_idx = batch_index * batch_size + i
            img_base64 = self._trajectory_to_base64(trajectory)

            request = {
                "custom_id": f"trajectory_{traj_idx}",
                "params": {
                    "model": self.model,
                    "max_tokens": 2048,
                    "thinking": thinking_config,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            },
                            {"type": "text", "text": self.prompt}
                        ]
                    }]
                }
            }
            requests.append(request)
        
        batch = self.client.messages.batches.create(requests=requests)
        return batch
    
    def _wait_for_batch(self, batch_id, poll_interval=10):
        """
        Poll the batch API until processing is complete.
        
        Args:
            batch_id (str): The batch ID to monitor
            poll_interval (int): Seconds between status checks (default: 10)
        
        Returns:
            dict: Final batch object with results
        """
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            
            tqdm.write(f"  → Status: {status}")
            
            if status == "ended":
                tqdm.write(f"  ✓ Batch complete! Processed: {batch.request_counts.succeeded}, Failed: {batch.request_counts.errored}")
                return batch
            elif status == "failed":
                raise Exception(f"Batch {batch_id} failed")
            
            time.sleep(poll_interval)
    
    def process_batches(self, batch_size: int = 10, checkpoint_file: str = None):
        """
        Labels images in batches using the Anthropic Batch API.

        Args:
            batch_size (int): The no. of trajectories per batch (default is 10)
            checkpoint_file (str): Path to save checkpoint JSON file (default: auto-generated)
        
        Returns:
            dict: Dictionary mapping trajectory indices to their labels
        """
        # Set default checkpoint file
        if checkpoint_file is None:
            checkpoint_file = f"datasets/labeled/{self.dataset_name}_checkpoint.json"
        
        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        
        # Load existing checkpoint if available
        labels = {}
        start_batch_idx = 0
        if os.path.exists(checkpoint_file):
            print(f"Found checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                labels = {int(k): v for k, v in checkpoint_data['labels'].items()}
                start_batch_idx = checkpoint_data.get('last_batch_idx', 0) + 1
            print(f"Resuming from batch {start_batch_idx + 1} with {len(labels)} existing labels\n")
        
        batches_data = self._chunk_trajectories(batch_size)
        
        print(f"Processing {len(batches_data)} batches of {batch_size} trajectories each...\n")
        
        for batch_idx, batch_data in tqdm(enumerate(batches_data), total=len(batches_data), desc="Processing batches", unit="batch"):
            # Skip already processed batches
            if batch_idx < start_batch_idx:
                continue
            
            # Create and submit batch
            batch = self._generate_label(batch_data, batch_idx, batch_size)
            tqdm.write(f"Batch {batch_idx + 1}/{len(batches_data)} submitted with ID: {batch.id}")
            
            # Wait for batch to complete
            completed_batch = self._wait_for_batch(batch.id)
            
            # Retrieve results
            for result in self.client.messages.batches.results(batch.id):
                if result.result.type == "succeeded":
                    custom_id = result.custom_id
                    trajectory_idx = int(custom_id.split('_')[1])
                    label = result.result.message.content[0].text
                    labels[trajectory_idx] = label
                else:
                    tqdm.write(f"Request {result.custom_id} failed: {result.result.error}")
            
            # Save checkpoint after each batch
            checkpoint_data = {
                'labels': labels,
                'last_batch_idx': batch_idx,
                'total_batches': len(batches_data),
                'batch_size': batch_size,
                'dataset_name': self.dataset_name
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            tqdm.write(f"  ✓ Checkpoint saved: {len(labels)} labels")
        
        print(f"\nCompleted! Generated {len(labels)} labels.")
        print(f"Checkpoint saved to: {checkpoint_file}")
        return labels
    
    def tokenize_and_save(self, labels: dict, output_file: str, tokenizer_name: str = "roberta-base", max_length: int = 1024):
        """
        Tokenize labels and save them with corresponding field data.
        
        Args:
            labels (dict): Dictionary mapping trajectory indices to label text
            output_file (str): Path to save the labeled dataset (e.g., 'shear_flow_labeled.npz')
            tokenizer_name (str): Hugging Face tokenizer to use (default: 'bert-base-uncased')
            max_length (int): Maximum token length for padding/truncation (default: 512)
        
        Returns:
            str: Path to the saved output file
        """
        print(f"\nTokenizing labels using {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure labels are sorted by trajectory index
        sorted_indices = sorted(labels.keys())
        n_trajectories = len(sorted_indices)
        
        # Check if all trajectories have labels
        if n_trajectories != len(self.data):
            print(f"Warning: {n_trajectories} labels but {len(self.data)} trajectories in data")
        
        # Tokenize all labels
        tokenized_labels = []
        raw_labels_dict = {}
        
        for idx in sorted_indices:
            label_text = labels[idx]
            raw_labels_dict[int(idx)] = label_text
            
            # Tokenize with padding and truncation
            tokens = tokenizer(
                label_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            
            # Extract input_ids (the tokenized representation)
            tokenized_labels.append(tokens['input_ids'][0])
        
        # Stack tokenized labels
        tokenized_labels_array = np.stack(tokenized_labels, axis=0)
        
        # Get corresponding field data
        fields = self.data[sorted_indices]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save tokenized data to NPZ file
        print(f"Saving labeled dataset to {output_file}...")
        np.savez_compressed(
            output_file,
            label=tokenized_labels_array,
            field=fields
        )
        
        # Save raw labels to JSON file
        json_file = output_file.replace('.npz', '_raw_labels.json')
        print(f"Saving raw labels to {json_file}...")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(raw_labels_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Saved successfully!")
        print(f"  - Labels shape: {tokenized_labels_array.shape}")
        print(f"  - Fields shape: {fields.shape}")
        print(f"  - Tokenizer: {tokenizer_name}")
        print(f"  - Max length: {max_length}")
        print(f"  - Raw labels saved to: {json_file}")

        return output_file
    
    def label(self, batch_size: int = 10, checkpoint_file: str = None, cleanup_checkpoint: bool = True):
        """
        Complete labeling pipeline with checkpointing support.
        
        Args:
            batch_size (int): Number of trajectories per batch (default: 10)
            checkpoint_file (str): Path to checkpoint file (default: auto-generated)
            cleanup_checkpoint (bool): Whether to delete checkpoint after completion (default: True)
        
        Returns:
            str: Path to the saved output file
        """
        labels = self.process_batches(batch_size=batch_size, checkpoint_file=checkpoint_file)
        output_file = self.tokenize_and_save(
            labels=labels,
            output_file="datasets/shear_flow_frame60_labeled.npz",
            tokenizer_name="bert-base-uncased",
            max_length=1024
        )
        
        # Optionally cleanup checkpoint file
        if cleanup_checkpoint and checkpoint_file:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"Checkpoint file removed: {checkpoint_file}")
        
        return output_file



if __name__ == "__main__":

    smoke_labeler = LabelData("smoke", "/home/ammark/text2physics/text2physics-labelling/datasets/trajectory.npy")
    labels = smoke_labeler.process_batches(
        batch_size=10,
        checkpoint_file="datasets/labeled/smoke_checkpoint.json"
    )
    output_file = smoke_labeler.tokenize_and_save(
        labels=labels,
        output_file="datasets/labeled/smoke_frame60_labeled.npz",
        tokenizer_name="roberta-base",
        max_length=1024
    )
