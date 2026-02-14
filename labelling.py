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
from transformers import AutoTokenizer
from tqdm import tqdm

"""
1 - Visualize data?

2 - Batch messages

3 - Attach labels 
"""

load_dotenv()

class LabelData:
    def __init__(self, dataset_name: str, file_path: Path):
        """
        Initialize the Data Labeller

        Args:
            dataset (str): Name of the dataset to be labelled e.g. (shear flow, rayleigh benard)
            file_path (Path): root path of the dataset on the PC
        
        Returns:
            None
        """
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.data = np.load(file_path)['field']
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
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(trajectory_data, cmap='viridis', origin='lower', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        ax.set_title(f'{self.dataset_name} Trajectory')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64 

    def _generate_label(self, batch_data, batch_index, batch_size):
        """
        Create a batch request with separate requests for each trajectory.
        
        Args:
            batch_data (numpy.ndarray): Array of trajectories with shape (n_trajectories, height, width)
            batch_index (int): Index of the current batch
            batch_size (int): Number of trajectories per batch
        
        Returns:
            dict: Batch object containing batch_id and status
        """
        requests = []
        
        for i, trajectory in enumerate(batch_data):
            traj_idx = batch_index * batch_size + i
            img_base64 = self._trajectory_to_base64(trajectory)

            request = {
                "custom_id": f"trajectory_{traj_idx}",
                "params": {
                    "model": self.model,
                    "max_tokens": 2048,
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
    
    def process_batches(self, batch_size: int = 10):
        """
        Labels images in batches using the Anthropic Batch API.

        Args:
            batch_size (int): The no. of trajectories per batch (default is 10)
        
        Returns:
            dict: Dictionary mapping trajectory indices to their labels
        """
        labels = {}
        batches_data = self._chunk_trajectories(batch_size)
        
        print(f"Processing {len(batches_data)} batches of {batch_size} trajectories each...\n")
        
        for batch_idx, batch_data in tqdm(enumerate(batches_data), total=len(batches_data), desc="Processing batches", unit="batch"):
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
        
        print(f"\nCompleted! Generated {len(labels)} labels.")
        return labels
    
    def tokenize_and_save(self, labels: dict, output_file: str, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Tokenize labels and save them with corresponding field data.
        
        Args:
            labels (dict): Dictionary mapping trajectory indices to label text
            output_file (str): Path to save the labeled dataset (e.g., 'shear_flow_labeled.npz')
            tokenizer_name (str): Hugging Face tokenizer to use (default: 'bert-base-uncased')
            max_length (int): Maximum token length for padding/truncation (default: 512)
        
        Returns:
            tuple: (output_file path, list of raw label texts)
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
        raw_labels = []
        
        for idx in sorted_indices:
            label_text = labels[idx]
            raw_labels.append(label_text)
            
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
        
        # Save to NPZ file
        print(f"Saving labeled dataset to {output_file}...")
        np.savez_compressed(
            output_file,
            label=tokenized_labels_array,
            field=fields
        )
        
        print(f"Saved successfully!")
        print(f"  - Labels shape: {tokenized_labels_array.shape}")
        print(f"  - Fields shape: {fields.shape}")
        print(f"  - Tokenizer: {tokenizer_name}")
        print(f"  - Max length: {max_length}")

        

        return output_file

    def test_first_batch(self, batch_size: int = 10, output_file: str = None, tokenizer_name: str = "bert-base-uncased", max_length: int = 512):
        """
        Test function to process only the first batch of trajectories.
        
        Args:
            batch_size (int): Number of trajectories to process (default: 10)
            output_file (str): Path to save test results. If None, uses default naming.
            tokenizer_name (str): Hugging Face tokenizer to use (default: 'bert-base-uncased')
            max_length (int): Maximum token length for padding/truncation (default: 512)
        
        Returns:
            tuple: (labels dict, output_file path)
        """
        print(f"\n{'='*60}")
        print(f"TEST MODE: Processing first {batch_size} trajectories only")
        print(f"{'='*60}\n")
        
        # Temporarily store original data and limit to first batch
        original_data = self.data
        self.data = self.data[:batch_size]
        
        print(f"Dataset shape (test): {self.data.shape}")
        print(f"Processing {batch_size} trajectories...\n")
        
        try:
            # Create single batch request
            batch_data = self.data
            batch = self._generate_label(batch_data, batch_index=0, batch_size=batch_size)
            print(f"Batch submitted with ID: {batch.id}")
            
            # Wait for completion
            completed_batch = self._wait_for_batch(batch.id)
            
            # Retrieve results
            labels = {}
            print(f"\nRetrieving results...")
            for result in self.client.messages.batches.results(batch.id):
                if result.result.type == "succeeded":
                    custom_id = result.custom_id
                    trajectory_idx = int(custom_id.split('_')[1])
                    label = result.result.message.content[0].text
                    labels[trajectory_idx] = label
                    
                    # Show first label as example
                    if trajectory_idx == 0:
                        print(f"\nExample label (trajectory 0):")
                        print(f"-" * 60)
                        print(label[:300] + "..." if len(label) > 300 else label)
                        print(f"-" * 60)
                else:
                    print(f"Request {result.custom_id} failed: {result.result.error}")
            
            print(f"\n✓ Generated {len(labels)} labels")
            
            # Set default output file name
            if output_file is None:
                output_file = f"datasets/labeled/{self.dataset_name}_frame60_test_batch.npz"
            
            # Tokenize and save
            saved_file, raw_labels = self.tokenize_and_save(
                labels=labels,
                output_file=output_file,
                tokenizer_name=tokenizer_name,
                max_length=max_length
            )
            
            # Load and display summary
            print(f"\n{'='*60}")
            print(f"TEST RESULTS SUMMARY")
            print(f"{'='*60}")
            
            test_data = np.load(saved_file, allow_pickle=True)
            print(f"Saved to: {saved_file}")
            print(f"\nDataset structure:")
            print(f"  - labels:              {test_data['labels'].shape}")
            print(f"  - field:               {test_data['field'].shape}")
            
            # Show token length statistics
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            token_lengths = []
            for label_text in raw_labels:
                tokens = tokenizer.encode(label_text, add_special_tokens=True)
                token_lengths.append(len(tokens))
            
            print(f"\nToken length statistics:")
            print(f"  - Min:     {min(token_lengths)} tokens")
            print(f"  - Max:     {max(token_lengths)} tokens")
            print(f"  - Mean:    {np.mean(token_lengths):.1f} tokens")
            print(f"  - Median:  {np.median(token_lengths):.0f} tokens")
            
            if max(token_lengths) > max_length:
                print(f"\n⚠ Warning: {sum(1 for l in token_lengths if l > max_length)} labels exceed max_length={max_length}")
            
            print(f"{'='*60}\n")
            
            return labels, saved_file
            
        finally:
            # Restore original data
            self.data = original_data
    
    def label(self):
        labels = self.process_batches()
        output_file, _ = self.tokenize_and_save(
            labels=labels,
            output_file="datasets/shear_flow_frame60_labeled.npz",
            tokenizer_name="bert-base-uncased",
            max_length=512
        )
        return output_file



if __name__ == "__main__":

    rb_labeler = LabelData("rayleigh_benard", "/home/ammark/text2physics/text2physics-labelling/datasets/rayleigh_benard_frame60.npz")
    
    labels = rb_labeler.process_batches()
    output_file = rb_labeler.tokenize_and_save(
        labels=labels,
        output_file="datasets/labeled/rayleigh_benard_frame60_labeled.npz",
        tokenizer_name="roberta-base",
        max_length=512
    )