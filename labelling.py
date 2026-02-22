import argparse
import base64
import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from base_prompt import DATASET_CONFIGS, get_prompt


load_dotenv()
OPENAI_MODEL = "gpt-5.2"

"""
Pipeline overview:
1) Load NPZ flow fields (key: field/trajectory)
2) Render each 2D field to PNG (viridis, origin='lower')
3) Send image + structured prompt to OpenAI
4) Write per-sample checkpoints for resume
5) Tokenize and save final NPZ outputs
"""

class LabelData:
    def __init__(self, dataset_name: str, file_path: str):
        self.dataset_name = dataset_name
        self.file_path = Path(file_path)
        self.data = self._load_npz_field(self.file_path)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset_name '{self.dataset_name}'. "
                f"Valid options: {list(DATASET_CONFIGS.keys())}"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL
        self.prompt = get_prompt(self.dataset_name)

    def _load_npz_field(self, file_path: Path) -> np.ndarray:
        # Accept both dataset key conventions used in this project.
        with np.load(file_path, allow_pickle=True) as npz_data:
            if "field" in npz_data:
                return npz_data["field"]
            if "trajectory" in npz_data:
                return npz_data["trajectory"]
            raise KeyError(
                f"Neither 'field' nor 'trajectory' key found in {file_path}. "
                f"Available keys: {list(npz_data.keys())}"
            )

    def _chunk_trajectories(self, batch_size: int = 20) -> List[np.ndarray]:
        batches = []
        for i in range(0, len(self.data), batch_size):
            batches.append(self.data[i : i + batch_size])
        return batches

    def _save_debug_preview(self, field_2d: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(output_path), field_2d, cmap="viridis", origin="lower", format="png")

    def _trajectory_to_base64(self, trajectory_data: np.ndarray) -> str:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(trajectory_data, cmap="viridis", origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.set_title(f"{self.dataset_name} Trajectory")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _extract_text(self, response) -> str:
        if getattr(response, "output_text", None):
            return response.output_text.strip()

        texts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    txt = getattr(content, "text", "")
                    if txt:
                        texts.append(txt)
        return "\n".join(texts).strip()

    def _generate_single_label(self, trajectory: np.ndarray) -> str:
        # One request per sample keeps retries/checkpointing simple and robust.
        img_base64 = self._trajectory_to_base64(trajectory)
        data_url = f"data:image/png;base64,{img_base64}"

        max_retries = 3
        response = None
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": self.prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                    max_output_tokens=1024,
                    timeout=120,
                    reasoning={"effort": "high"},
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    raise
                wait_seconds = min(2 ** attempt, 8)
                tqdm.write(
                    f"Request failed (attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
        if response is None:
            raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

        text = self._extract_text(response)
        if not text:
            raise RuntimeError("OpenAI returned empty text output.")
        return text

    def _checkpoint_path(self, output_file: str) -> Path:
        return Path(f"{output_file}.checkpoint.jsonl")

    def _load_checkpoint_labels(self, checkpoint_path: Path) -> Dict[int, str]:
        labels: Dict[int, str] = {}
        if not checkpoint_path.exists():
            return labels

        with checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    idx = int(rec["idx"])
                    text = str(rec["label_text"])
                    labels[idx] = text
                except Exception:
                    # Skip malformed rows but keep valid progress.
                    continue
        return labels

    def _append_checkpoint_entry(self, checkpoint_path: Path, idx: int, label_text: str) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": idx, "label_text": label_text}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def process_batches(
        self,
        batch_size: int = 10,
        output_file: str = "",
        checkpoint_every: int = 1,
    ) -> Dict[int, str]:
        checkpoint_path = self._checkpoint_path(output_file)
        labels = self._load_checkpoint_labels(checkpoint_path) if output_file else {}
        if labels:
            print(f"Loaded {len(labels)} labels from checkpoint: {checkpoint_path}")

        batches_data = self._chunk_trajectories(batch_size)
        print(f"Processing {len(batches_data)} batches of up to {batch_size} trajectories each...\n")
        newly_saved = 0

        for batch_idx, batch_data in tqdm(
            enumerate(batches_data),
            total=len(batches_data),
            desc="Processing batches",
            unit="batch",
        ):
            for i, trajectory in tqdm(
                enumerate(batch_data),
                total=len(batch_data),
                desc=f"Batch {batch_idx + 1}/{len(batches_data)} trajectories",
                unit="traj",
                leave=False,
            ):
                traj_idx = batch_idx * batch_size + i
                if traj_idx in labels:
                    continue
                try:
                    label = self._generate_single_label(trajectory)
                    labels[traj_idx] = label
                    if output_file:
                        self._append_checkpoint_entry(checkpoint_path, traj_idx, label)
                        newly_saved += 1
                        if newly_saved % max(1, checkpoint_every) == 0:
                            tqdm.write(
                                f"Checkpoint updated: {len(labels)} labels saved "
                                f"({checkpoint_path.name})"
                            )
                except Exception as exc:
                    tqdm.write(f"Trajectory {traj_idx} failed: {exc}")

        print(f"\nCompleted! Generated {len(labels)} labels.")
        return labels

    def _safe_save_npz(self, output_path: Path, **arrays) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{output_path.stem}_tmp_",
            suffix=".npz",
            dir=str(output_path.parent),
        )
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return str(output_path)

    def tokenize_and_save(
        self,
        labels: Dict[int, str],
        output_file: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 1024,
    ) -> Tuple[str, List[str]]:
        print(f"\nTokenizing labels using {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        sorted_indices = sorted(labels.keys())
        if not sorted_indices:
            raise ValueError("No labels to tokenize. All API calls may have failed.")

        if len(sorted_indices) != len(self.data):
            print(f"Warning: {len(sorted_indices)} labels but {len(self.data)} trajectories in data")

        tokenized_labels = []
        raw_labels: List[str] = []
        for idx in sorted_indices:
            label_text = labels[idx]
            raw_labels.append(label_text)
            tokens = tokenizer(
                label_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            tokenized_labels.append(tokens["input_ids"][0])

        tokenized_labels_array = np.stack(tokenized_labels, axis=0).astype(np.int32)
        fields = self.data[sorted_indices]
        output_path = Path(output_file)

        print(f"Saving labeled dataset to {output_path}...")
        saved_path = self._safe_save_npz(
            output_path=output_path,
            label=tokenized_labels_array,
            field=fields,
        )

        print("Saved successfully!")
        print(f"  - Labels shape: {tokenized_labels_array.shape}")
        print(f"  - Fields shape: {fields.shape}")
        print(f"  - Tokenizer: {tokenizer_name}")
        print(f"  - Max length: {max_length}")

        return saved_path, raw_labels

    def label(
        self,
        output_file: str,
        batch_size: int = 10,
        tokenizer_name: str = "roberta-base",
        max_length: int = 1024,
        checkpoint_every: int = 1,
    ) -> str:
        labels = self.process_batches(
            batch_size=batch_size,
            output_file=output_file,
            checkpoint_every=checkpoint_every,
        )
        saved_file, _ = self.tokenize_and_save(
            labels=labels,
            output_file=output_file,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )
        return saved_file

    def run_sample(
        self,
        sample_size: int,
        output_file: str,
        tokenizer_name: str = "roberta-base",
        max_length: int = 1024,
    ) -> str:
        n = min(sample_size, len(self.data))
        print(f"Running sample mode with first {n} samples...")

        debug_dir = Path("debug_previews")
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            preview_path = debug_dir / f"{self.dataset_name}_sample_{i:04d}.png"
            self._save_debug_preview(self.data[i], preview_path)

        prompt_lines = self.prompt.strip().splitlines()
        print("\nPrompt preview (first 30 lines):")
        for line in prompt_lines[:30]:
            print(line)

        labels: Dict[int, str] = {}
        for i in tqdm(range(n), desc="Dry run labelling", unit="traj"):
            labels[i] = self._generate_single_label(self.data[i])

        first_label = labels[0]
        first_word_count = len(first_label.split())
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        first_token_count = len(tokenizer.encode(first_label, add_special_tokens=True))
        print(f"\nFirst response word count: {first_word_count}")
        print(f"First response token count ({tokenizer_name}): {first_token_count}")

        saved_path, raw_labels = self.tokenize_and_save(
            labels=labels,
            output_file=output_file,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )

        token_lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in raw_labels]
        print(
            "Token length stats: "
            f"min={min(token_lengths)}, mean={np.mean(token_lengths):.1f}, max={max(token_lengths)}"
        )
        return saved_path


def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI GPT-5.2 physics-flow labelling pipeline")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset key from DATASET_CONFIGS in base_prompt.py",
    )
    parser.add_argument(
        "--input",
        "--input-file",
        dest="input_file",
        required=True,
        help="Path to input NPZ file with key 'field' or 'trajectory' and shape (N, H, W)",
    )
    parser.add_argument(
        "--output",
        "--output-file",
        dest="output_file",
        required=True,
        help="Path to save output NPZ with keys: field, label",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Sequential batch chunk size")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max_length")
    parser.add_argument("--tokenizer-name", default="roberta-base", help="HuggingFace tokenizer name")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Flush checkpoint every N newly generated labels (default: 1).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If > 0, run sample mode on first N items and save previews/output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    labeler = LabelData(dataset_name=args.dataset, file_path=args.input_file)
    if args.sample > 0:
        output_file = labeler.run_sample(
            sample_size=args.sample,
            output_file=args.output_file,
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
        )
    else:
        output_file = labeler.label(
            output_file=args.output_file,
            batch_size=args.batch_size,
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
            checkpoint_every=args.checkpoint_every,
        )
    print(f"\nDone. Output saved at: {output_file}")
