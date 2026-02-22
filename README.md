# Text2Physics Labelling

This repository contains the labelling pipeline for pairing 2D fluid simulation fields with structured natural-language descriptions using a vision-language model.

The current implementation focuses on generating high-quality structured captions for physical simulation fields using GPT-5.2 with explicit high-effort reasoning.

## File Structure and Roles

### `label_gpt/labelling.py`

Main labelling script.

Functionality:

NPZ loading -> image rendering -> OpenAI GPT-5.2 call (Thinking=high) -> text tokenization (roberta-base) -> save labelled NPZ.

Key properties:

- API: OpenAI Responses API
- Model: `gpt-5.2`
- Thinking fixed:

```python
reasoning={"effort":"high"}
```

- Output keys: `label`, `field`
- Required arguments: `--dataset --input --output`
- Optional sampling mode: `--sample N`
- Checkpoint resume supported via `*.checkpoint.jsonl`

Example usage:

```bash
python labelling.py --dataset shear_flow --input ..\shear_flow_frame60.npz --output ..\datasets\labeled\shear_flow_labeled_openai.npz
```

Sample mode:

```bash
python labelling.py --dataset shear_flow --input ..\shear_flow_frame60.npz --output ..\datasets\labeled\shear_flow_sample.npz --sample 5
```

### `label_gpt/base_prompt.py`

Prompt templates and dataset-specific configuration.
`labelling.py` retrieves prompts via:

```python
get_prompt(dataset_key)
```

### `label_gpt/utils.py`

General utility functions used by the labelling pipeline.

## Input / Output Format

### Input NPZ

Supported keys:

- `field`
- or `trajectory`

Shape format:

`(N, H, W)`

### Output NPZ

The current implementation saves:

- `field`: original field data
- `label`: token IDs (Roberta tokenizer)

Each `(field, label)` pair corresponds to one labelled data point.

## Current Implementation in this Repository

- API: OpenAI Responses API
- Model: `gpt-5.2`
- Vision input: rendered PNG from each 2D field using `viridis` and `origin="lower"`
- Output control: `max_output_tokens=1024`
- Thinking mode: explicitly enabled with:

```python
reasoning={"effort":"high"}
```

High reasoning effort is used to improve:

- Structural decomposition quality
- Boundary consistency
- Hierarchical spatial interpretation
- Format stability for scientific descriptions

## Environment Setup

```bash
cd C:\Users\majun\Desktop\dataset\label-gpt
conda activate t2p_label
$env:OPENAI_API_KEY="your_key"
```

Install dependencies if needed:

```bash
pip install openai python-dotenv numpy matplotlib tqdm transformers
```

## Full Dataset Runs

### Shear Flow

```bash
python labelling.py --dataset shear_flow --input ..\shear_flow_frame60.npz --output ..\datasets\labeled\shear_flow_labeled_openai.npz
```
