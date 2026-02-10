# Text2Physics Labelling

This repository contains data, preprocessing conventions, and experimental setup for a text-to-physics labelling project focused on fluid dynamics simulations. 
The core objective is to pair physical simulation fields with concise natural-language descriptions, 
enabling downstream tasks such as conditional generative modeling using latent diffusion or flow-matching methods.

## Project Overview

The project works with several canonical fluid dynamics scenarios. 
For each scenario, a single parameter configuration is selected from a larger dataset, and representative temporal snapshots or trajectories are extracted. 
These fields are paired with textual labels that describe the physical setting and observed dynamics.

Current experiments emphasize latent diffusion and flow-matching approaches trained on these labelled field–text pairs.

## Datasets and Scenarios

The following physical scenarios are included:

### Shear Flow

- Original resolution: 256 × 512
- Downsampled resolution: 128 × 256
- Time index: t = 60
- Data type: tracer-field trajectories
- Focus: intermediate-stage dynamics

### Rayleigh–Bénard Convection

- Resolution: 512 × 128
- Time index: t = 60
- Data type: buoyancy-driven convection fields
- Focus: intermediate-stage plume and roll structures

### Smoke Simulation

- Resolution: 128 × 128
- Time index: t = 20
- Data type: smoke density fields

### Turbulent Channel Flow (TCF)

- Resolution: 192 × 192
- Time index: t = 0
- Data type: velocity fields (streamwise component, Vx)

## Data Format

All labelled datasets are stored in .npz files with the following keys:
- field: the physical simulation field (e.g. velocity, tracer, buoyancy, or density)
- label: a natural-language description of the corresponding field

Each (label, field) pair represents one data point. For example, the shear flow dataset contains 1120 such pairs.

## Trajectories and Preprocessing

For trajectory-based scenarios, the pre-prompt data used for text labelling follows the structure: `trajectory, x, y`

Intermediate frames are selected to emphasize informative and visually rich stages of the dynamics, rather than initial transients or fully developed steady states.

## Text Labelling

Textual labels are generated using large language models. 
Prompts are designed to describe the physical configuration, flow behavior, and key observable structures in each field. 
Multiple models are used to introduce diversity and robustness in the generated descriptions.
