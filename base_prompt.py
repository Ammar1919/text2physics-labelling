 # prompts/base_prompt.py

DATASET_CONFIGS = {
    "rayleigh_benard": {
        "flow_config": "Rayleigh–Bénard convection",
        "domain": "horizontal fluid layer heated from below and cooled from above",
        "quantity": "temperature field",
        "dimension": "2D"
    },
    "shear_flow": {
        "flow_config": "Shear Flow",
        "domain": "periodic domain with parallel fluid layers moving at different velocities",
        "quantity": "tracer scalar field",
        "dimension": "2D"
    },
    "turbulent_channel_flow": {
        "flow_config": "Turbulent channel flow",
        "domain": "flow between two flat plates driven by a constant pressure gradient in the streamwise direction",
        "quantity": "velocity field",
        "dimension": "2D"
    },
    "ns2d_smoke": {
        "flow_config": "Two-dimensional incompressible Navier–Stokes flow with buoyancy-driven forcing",
        "domain": "closed square cavity with no-slip wall boundaries",
        "quantity": "scalar smoke density field",
        "dimension": "2D"
    }
}


def get_prompt(dataset_key: str) -> str:
    config = DATASET_CONFIGS[dataset_key]

    return f"""
You are given a single flow field from the dataset.

This is a single static image.
Describe ONLY what is visually present.
Do NOT describe temporal behavior.
Do NOT infer physical causes.
Do NOT speculate about dynamics.
Provide a purely visual structural description.

------------------------------------------------------------
SIMULATION CONTEXT (Ground Truth)
------------------------------------------------------------
Flow configuration: {config["flow_config"]}
Domain: {config["domain"]}
Quantity shown: {config["quantity"]}
Dimensionality: {config["dimension"]}

------------------------------------------------------------
OUTPUT FORMAT (STRICT)
------------------------------------------------------------

Identified dominant spatial structures:

Provide a numbered list.

For each structure, write a short paragraph (3–5 sentences) that clearly and explicitly states:

- The structure type.
- Its relative scale.
- Its approximate spatial location.
- Its geometric form.
- Its edge characteristics.
- Its internal texture.

Each paragraph MUST clearly state these attributes in full sentences.

Only describe structures that are clearly visible and visually dominant.

------------------------------------------------------------
Global spatial summary:

Write a separate paragraph summarizing:

- The overall structural organization.
- Whether the field is ordered or irregular.
- Whether structures are localized or domain-spanning.
- The degree of symmetry or asymmetry.
- The overall spatial complexity.

This summary must remain purely visual.

LENGTH CONSTRAINT (MANDATORY)
- Total output must be under 350 words.
- Do not add any extra sections, headers, or commentary.

END OF OUTPUT FORMAT
"""
