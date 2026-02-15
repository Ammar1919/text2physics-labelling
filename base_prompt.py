 

DATASET_CONFIGS = {
    "shear_flow": {
        "flow_config": "Shear Flow",
        "domain": "periodic domain with parallel fluid layers moving at different velocities",
        "quantity": "tracer scalar field",
        "dimension": "2d"
    },
    "rayleigh_benard": {
        "flow_config": "Rayleigh Benard Convection",
        "domain": "horizontal fluid layer heated from below and cooled from above",
        "quantity": "temperature field",
        "dimension": "2d"
    }
}

def get_prompt(flow_config, domain, quantity, dimension):
    return f"""You are given a single flow field from the Shear Flow dataset.

    This is a single static image.
    Describe ONLY what is visually present.
    Do NOT describe temporal behavior.
    Do NOT infer physical causes.
    Do NOT speculate about dynamics.
    Provide a purely visual structural description.

    ------------------------------------------------------------
    SIMULATION CONTEXT  (Ground Truth)
    ------------------------------------------------------------
    Flow configuration: {flow_config}
    Domain: {domain}
    Quantity shown: {quantity}
    Dimensionality: {dimension}

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
    
    END OF OUTPUT FORMAT
    """