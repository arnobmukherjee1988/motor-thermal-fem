"""
materials.py
============
Material properties for the motor FEM project.

No classes. Everything is stored in plain Python dictionaries.
A dictionary is just a lookup table: you give it a key (like "iron")
and it gives you a value (like the thermal conductivity).

Usage
-----
    k = MATERIALS["iron"]["k"]       # 50.0  W/(m.K)
    q = MATERIALS["copper"]["q_dot"] # 5e6   W/m3
"""

# ── Material property table ───────────────────────────────────────────────────
# Each entry is a dictionary with two keys:
#   "k"     : thermal conductivity [W / (m.K)]
#   "q_dot" : volumetric heat source [W / m3]

MATERIALS = {
    "iron": {
        "name":  "Electrical steel (stator iron)",
        "k":     50.0,        # W/(m.K)  -- laminated silicon steel
        "q_dot": 0.0,         # no heat generation in iron
    },
    "copper": {
        "name":  "Copper winding",
        "k":     385.0,       # W/(m.K)  -- pure copper
        "q_dot": 5.0e6,       # W/m3     -- Joule heating in windings
    },
    "air": {
        "name":  "Air gap",
        "k":     0.026,       # W/(m.K)  -- still air ~50 deg C
        "q_dot": 0.0,
    },
    "homogeneous": {
        "name":  "Homogeneous material (Phase 1 validation)",
        "k":     50.0,        # same as iron
        "q_dot": 0.0,
    },
}


# ── Tag-to-material mappings ──────────────────────────────────────────────────
# The mesh assigns each triangle an integer "tag" to indicate which
# material region it belongs to.  These dictionaries say which tag
# corresponds to which material.
#
# Phase 1 (plain annulus): only one region, tag = 1
PHASE1_TAGS = {
    1: MATERIALS["homogeneous"],
}

# Phase 2 (motor cross-section):
#   tag 1 = stator iron
#   tag 2 = copper winding slots
#   tag 3 = air gap
PHASE2_TAGS = {
    1: MATERIALS["iron"],
    2: MATERIALS["copper"],
    3: MATERIALS["air"],
}


def get_element_properties(element_tags, tag_to_material):
    """
    Build per-element conductivity and heat-source arrays.

    Parameters
    ----------
    element_tags    : 1-D integer array, length Ne (one tag per triangle)
    tag_to_material : dictionary  { tag_integer : material_dictionary }

    Returns
    -------
    k_array : 1-D float array, length Ne   (conductivity per element)
    q_array : 1-D float array, length Ne   (heat source per element)
    """
    import numpy as np

    Ne = len(element_tags)
    k_array = np.zeros(Ne)
    q_array = np.zeros(Ne)

    # Loop over every element and look up its material properties
    for i in range(Ne):
        tag = int(element_tags[i])
        mat = tag_to_material[tag]   # look up the dictionary entry
        k_array[i] = mat["k"]
        q_array[i] = mat["q_dot"]

    return k_array, q_array
