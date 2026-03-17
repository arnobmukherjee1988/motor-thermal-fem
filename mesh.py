"""
mesh.py
=======
Mesh generation for both Phase 1 (annulus) and Phase 2 (motor).

No classes.  Every function returns plain numpy arrays.

Functions
---------
make_annulus_mesh(r_inner, r_outer, mesh_size)
    Returns: nodes, elements, element_tags,
             inner_bc_nodes, outer_bc_nodes, outer_bc_edges

make_motor_mesh(r_rotor, r_airgap, r_slot_in, r_slot_out,
                r_stator, n_slots, mesh_size_iron,
                mesh_size_slot, mesh_size_gap)
    Returns: same six arrays

Array descriptions
------------------
nodes           : shape (N, 2)    -- x-y coordinates of every mesh node
elements        : shape (Ne, 3)   -- three node indices per triangle (0-based)
element_tags    : shape (Ne,)     -- integer material tag per triangle
inner_bc_nodes  : shape (Ni,)     -- node indices on the inner circle
outer_bc_nodes  : shape (No,)     -- node indices on the outer circle
outer_bc_edges  : shape (M, 2)    -- pairs of adjacent outer-boundary nodes
"""

import numpy as np


# =============================================================================
# PHASE 1  --  Structured polar mesh for the plain annulus
# =============================================================================

def make_annulus_mesh(r_inner=0.04, r_outer=0.10, mesh_size=0.005):
    """
    Build a structured triangular mesh of the annular ring
        r_inner  <=  r  <=  r_outer

    Strategy
    --------
    Place nodes on (N_r + 1) concentric rings.
    Each ring has N_theta equally-spaced angular points.
    Each rectangular quad cell is split into two triangles.

    This produces a very regular, high-quality mesh -- ideal for
    convergence studies because refining is just reducing mesh_size.

    Parameters
    ----------
    r_inner   : inner radius [m]
    r_outer   : outer radius [m]
    mesh_size : approximate side length of each triangle [m]

    Returns
    -------
    nodes, elements, element_tags,
    inner_bc_nodes, outer_bc_nodes, outer_bc_edges
    """

    # ── How many rings and angular divisions? ─────────────────────────────────
    radial_span  = r_outer - r_inner
    mid_radius   = 0.5 * (r_inner + r_outer)
    circumference = 2.0 * np.pi * mid_radius

    N_r     = max(4, int(round(radial_span  / mesh_size)))
    N_theta = max(8, int(round(circumference / mesh_size)))

    # Make N_theta even so the mesh is symmetric
    if N_theta % 2 != 0:
        N_theta = N_theta + 1

    # Arrays of radii and angles
    radii  = np.linspace(r_inner, r_outer, N_r + 1)   # N_r+1 values
    thetas = np.linspace(0.0, 2.0 * np.pi, N_theta, endpoint=False)

    # ── Build node coordinate arrays ──────────────────────────────────────────
    # Total nodes = (N_r + 1) * N_theta
    # Nodes are stored ring by ring:
    #   ring 0 (inner): indices 0 .. N_theta-1
    #   ring 1        : indices N_theta .. 2*N_theta-1
    #   ...
    #   ring N_r (outer): indices N_r*N_theta .. (N_r+1)*N_theta-1

    x_list = []
    y_list = []
    for r in radii:
        for theta in thetas:
            x_list.append(r * np.cos(theta))
            y_list.append(r * np.sin(theta))

    nodes = np.zeros((len(x_list), 2))
    for i in range(len(x_list)):
        nodes[i, 0] = x_list[i]
        nodes[i, 1] = y_list[i]

    # ── Build element connectivity ─────────────────────────────────────────────
    # For each quad cell (ring ir, angle slot it), we form two triangles.
    # The four corners of the quad are:
    #   n00 = node at (ring ir,   angle it)
    #   n10 = node at (ring ir+1, angle it)
    #   n01 = node at (ring ir,   angle it+1)   -- wraps around
    #   n11 = node at (ring ir+1, angle it+1)   -- wraps around

    element_list = []
    for ir in range(N_r):
        for it in range(N_theta):
            # Compute the four corner node indices
            n00 = ir * N_theta + (it % N_theta)
            n10 = (ir + 1) * N_theta + (it % N_theta)
            n01 = ir * N_theta + ((it + 1) % N_theta)
            n11 = (ir + 1) * N_theta + ((it + 1) % N_theta)

            # Triangle 1: n00 -> n10 -> n11  (counter-clockwise)
            element_list.append([n00, n10, n11])
            # Triangle 2: n00 -> n11 -> n01  (counter-clockwise)
            element_list.append([n00, n11, n01])

    elements     = np.array(element_list, dtype=np.int32)
    element_tags = np.ones(len(elements), dtype=np.int32)   # all tag 1

    # ── Boundary node index arrays ────────────────────────────────────────────
    # Inner boundary = ring 0
    inner_bc_nodes = np.zeros(N_theta, dtype=np.int32)
    for it in range(N_theta):
        inner_bc_nodes[it] = it   # ring 0 * N_theta + it

    # Outer boundary = ring N_r
    outer_bc_nodes = np.zeros(N_theta, dtype=np.int32)
    for it in range(N_theta):
        outer_bc_nodes[it] = N_r * N_theta + it

    # ── Outer boundary edge pairs ─────────────────────────────────────────────
    # Consecutive nodes on the outer ring form boundary edges
    outer_bc_edges = np.zeros((N_theta, 2), dtype=np.int32)
    for it in range(N_theta):
        outer_bc_edges[it, 0] = N_r * N_theta + it
        outer_bc_edges[it, 1] = N_r * N_theta + ((it + 1) % N_theta)

    return nodes, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges


# =============================================================================
# Boundary utilities
# =============================================================================

def make_closed_boundary_edges(boundary_nodes):
    """
    Build edges for an ordered closed-loop node sequence.
    Example: [n0, n1, n2] -> [[n0,n1], [n1,n2], [n2,n0]]
    """
    n = len(boundary_nodes)
    edges = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        edges[i, 0] = int(boundary_nodes[i])
        edges[i, 1] = int(boundary_nodes[(i + 1) % n])
    return edges


# =============================================================================
# PHASE 2  --  Interface-conforming polar mesh for the slotted motor section
# =============================================================================

def make_motor_mesh(
    r_rotor         = 0.030,
    r_airgap        = 0.035,
    r_slot_in       = 0.038,
    r_slot_out      = 0.070,
    r_stator        = 0.080,
    n_slots         = 12,
    slot_angle_deg  = 14.0,
    mesh_size_iron  = 0.004,
    mesh_size_slot  = 0.002,
    mesh_size_gap   = 0.001,
):
    """
    Build a triangular mesh for the slotted motor cross-section.
    This version uses a structured polar grid so material interfaces
    are explicitly aligned with element edges.

    Returns
    -------
    nodes, elements, element_tags,
    inner_bc_nodes, outer_bc_nodes, outer_bc_edges
    """

    slot_half_angle = np.radians(slot_angle_deg / 2.0)
    slot_pitch = 2.0 * np.pi / n_slots
    slot_angle_tol = np.radians(0.2)

    def angle_in_slot(theta):
        for s in range(n_slots):
            center = s * slot_pitch
            diff = ((theta - center + np.pi) % (2.0 * np.pi)) - np.pi
            if abs(diff) <= slot_half_angle + slot_angle_tol:
                return True
        return False

    # Build angular breakpoints so every slot side is an explicit mesh line.
    breakpoints = [0.0]
    for s in range(n_slots):
        center = s * slot_pitch
        left = (center - slot_half_angle) % (2.0 * np.pi)
        right = (center + slot_half_angle) % (2.0 * np.pi)
        breakpoints.append(left)
        breakpoints.append(right)

    breakpoints = np.array(sorted(set(float(round(v, 12)) for v in breakpoints)))
    if breakpoints[0] != 0.0:
        breakpoints = np.concatenate([[0.0], breakpoints])

    thetas = []
    for i in range(len(breakpoints)):
        t0 = breakpoints[i]
        t1 = breakpoints[(i + 1) % len(breakpoints)]
        if t1 <= t0:
            t1 += 2.0 * np.pi
        tm = 0.5 * (t0 + t1)
        in_slot_sector = angle_in_slot(tm % (2.0 * np.pi))
        target_h = mesh_size_slot if in_slot_sector else mesh_size_iron
        mean_r = 0.5 * (r_airgap + r_stator)
        n_seg = max(2, int(np.ceil(mean_r * (t1 - t0) / target_h)))
        local = np.linspace(t0, t1, n_seg, endpoint=False)
        for t in local:
            thetas.append(t % (2.0 * np.pi))

    thetas = np.array(sorted(set(float(round(v, 12)) for v in thetas)))
    n_theta = len(thetas)

    # Radial bands aligned with all region boundaries.
    radial_bands = [
        (r_rotor, r_airgap, mesh_size_gap),
        (r_airgap, r_slot_in, mesh_size_iron),
        (r_slot_in, r_slot_out, min(mesh_size_iron, mesh_size_slot)),
        (r_slot_out, r_stator, mesh_size_iron),
    ]

    radii = [r_rotor]
    for r0, r1, h in radial_bands:
        n_seg = max(2, int(np.ceil((r1 - r0) / h)))
        segment = np.linspace(r0, r1, n_seg + 1)
        for rv in segment[1:]:
            radii.append(float(rv))
    radii = np.array(radii)

    # Node coordinates, ring by ring.
    nodes = np.zeros((len(radii) * n_theta, 2), dtype=np.float64)
    for ir, r in enumerate(radii):
        for it, theta in enumerate(thetas):
            idx = ir * n_theta + it
            nodes[idx, 0] = r * np.cos(theta)
            nodes[idx, 1] = r * np.sin(theta)

    # Triangulate each polar quad with consistent orientation.
    elements_list = []
    for ir in range(len(radii) - 1):
        for it in range(n_theta):
            n00 = ir * n_theta + it
            n01 = ir * n_theta + ((it + 1) % n_theta)
            n10 = (ir + 1) * n_theta + it
            n11 = (ir + 1) * n_theta + ((it + 1) % n_theta)
            elements_list.append([n00, n10, n11])
            elements_list.append([n00, n11, n01])
    elements = np.array(elements_list, dtype=np.int32)

    # Material tags from element centroids. Interfaces are mesh-aligned now.
    element_tags = np.ones(len(elements), dtype=np.int32)
    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])
        cx = (nodes[g0, 0] + nodes[g1, 0] + nodes[g2, 0]) / 3.0
        cy = (nodes[g0, 1] + nodes[g1, 1] + nodes[g2, 1]) / 3.0
        r_e = np.sqrt(cx * cx + cy * cy)
        theta_e = np.arctan2(cy, cx) % (2.0 * np.pi)

        if r_e < r_airgap:
            element_tags[e] = 3
        elif r_slot_in <= r_e <= r_slot_out and angle_in_slot(theta_e):
            element_tags[e] = 2
        else:
            element_tags[e] = 1

    # Ordered boundary nodes and edge pairs.
    inner_bc_nodes = np.arange(0, n_theta, dtype=np.int32)
    outer_start = (len(radii) - 1) * n_theta
    outer_bc_nodes = np.arange(outer_start, outer_start + n_theta, dtype=np.int32)
    outer_bc_edges = make_closed_boundary_edges(outer_bc_nodes)

    return nodes, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges
