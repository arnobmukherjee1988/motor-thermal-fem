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
from scipy.spatial import Delaunay


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
# PHASE 2  --  Delaunay mesh for the slotted motor cross-section
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

    Geometry (rings from inside out)
    ---------------------------------
    r < r_rotor          -- hollow (not meshed, inner BC)
    r_rotor  to r_airgap -- air gap ring (tag 3)
    r_airgap to r_stator -- stator iron (tag 1), with 12 copper slots (tag 2)

    Strategy
    --------
    1. Scatter a cloud of points inside each material zone separately.
    2. Combine all points and run scipy Delaunay triangulation.
    3. Throw away triangles outside the domain.
    4. Assign material tags by checking each triangle centroid.

    Returns
    -------
    nodes, elements, element_tags,
    inner_bc_nodes, outer_bc_nodes, outer_bc_edges
    """

    slot_half_angle = np.radians(slot_angle_deg / 2.0)
    slot_pitch      = 2.0 * np.pi / n_slots

    # ── Helper: points on a ring ───────────────────────────────────────────────
    def ring_points(r, n):
        pts = np.zeros((n, 2))
        for i in range(n):
            theta = 2.0 * np.pi * i / n
            pts[i, 0] = r * np.cos(theta)
            pts[i, 1] = r * np.sin(theta)
        return pts

    # ── Helper: fill an annular band with a regular grid of points ─────────────
    def annular_band(r_in, r_out, mesh_sz):
        n_r = max(3, int(round((r_out - r_in) / mesh_sz)))
        n_t = max(8, int(round(np.pi * (r_in + r_out) / mesh_sz)))
        pts = []
        for r in np.linspace(r_in, r_out, n_r + 1):
            for i in range(n_t):
                theta = 2.0 * np.pi * i / n_t
                pts.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(pts)

    # ── Air gap points ─────────────────────────────────────────────────────────
    pts_gap = annular_band(r_rotor, r_airgap, mesh_size_gap)

    # ── Stator iron points (excluding slot regions) ────────────────────────────
    pts_iron_raw = annular_band(r_airgap, r_stator, mesh_size_iron)

    # Remove points that fall inside any copper slot
    keep = []
    for p in range(len(pts_iron_raw)):
        x = pts_iron_raw[p, 0]
        y = pts_iron_raw[p, 1]
        r_p = np.sqrt(x*x + y*y)

        # Only check points in the radial range of the slots
        if r_p < r_slot_in or r_p > r_slot_out:
            keep.append(True)
            continue

        in_any_slot = False
        ang_p = np.arctan2(y, x) % (2.0 * np.pi)

        for s in range(n_slots):
            ang_center = (s * slot_pitch) % (2.0 * np.pi)
            # Angular difference (shortest arc)
            diff = ang_p - ang_center
            diff = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
            if abs(diff) < slot_half_angle + 0.01:
                in_any_slot = True
                break

        keep.append(not in_any_slot)

    pts_iron = pts_iron_raw[keep]

    # ── Copper slot points ─────────────────────────────────────────────────────
    pts_slots = []
    for s in range(n_slots):
        ang_center = s * slot_pitch
        ang_lo = ang_center - slot_half_angle
        ang_hi = ang_center + slot_half_angle

        n_r = max(3, int(round((r_slot_out - r_slot_in) / mesh_size_slot)))
        arc_len = (r_slot_in + r_slot_out) / 2.0 * slot_angle_deg * np.pi / 180.0
        n_t = max(3, int(round(arc_len / mesh_size_slot)))

        for r in np.linspace(r_slot_in, r_slot_out, n_r + 1):
            for theta in np.linspace(ang_lo, ang_hi, n_t + 1):
                pts_slots.append([r * np.cos(theta), r * np.sin(theta)])

    pts_slots = np.array(pts_slots)

    # ── Combine all points and remove duplicates ───────────────────────────────
    all_pts = np.vstack([pts_gap, pts_iron, pts_slots])
    unique_pts, inv = np.unique(np.round(all_pts, 8), axis=0, return_inverse=True)

    # ── Delaunay triangulation ─────────────────────────────────────────────────
    tri = Delaunay(unique_pts)
    elements_raw = tri.simplices.astype(np.int32)

    # ── Filter: keep only elements inside the domain ───────────────────────────
    # The centroid of each triangle must lie within the annular domain
    keep_elem = []
    for e in range(len(elements_raw)):
        # Centroid = average of the three node positions
        cx = (unique_pts[elements_raw[e, 0], 0] +
              unique_pts[elements_raw[e, 1], 0] +
              unique_pts[elements_raw[e, 2], 0]) / 3.0
        cy = (unique_pts[elements_raw[e, 0], 1] +
              unique_pts[elements_raw[e, 1], 1] +
              unique_pts[elements_raw[e, 2], 1]) / 3.0
        r_c = np.sqrt(cx*cx + cy*cy)
        if r_c >= r_rotor - 1e-6 and r_c <= r_stator + 1e-6:
            keep_elem.append(e)

    elements = elements_raw[keep_elem].astype(np.int32)

    # ── Assign material tags ───────────────────────────────────────────────────
    # Default tag = 1 (iron).  Override for air gap and copper slots.
    element_tags = np.ones(len(elements), dtype=np.int32)

    for e in range(len(elements)):
        cx = (unique_pts[elements[e, 0], 0] +
              unique_pts[elements[e, 1], 0] +
              unique_pts[elements[e, 2], 0]) / 3.0
        cy = (unique_pts[elements[e, 0], 1] +
              unique_pts[elements[e, 1], 1] +
              unique_pts[elements[e, 2], 1]) / 3.0
        r_e = np.sqrt(cx*cx + cy*cy)

        # Air gap region
        if r_e < r_airgap:
            element_tags[e] = 3
            continue

        # Check if centroid is inside a copper slot
        if r_slot_in <= r_e <= r_slot_out:
            ang_e = np.arctan2(cy, cx) % (2.0 * np.pi)
            for s in range(n_slots):
                ang_center = (s * slot_pitch) % (2.0 * np.pi)
                diff = ang_e - ang_center
                diff = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
                if abs(diff) < slot_half_angle:
                    element_tags[e] = 2
                    break

    # ── Boundary nodes ─────────────────────────────────────────────────────────
    tol = 1.5 * max(mesh_size_iron, mesh_size_gap) / 2.0
    r_nodes = np.sqrt(unique_pts[:, 0]**2 + unique_pts[:, 1]**2)

    inner_list = []
    outer_list = []
    for i in range(len(unique_pts)):
        if abs(r_nodes[i] - r_rotor) < tol:
            inner_list.append(i)
        if abs(r_nodes[i] - r_stator) < tol:
            outer_list.append(i)

    inner_bc_nodes = np.array(inner_list, dtype=np.int32)
    outer_bc_nodes = np.array(outer_list, dtype=np.int32)

    # ── Outer boundary edges ───────────────────────────────────────────────────
    # An edge is a boundary edge if both its nodes are on the outer boundary
    outer_set = set(outer_bc_nodes.tolist())
    edge_list = []
    for e in range(len(elements)):
        # The three edges of triangle e
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            na = int(elements[e, a])
            nb = int(elements[e, b])
            if na in outer_set and nb in outer_set:
                # Store in sorted order to avoid duplicates
                edge_list.append([min(na, nb), max(na, nb)])

    if len(edge_list) > 0:
        # Remove duplicate edges
        edge_arr = np.array(edge_list, dtype=np.int32)
        outer_bc_edges = np.unique(edge_arr, axis=0)
    else:
        outer_bc_edges = np.zeros((0, 2), dtype=np.int32)

    return unique_pts, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges
