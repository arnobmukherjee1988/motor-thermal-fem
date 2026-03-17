"""
phase2.py
=========
Phase 2: thermal FEM analysis of a slotted motor cross-section.

Upgrades in this prototype version:
1) Interface-conforming mesh
2) Effective anisotropic conductivity in stator iron
3) Temperature-dependent copper loss (fixed-point nonlinear solve)
4) Multi-surface convective cooling (outer + inner airgap)
5) Sensitivity plots for q_winding, h_outer, and k_slot_eff
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Keep plotting/cache robust in restricted environments.
CACHE_DIR = os.path.join(tempfile.gettempdir(), "motor_fem_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
})
import matplotlib.pyplot as plt

from mesh import make_motor_mesh, make_closed_boundary_edges
from fem import assemble, apply_robin, solve
from postprocessing import (
    plot_mesh,
    plot_temperature,
    plot_heat_flux,
    compute_heat_flux,
    smooth_flux_to_nodes,
    check_energy_balance_multi,
    compute_boundary_conductive_flux,
)


# =============================================================================
# Problem parameters
# =============================================================================

# Geometry [m]
R_ROTOR = 0.030
R_AIRGAP = 0.035
R_SLOT_IN = 0.038
R_SLOT_OUT = 0.070
R_STATOR = 0.080
N_SLOTS = 12

# Mesh sizes [m]
H_IRON = 0.003
H_SLOT = 0.0015
H_GAP = 0.001

# Effective material model
K_IRON_RADIAL = 22.0       # W/(m.K)
K_IRON_TANGENTIAL = 30.0   # W/(m.K)
K_SLOT_EFF = 5.0           # W/(m.K), winding+insulation effective
K_AIR_EFF = 0.06           # W/(m.K), effective for narrow air-gap model

# Cooling model (Robin BC)
H_OUTER = 180.0            # W/(m2.K)
T_AMBIENT_OUTER = 20.0     # degC
H_GAP = 120.0              # W/(m2.K)
T_ROTOR_COOLANT = 60.0     # degC

# Copper loss model
Q_WINDING_REF = 1.8e6      # W/m3
T_REF = 120.0              # degC
ALPHA_CU = 0.0039          # 1/K, copper resistivity slope
USE_TEMP_DEP_COPPER = True
USE_TEMP_DEP_IN_SWEEPS = False
NONLIN_MAX_ITERS = 20
NONLIN_UNDERRELAX = 0.25
NONLIN_T_TOL = 1.00        # degC
NONLIN_Q_REL_TOL = 2e-2
LOSS_FACTOR_MIN = 0.20
LOSS_FACTOR_MAX = 2.00

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _build_conductivity_tensors(nodes, elements, element_tags,
                                k_iron_r, k_iron_t, k_slot_eff, k_air_eff):
    """
    Build per-element conductivity tensors K_e (2x2).
    Tag convention:
      1 = iron (anisotropic)
      2 = slot/winding effective medium (isotropic)
      3 = airgap (isotropic)
    """
    Ne = len(elements)
    k_tensors = np.zeros((Ne, 2, 2), dtype=np.float64)

    for e in range(Ne):
        tag = int(element_tags[e])
        n0 = int(elements[e, 0])
        n1 = int(elements[e, 1])
        n2 = int(elements[e, 2])
        cx = (nodes[n0, 0] + nodes[n1, 0] + nodes[n2, 0]) / 3.0
        cy = (nodes[n0, 1] + nodes[n1, 1] + nodes[n2, 1]) / 3.0

        if tag == 1:
            # Polar orthotropy transformed to Cartesian coordinates.
            theta = np.arctan2(cy, cx)
            c = np.cos(theta)
            s = np.sin(theta)
            kxx = k_iron_r * c * c + k_iron_t * s * s
            kyy = k_iron_r * s * s + k_iron_t * c * c
            kxy = (k_iron_r - k_iron_t) * c * s
            k_tensors[e, 0, 0] = kxx
            k_tensors[e, 0, 1] = kxy
            k_tensors[e, 1, 0] = kxy
            k_tensors[e, 1, 1] = kyy
        elif tag == 2:
            k_tensors[e, 0, 0] = k_slot_eff
            k_tensors[e, 1, 1] = k_slot_eff
        else:
            k_tensors[e, 0, 0] = k_air_eff
            k_tensors[e, 1, 1] = k_air_eff

    return k_tensors


def _collect_copper_nodes(elements, element_tags):
    copper_nodes = set()
    for e in range(len(elements)):
        if int(element_tags[e]) == 2:
            copper_nodes.add(int(elements[e, 0]))
            copper_nodes.add(int(elements[e, 1]))
            copper_nodes.add(int(elements[e, 2]))
    return np.array(sorted(copper_nodes), dtype=np.int32)


def solve_motor_case(
    nodes,
    elements,
    element_tags,
    outer_bc_edges,
    inner_bc_edges,
    q_winding_ref,
    h_outer,
    h_gap,
    k_slot_eff,
    use_temp_dep=True,
):
    """
    Solve one motor case (possibly nonlinear due to q_copper(T)).
    Returns: T_h, q_per_element, k_tensors, n_iters
    """
    Ne = len(elements)
    N = len(nodes)
    copper_mask = (element_tags == 2)

    k_tensors = _build_conductivity_tensors(
        nodes=nodes,
        elements=elements,
        element_tags=element_tags,
        k_iron_r=K_IRON_RADIAL,
        k_iron_t=K_IRON_TANGENTIAL,
        k_slot_eff=k_slot_eff,
        k_air_eff=K_AIR_EFF,
    )

    q_per_element = np.zeros(Ne, dtype=np.float64)
    q_per_element[copper_mask] = q_winding_ref

    T_prev = np.full(N, T_AMBIENT_OUTER, dtype=np.float64)
    T_h = T_prev.copy()

    n_iters = 0
    converged = False
    for it in range(NONLIN_MAX_ITERS):
        n_iters = it + 1

        K, f = assemble(nodes, elements, k_tensors, q_per_element)
        K, f = apply_robin(K, f, outer_bc_edges, nodes, h_outer, T_AMBIENT_OUTER)
        K, f = apply_robin(K, f, inner_bc_edges, nodes, h_gap, T_ROTOR_COOLANT)
        T_h = solve(K, f)

        max_dT = float(np.max(np.abs(T_h - T_prev)))
        T_prev = T_h.copy()

        if (not use_temp_dep) or (not np.any(copper_mask)):
            break

        q_target = np.array(q_per_element, copy=True)
        copper_indices = np.where(copper_mask)[0]
        for e in copper_indices:
            n0 = int(elements[e, 0])
            n1 = int(elements[e, 1])
            n2 = int(elements[e, 2])
            T_e = (T_h[n0] + T_h[n1] + T_h[n2]) / 3.0
            factor = 1.0 + ALPHA_CU * (T_e - T_REF)
            factor = min(LOSS_FACTOR_MAX, max(LOSS_FACTOR_MIN, factor))
            q_target[e] = q_winding_ref * factor

        old_q = q_per_element[copper_mask]
        new_q = (1.0 - NONLIN_UNDERRELAX) * old_q + NONLIN_UNDERRELAX * q_target[copper_mask]
        rel_dq = float(np.max(np.abs(new_q - old_q) / np.maximum(np.abs(old_q), 1.0)))
        q_per_element[copper_mask] = new_q

        if max_dT < NONLIN_T_TOL and rel_dq < NONLIN_Q_REL_TOL:
            converged = True
            break

    # Keep returned T consistent with returned q_per_element.
    if use_temp_dep and np.any(copper_mask):
        K, f = assemble(nodes, elements, k_tensors, q_per_element)
        K, f = apply_robin(K, f, outer_bc_edges, nodes, h_outer, T_AMBIENT_OUTER)
        K, f = apply_robin(K, f, inner_bc_edges, nodes, h_gap, T_ROTOR_COOLANT)
        T_h = solve(K, f)

    if use_temp_dep and (not converged):
        print("    [warn] Nonlinear solve reached max iterations; returning clipped fixed-point state.")

    return T_h, q_per_element, k_tensors, n_iters


def _peak_winding_temperature(T_h, copper_node_indices):
    if len(copper_node_indices) == 0:
        return float("nan")
    return float(np.max(T_h[copper_node_indices]))


def _save_curve_plot(x, y, xlabel, ylabel, title, save_path, limit_line=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, "o-", linewidth=2, markersize=6, color="#d62728")
    if limit_line is not None:
        ax.axhline(limit_line, linestyle="--", linewidth=1.5, color="gray")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


print("=" * 60)
print("PHASE 2 -- Slotted Motor Cross-Section (Improved Prototype)")
print("  Nonlinear copper loss: {}".format("ON" if USE_TEMP_DEP_COPPER else "OFF"))
print("  Nonlinear copper in sweeps: {}".format(
    "ON" if USE_TEMP_DEP_IN_SWEEPS else "OFF (faster)"))
print("  Iron anisotropy: k_r = {:.1f}, k_t = {:.1f} W/(m.K)".format(
    K_IRON_RADIAL, K_IRON_TANGENTIAL))
print("  Slot effective k: {:.2f} W/(m.K)".format(K_SLOT_EFF))
print("=" * 60)

print("\n[1] Generating interface-conforming motor mesh ...")
nodes, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges = make_motor_mesh(
    r_rotor=R_ROTOR,
    r_airgap=R_AIRGAP,
    r_slot_in=R_SLOT_IN,
    r_slot_out=R_SLOT_OUT,
    r_stator=R_STATOR,
    n_slots=N_SLOTS,
    mesh_size_iron=H_IRON,
    mesh_size_slot=H_SLOT,
    mesh_size_gap=H_GAP,
)
inner_bc_edges = make_closed_boundary_edges(inner_bc_nodes)

N = len(nodes)
Ne = len(elements)
print("    Nodes: {}".format(N))
print("    Elements: {}".format(Ne))
print("    Outer BC edges: {}".format(len(outer_bc_edges)))
print("    Inner BC edges: {}".format(len(inner_bc_edges)))

n_iron = int(np.sum(element_tags == 1))
n_slot = int(np.sum(element_tags == 2))
n_air = int(np.sum(element_tags == 3))
print("    Element tags -> iron: {}, slot: {}, air: {}".format(n_iron, n_slot, n_air))

print("\n[2] Solving baseline case ...")
T_h, q_arr, k_tensors, n_iters = solve_motor_case(
    nodes=nodes,
    elements=elements,
    element_tags=element_tags,
    outer_bc_edges=outer_bc_edges,
    inner_bc_edges=inner_bc_edges,
    q_winding_ref=Q_WINDING_REF,
    h_outer=H_OUTER,
    h_gap=H_GAP,
    k_slot_eff=K_SLOT_EFF,
    use_temp_dep=USE_TEMP_DEP_COPPER,
)
print("    Nonlinear iterations: {}".format(n_iters))
print("    T range: [{:.2f}, {:.2f}] degC".format(T_h.min(), T_h.max()))

copper_nodes = _collect_copper_nodes(elements, element_tags)
T_winding_max = _peak_winding_temperature(T_h, copper_nodes)
print("    Peak winding temperature: {:.2f} degC".format(T_winding_max))

print("\n[3] Energy balance checks ...")
robin_sets = [
    {"name": "outer", "edges": outer_bc_edges, "h": H_OUTER, "T_inf": T_AMBIENT_OUTER},
    {"name": "inner_gap", "edges": inner_bc_edges, "h": H_GAP, "T_inf": T_ROTOR_COOLANT},
]
Q_gen, Q_conv_total, rel_err_robin, conv_parts = check_energy_balance_multi(
    nodes=nodes,
    elements=elements,
    T_nodal=T_h,
    q_per_element=q_arr,
    robin_sets=robin_sets,
)
print("    Q_generated            = {:.4f} W/m".format(Q_gen))
print("    Q_conv_total (Robin)   = {:.4f} W/m".format(Q_conv_total))
print("      outer contribution   = {:.4f} W/m".format(conv_parts["outer"]))
print("      inner contribution   = {:.4f} W/m".format(conv_parts["inner_gap"]))
print("    Robin-balance error    = {:.4f}%".format(100.0 * rel_err_robin))

flux_x_elem, flux_y_elem = compute_heat_flux(nodes, elements, T_h, k_tensors)
nodal_fx, nodal_fy = smooth_flux_to_nodes(
    nodes, elements, flux_x_elem, flux_y_elem, k_tensors
)

Q_cond_outer = compute_boundary_conductive_flux(
    nodes, outer_bc_edges, nodal_fx, nodal_fy, normal_sign=+1.0
)
Q_cond_inner = compute_boundary_conductive_flux(
    nodes, inner_bc_edges, nodal_fx, nodal_fy, normal_sign=-1.0
)
Q_cond_total = Q_cond_outer + Q_cond_inner
rel_err_cond = abs(Q_gen - Q_cond_total) / max(abs(Q_gen), 1e-30)
print("    Q_cond_total (indep.)  = {:.4f} W/m".format(Q_cond_total))
print("      outer conductive     = {:.4f} W/m".format(Q_cond_outer))
print("      inner conductive     = {:.4f} W/m".format(Q_cond_inner))
print("    Conductive-balance err = {:.4f}%".format(100.0 * rel_err_cond))

print("\n[4] Saving baseline plots ...")
plot_mesh(
    nodes, elements, element_tags,
    title="Phase 2 -- Motor Mesh (interface-conforming)",
    save_path=os.path.join(OUT_DIR, "phase2_mesh.pdf"),
)
plot_temperature(
    nodes, elements, T_h,
    title="Phase 2 -- Temperature Field (improved model)",
    save_path=os.path.join(OUT_DIR, "phase2_temperature.pdf"),
    element_tags=element_tags,
)
plot_heat_flux(
    nodes, elements, T_h, nodal_fx, nodal_fy,
    title="Phase 2 -- Heat Flux Magnitude + Direction",
    save_path=os.path.join(OUT_DIR, "phase2_heatflux.pdf"),
    element_tags=element_tags,
)
print("    Saved: phase2_mesh.pdf, phase2_temperature.pdf, phase2_heatflux.pdf")

print("\n[5] Sensitivity study: T_peak vs winding heat load ...")
q_values = np.array([1.0e6, 2.5e6, 5.0e6, 7.5e6, 1.0e7], dtype=np.float64)
T_peak_q = []
for qv in q_values:
    T_case, _, _, _ = solve_motor_case(
        nodes=nodes,
        elements=elements,
        element_tags=element_tags,
        outer_bc_edges=outer_bc_edges,
        inner_bc_edges=inner_bc_edges,
        q_winding_ref=float(qv),
        h_outer=H_OUTER,
        h_gap=H_GAP,
        k_slot_eff=K_SLOT_EFF,
        use_temp_dep=USE_TEMP_DEP_IN_SWEEPS,
    )
    Tpk = _peak_winding_temperature(T_case, copper_nodes)
    T_peak_q.append(Tpk)
    print("    q = {:.1e} W/m3 -> T_peak = {:.1f} degC".format(qv, Tpk))

_save_curve_plot(
    x=q_values / 1.0e6,
    y=T_peak_q,
    xlabel="Winding heat load q_ref [MW/m3]",
    ylabel="Peak winding temperature [degC]",
    title="Phase 2 -- T_peak vs q_ref",
    save_path=os.path.join(OUT_DIR, "phase2_parametric.pdf"),
    limit_line=180.0,
)
print("    Saved: phase2_parametric.pdf")

print("\n[6] Sensitivity study: T_peak vs outer convection ...")
h_values = np.array([60.0, 100.0, 160.0, 260.0, 400.0], dtype=np.float64)
T_peak_h = []
for hv in h_values:
    T_case, _, _, _ = solve_motor_case(
        nodes=nodes,
        elements=elements,
        element_tags=element_tags,
        outer_bc_edges=outer_bc_edges,
        inner_bc_edges=inner_bc_edges,
        q_winding_ref=Q_WINDING_REF,
        h_outer=float(hv),
        h_gap=H_GAP,
        k_slot_eff=K_SLOT_EFF,
        use_temp_dep=USE_TEMP_DEP_IN_SWEEPS,
    )
    Tpk = _peak_winding_temperature(T_case, copper_nodes)
    T_peak_h.append(Tpk)
    print("    h_outer = {:6.1f} W/m2.K -> T_peak = {:.1f} degC".format(hv, Tpk))

_save_curve_plot(
    x=h_values,
    y=T_peak_h,
    xlabel="Outer convection coefficient h_outer [W/(m2.K)]",
    ylabel="Peak winding temperature [degC]",
    title="Phase 2 -- T_peak vs h_outer",
    save_path=os.path.join(OUT_DIR, "phase2_parametric_houter.pdf"),
    limit_line=180.0,
)
print("    Saved: phase2_parametric_houter.pdf")

print("\n[7] Sensitivity study: T_peak vs slot effective conductivity ...")
k_slot_values = np.array([1.0, 2.0, 3.0, 6.0, 12.0], dtype=np.float64)
T_peak_k = []
for kv in k_slot_values:
    T_case, _, _, _ = solve_motor_case(
        nodes=nodes,
        elements=elements,
        element_tags=element_tags,
        outer_bc_edges=outer_bc_edges,
        inner_bc_edges=inner_bc_edges,
        q_winding_ref=Q_WINDING_REF,
        h_outer=H_OUTER,
        h_gap=H_GAP,
        k_slot_eff=float(kv),
        use_temp_dep=USE_TEMP_DEP_IN_SWEEPS,
    )
    Tpk = _peak_winding_temperature(T_case, copper_nodes)
    T_peak_k.append(Tpk)
    print("    k_slot_eff = {:5.1f} W/m.K -> T_peak = {:.1f} degC".format(kv, Tpk))

_save_curve_plot(
    x=k_slot_values,
    y=T_peak_k,
    xlabel="Slot effective conductivity k_slot_eff [W/(m.K)]",
    ylabel="Peak winding temperature [degC]",
    title="Phase 2 -- T_peak vs k_slot_eff",
    save_path=os.path.join(OUT_DIR, "phase2_parametric_kslot.pdf"),
    limit_line=180.0,
)
print("    Saved: phase2_parametric_kslot.pdf")

print("\n" + "=" * 60)
print("Phase 2 complete.")
print("  Peak winding temperature (baseline): {:.1f} degC".format(T_winding_max))
print("  Robin-balance error: {:.3f}%".format(100.0 * rel_err_robin))
print("  Conductive-balance error: {:.3f}%".format(100.0 * rel_err_cond))
print("=" * 60)
