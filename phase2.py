"""
phase2.py
=========
Phase 2: FEM thermal analysis of a slotted electric motor cross-section.

What this script does (step by step)
--------------------------------------
1. Generate a triangular mesh of the motor cross-section
   (stator iron + 12 copper winding slots + air gap).
2. Assign material properties per region.
3. Assemble the global stiffness matrix K and load vector f.
4. Apply Robin (convective cooling) BC on the outer surface.
   (No BC on the inner surface -- it is naturally adiabatic.)
5. Solve the linear system K * T = f.
6. Check the energy balance: heat generated == heat removed by convection.
7. Save plots to the results/ folder.
8. Run a parametric study: peak winding temperature vs heat load.

How to run
----------
    cd fem_simple
    python phase2.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh           import make_motor_mesh
from materials      import PHASE2_TAGS, get_element_properties
from fem            import assemble, apply_robin, solve
from postprocessing import (plot_mesh, plot_temperature, plot_heat_flux,
                             compute_heat_flux, smooth_flux_to_nodes,
                             check_energy_balance)


# =============================================================================
# Problem parameters  --  change these to explore different setups
# =============================================================================

# Motor geometry [m]
R_ROTOR    = 0.030
R_AIRGAP   = 0.035
R_SLOT_IN  = 0.038
R_SLOT_OUT = 0.070
R_STATOR   = 0.080
N_SLOTS    = 12

# Convective cooling on the outer (stator) surface
H_CONV = 100.0    # W/(m2.K)  convection coefficient
T_INF  = 20.0     # degC      coolant / ambient temperature

# Default winding heat load
Q_WINDING = 5.0e6  # W/m3   Joule heating in copper windings

# Mesh sizes
H_IRON = 0.003     # m  element size in iron region
H_SLOT = 0.0015    # m  element size in slot region
H_GAP  = 0.001     # m  element size in air gap

# Output folder
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Step 1 -- Generate motor mesh
# =============================================================================
print("=" * 60)
print("PHASE 2 -- Slotted Motor Cross-Section")
print("  q_winding = {:.2e} W/m3".format(Q_WINDING))
print("  h_conv    = {} W/(m2.K)".format(H_CONV))
print("  T_inf     = {} degC".format(T_INF))
print("=" * 60)

print("\n[1] Generating motor mesh ...")
nodes, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges = \
    make_motor_mesh(
        r_rotor        = R_ROTOR,
        r_airgap       = R_AIRGAP,
        r_slot_in      = R_SLOT_IN,
        r_slot_out     = R_SLOT_OUT,
        r_stator       = R_STATOR,
        n_slots        = N_SLOTS,
        mesh_size_iron = H_IRON,
        mesh_size_slot = H_SLOT,
        mesh_size_gap  = H_GAP,
    )

N  = len(nodes)
Ne = len(elements)
print("    Nodes: {}".format(N))
print("    Elements: {}".format(Ne))
print("    Outer BC edges: {}".format(len(outer_bc_edges)))


# =============================================================================
# Step 2 -- Assign material properties
# =============================================================================
print("\n[2] Assigning material properties ...")
k_arr, q_arr = get_element_properties(element_tags, PHASE2_TAGS)

# Override the copper heat source with the chosen parameter
for i in range(Ne):
    if element_tags[i] == 2:   # tag 2 = copper
        q_arr[i] = Q_WINDING

# Print a summary
n_iron   = 0
n_copper = 0
n_air    = 0
for i in range(Ne):
    if   element_tags[i] == 1:  n_iron   += 1
    elif element_tags[i] == 2:  n_copper += 1
    elif element_tags[i] == 3:  n_air    += 1

print("    Iron elements:   {}  (k = 50 W/m.K,  q = 0)".format(n_iron))
print("    Copper elements: {}  (k = 385 W/m.K, q = {:.1e} W/m3)".format(
    n_copper, Q_WINDING))
print("    Air elements:    {}  (k = 0.026 W/m.K, q = 0)".format(n_air))


# =============================================================================
# Step 3 -- Assemble global system
# =============================================================================
print("\n[3] Assembling K and f ...")
K, f = assemble(nodes, elements, k_arr, q_arr)
print("    Matrix size: {} x {}".format(K.shape[0], K.shape[1]))
print("    Non-zero entries: {}".format(K.nnz))


# =============================================================================
# Step 4 -- Apply Robin BC on the outer boundary
# =============================================================================
print("\n[4] Applying Robin BC (convective cooling on outer surface) ...")
print("    h = {} W/(m2.K),  T_inf = {} degC".format(H_CONV, T_INF))
K, f = apply_robin(K, f, outer_bc_edges, nodes, H_CONV, T_INF)

# Note: the inner boundary is adiabatic (zero heat flux through the rotor gap).
# In FEM, a zero-Neumann BC requires NO action -- it is automatically satisfied
# by the weak form.  So we do nothing for the inner boundary.
print("    Inner boundary: adiabatic (no action needed)")


# =============================================================================
# Step 5 -- Solve
# =============================================================================
print("\n[5] Solving K * T = f ...")
T_h = solve(K, f)
print("    T range: [{:.2f}, {:.2f}] degC".format(T_h.min(), T_h.max()))

# Find the hottest copper winding node
T_copper_max = 0.0
for i in range(N):
    # Check if node i belongs to any copper element
    pass   # (we check via elements below)

copper_node_indices = set()
for e in range(Ne):
    if element_tags[e] == 2:
        copper_node_indices.add(int(elements[e, 0]))
        copper_node_indices.add(int(elements[e, 1]))
        copper_node_indices.add(int(elements[e, 2]))

T_winding_max = 0.0
for n in copper_node_indices:
    if T_h[n] > T_winding_max:
        T_winding_max = T_h[n]

print("    Peak winding temperature: {:.2f} degC".format(T_winding_max))


# =============================================================================
# Step 6 -- Energy balance check
# =============================================================================
print("\n[6] Energy balance check ...")
Q_gen, Q_conv, rel_err = check_energy_balance(
    nodes, elements, T_h, q_arr,
    outer_bc_edges, H_CONV, T_INF
)
print("    Q_generated  = {:.4f} W/m".format(Q_gen))
print("    Q_convection = {:.4f} W/m".format(Q_conv))
print("    Relative error = {:.4f}%   ({})".format(
    rel_err * 100.0,
    "PASS" if rel_err < 0.01 else "FAIL -- check BC implementation"
))


# =============================================================================
# Step 7 -- Save plots
# =============================================================================
print("\n[7] Saving plots to {} ...".format(OUT_DIR))

plot_mesh(
    nodes, elements, element_tags,
    title="Phase 2 -- Motor Mesh (blue=iron, orange=copper, green=air)",
    save_path=os.path.join(OUT_DIR, "phase2_mesh.png"),
)

plot_temperature(
    nodes, elements, T_h,
    title="Phase 2 -- Temperature Field  (q = {:.1e} W/m3)".format(Q_WINDING),
    save_path=os.path.join(OUT_DIR, "phase2_temperature.png"),
)

flux_x, flux_y = compute_heat_flux(nodes, elements, T_h, k_arr)
nodal_fx, nodal_fy = smooth_flux_to_nodes(nodes, elements, flux_x, flux_y, k_arr)

plot_heat_flux(
    nodes, elements, T_h, nodal_fx, nodal_fy,
    title="Phase 2 -- Heat Flux Field",
    save_path=os.path.join(OUT_DIR, "phase2_heatflux.png"),
)

print("    Saved: phase2_mesh.png, phase2_temperature.png, phase2_heatflux.png")


# =============================================================================
# Step 8 -- Parametric study: peak temperature vs heat load
# =============================================================================
print("\n[8] Parametric study: T_peak vs winding heat load ...")

q_values    = [1.0e6, 2.0e6, 3.0e6, 5.0e6, 7.0e6, 1.0e7]
T_peak_list = []

for q_w in q_values:
    # Re-mesh (same geometry, same size)
    nd, el, et, ib, ob, oe = make_motor_mesh(
        r_rotor=R_ROTOR, r_airgap=R_AIRGAP,
        r_slot_in=R_SLOT_IN, r_slot_out=R_SLOT_OUT,
        r_stator=R_STATOR, n_slots=N_SLOTS,
        mesh_size_iron=H_IRON, mesh_size_slot=H_SLOT, mesh_size_gap=H_GAP,
    )
    k_a, q_a = get_element_properties(et, PHASE2_TAGS)
    for i in range(len(et)):
        if et[i] == 2:
            q_a[i] = q_w

    Kp, fp = assemble(nd, el, k_a, q_a)
    Kp, fp = apply_robin(Kp, fp, oe, nd, H_CONV, T_INF)
    T_p = solve(Kp, fp)

    # Peak winding temperature
    T_pk = 0.0
    for e in range(len(el)):
        if et[e] == 2:
            for loc in range(3):
                val = T_p[int(el[e, loc])]
                if val > T_pk:
                    T_pk = val

    T_peak_list.append(T_pk)
    print("    q = {:.1e} W/m3  ->  T_peak = {:.1f} degC".format(q_w, T_pk))

# Save the parametric chart
fig, ax = plt.subplots(figsize=(7, 5))
q_mw = [q / 1.0e6 for q in q_values]   # convert to MW/m3 for readability
ax.plot(q_mw, T_peak_list, "o-", color="#d62728", linewidth=2, markersize=7)
ax.axhline(180.0, color="gray", linestyle="--", linewidth=1.5,
           label="Insulation limit 180 degC")
ax.set_xlabel("Winding heat load  q  [MW/m3]")
ax.set_ylabel("Peak winding temperature  [degC]")
ax.set_title("Phase 2 -- Parametric Study: T_peak vs Heat Load")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "phase2_parametric.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

print("    Saved: phase2_parametric.png")

print("\n" + "=" * 60)
print("Phase 2 complete.")
print("  Energy balance error: {:.3f}%".format(rel_err * 100.0))
print("  Peak winding temperature: {:.1f} degC".format(T_winding_max))
print("=" * 60)
