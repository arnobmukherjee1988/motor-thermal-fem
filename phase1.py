"""
phase1.py
=========
Phase 1: FEM thermal analysis of a plain annular ring.

What this script does (step by step)
--------------------------------------
1. Generate a structured triangular mesh of the annulus.
2. Assign material properties (conductivity, heat source) to each element.
3. Assemble the global stiffness matrix K and load vector f.
4. Apply Dirichlet temperature boundary conditions (inner and outer rings).
5. Solve the linear system K * T = f.
6. Compare FEM solution to the exact analytical solution.
7. Run a convergence study (repeat at finer and finer meshes).
8. Save plots to the results/ folder.

How to run
----------
    cd fem_simple
    python phase1.py
"""

import sys
import os
import tempfile
import numpy as np

# Make sure Python can find the other files in this folder
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

from mesh          import make_annulus_mesh
from materials     import PHASE1_TAGS, get_element_properties
from fem           import assemble, apply_dirichlet, solve
from postprocessing import (plot_mesh, plot_temperature, plot_heat_flux,
                             plot_radial, plot_convergence,
                             compute_heat_flux, smooth_flux_to_nodes)
from validation    import (exact_temperature, exact_gradient,
                            evaluate_exact_at_nodes,
                            compute_L2_error, compute_H1_error,
                            run_convergence_study)


# =============================================================================
# Problem parameters  --  change these to explore different setups
# =============================================================================
R_INNER  = 0.04     # m   inner radius of the annulus
R_OUTER  = 0.10     # m   outer radius of the annulus
T_INNER  = 100.0    # degC  prescribed temperature at inner wall
T_OUTER  = 20.0     # degC  prescribed temperature at outer wall
K_IRON   = 50.0     # W/(m.K)  thermal conductivity (iron)
Q_DOT    = 0.0      # W/m3  volumetric heat source (0 = pure conduction)
H_MESH   = 0.004    # m    mesh element size for the single-solve run

# Output folder for plots
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Step 1 -- Generate mesh
# =============================================================================
print("=" * 60)
print("PHASE 1 -- Plain Annulus")
print("  r_inner = {} m,   r_outer = {} m".format(R_INNER, R_OUTER))
print("  T_inner = {} C,   T_outer = {} C".format(T_INNER, T_OUTER))
print("  k = {} W/(m.K),   q_dot = {} W/m3".format(K_IRON, Q_DOT))
print("  mesh_size = {} m".format(H_MESH))
print("=" * 60)

print("\n[1] Generating mesh ...")
nodes, elements, element_tags, inner_bc_nodes, outer_bc_nodes, outer_bc_edges = \
    make_annulus_mesh(R_INNER, R_OUTER, H_MESH)

N  = len(nodes)
Ne = len(elements)
print("    Nodes: {}".format(N))
print("    Elements: {}".format(Ne))
print("    Inner BC nodes: {}".format(len(inner_bc_nodes)))
print("    Outer BC nodes: {}".format(len(outer_bc_nodes)))


# =============================================================================
# Step 2 -- Assign material properties
# =============================================================================
print("\n[2] Assigning material properties ...")
k_arr, q_arr = get_element_properties(element_tags, PHASE1_TAGS)

# Override with our chosen parameters
for i in range(Ne):
    k_arr[i] = K_IRON
    q_arr[i] = Q_DOT

print("    Conductivity  k = {} W/(m.K) for all elements".format(K_IRON))
print("    Heat source  q = {} W/m3    for all elements".format(Q_DOT))


# =============================================================================
# Step 3 -- Assemble global stiffness matrix and load vector
# =============================================================================
print("\n[3] Assembling global system K and f ...")
K, f = assemble(nodes, elements, k_arr, q_arr)
print("    K size: {} x {}".format(K.shape[0], K.shape[1]))
print("    Non-zero entries in K: {}".format(K.nnz))


# =============================================================================
# Step 4 -- Apply Dirichlet boundary conditions
# =============================================================================
print("\n[4] Applying Dirichlet boundary conditions ...")

# Get the exact analytical solution (to prescribe exact values on the boundary,
# which removes BC error from the convergence study)
T_exact_fn = exact_temperature(R_INNER, R_OUTER, T_INNER, T_OUTER, K_IRON, Q_DOT)

# Build a dictionary: node index -> prescribed temperature
T_prescribed = {}
for n in inner_bc_nodes:
    n = int(n)
    r = np.sqrt(nodes[n, 0]**2 + nodes[n, 1]**2)
    T_prescribed[n] = float(T_exact_fn(r))

for n in outer_bc_nodes:
    n = int(n)
    r = np.sqrt(nodes[n, 0]**2 + nodes[n, 1]**2)
    T_prescribed[n] = float(T_exact_fn(r))

# Combine into one array of BC nodes
bc_nodes = np.concatenate([inner_bc_nodes, outer_bc_nodes])
K, f = apply_dirichlet(K, f, bc_nodes, T_prescribed)

print("    Total Dirichlet nodes: {}".format(len(bc_nodes)))


# =============================================================================
# Step 5 -- Solve the linear system
# =============================================================================
print("\n[5] Solving K * T = f ...")
T_h = solve(K, f)
print("    T range: [{:.2f}, {:.2f}] deg C".format(T_h.min(), T_h.max()))


# =============================================================================
# Step 6 -- Validate against analytical solution
# =============================================================================
print("\n[6] Validation against exact solution ...")

T_exact_nodes = evaluate_exact_at_nodes(nodes, T_exact_fn)

# Point-wise max error
max_error = 0.0
for i in range(N):
    err = abs(T_h[i] - T_exact_nodes[i])
    if err > max_error:
        max_error = err

dTdr_fn = exact_gradient(R_INNER, R_OUTER, T_INNER, T_OUTER, K_IRON, Q_DOT)
L2_err = compute_L2_error(nodes, elements, T_h, T_exact_fn)
H1_err = compute_H1_error(nodes, elements, T_h, dTdr_fn)

print("    Max pointwise error = {:.4e} deg C".format(max_error))
print("    L2 error = {:.4e}".format(L2_err))
print("    H1 error = {:.4e}".format(H1_err))


# =============================================================================
# Step 7 -- Post-process: heat flux
# =============================================================================
print("\n[7] Computing heat flux ...")
flux_x, flux_y = compute_heat_flux(nodes, elements, T_h, k_arr)
nodal_fx, nodal_fy = smooth_flux_to_nodes(nodes, elements, flux_x, flux_y, k_arr)


# =============================================================================
# Step 8 -- Save plots
# =============================================================================
print("\n[8] Saving plots to {} ...".format(OUT_DIR))

plot_mesh(
    nodes, elements, element_tags,
    title="Phase 1 -- Annulus Mesh",
    save_path=os.path.join(OUT_DIR, "phase1_mesh.pdf"),
)

plot_temperature(
    nodes, elements, T_h,
    title="Phase 1 -- Temperature Field",
    save_path=os.path.join(OUT_DIR, "phase1_temperature.pdf"),
    element_tags=element_tags,
)

plot_heat_flux(
    nodes, elements, T_h, nodal_fx, nodal_fy,
    title="Phase 1 -- Heat Flux Field",
    save_path=os.path.join(OUT_DIR, "phase1_heatflux.pdf"),
    element_tags=element_tags,
)

plot_radial(
    nodes, T_h, T_exact_fn,
    title="Phase 1 -- Radial Temperature (FEM vs Exact)",
    save_path=os.path.join(OUT_DIR, "phase1_radial.pdf"),
)

print("    Saved: phase1_mesh.pdf, phase1_temperature.pdf,")
print("           phase1_heatflux.pdf, phase1_radial.pdf")


# =============================================================================
# Step 9 -- Convergence study
# =============================================================================
print("\n[9] Running convergence study ...")
print("    (Solves at 4 mesh sizes, may take a minute)")

mesh_sizes = [0.012, 0.008, 0.005, 0.003]

h_list, L2_list, H1_list, L2_rate, H1_rate = run_convergence_study(
    mesh_sizes = mesh_sizes,
    r_inner    = R_INNER,
    r_outer    = R_OUTER,
    T_inner    = T_INNER,
    T_outer    = T_OUTER,
    k          = K_IRON,
    q_dot      = Q_DOT,
    verbose    = True,
)

print("\n    L2 convergence rate = {:.3f}  (expected ~2.0)".format(L2_rate))
print("    H1 convergence rate = {:.3f}  (expected ~1.0)".format(H1_rate))

plot_convergence(
    h_list, L2_list, H1_list,
    title="Phase 1 -- Convergence Study (P1 triangles)",
    save_path=os.path.join(OUT_DIR, "phase1_convergence.pdf"),
)

print("    Saved: phase1_convergence.pdf")

print("\n" + "=" * 60)
print("Phase 1 complete.")
print("  L2 rate = {:.3f}   H1 rate = {:.3f}".format(L2_rate, H1_rate))
print("=" * 60)
