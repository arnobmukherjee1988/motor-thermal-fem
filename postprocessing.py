"""
postprocessing.py
=================
Derive physical quantities from the nodal temperature solution,
and produce all plots.

Functions
---------
compute_heat_flux(nodes, elements, T_nodal, k_per_element)
    -> flux_x, flux_y  (one value per element, piecewise constant)

smooth_flux_to_nodes(nodes, elements, flux_x, flux_y, k_per_element)
    -> nodal_flux_x, nodal_flux_y  (area-weighted average at each node)

check_energy_balance(nodes, elements, T_nodal, q_per_element,
                     outer_bc_edges, h_coeff, T_inf)
    -> Q_gen, Q_conv, relative_error

plot_mesh(nodes, elements, element_tags, title, save_path)
plot_temperature(nodes, elements, T_nodal, title, save_path)
plot_heat_flux(nodes, elements, T_nodal, flux_x_nodal, flux_y_nodal,
               title, save_path)
plot_radial(nodes, T_nodal, T_exact_fn, title, save_path)
plot_convergence(mesh_sizes, L2_errors, H1_errors, title, save_path)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc


# =============================================================================
# Heat flux computation
# =============================================================================

def compute_heat_flux(nodes, elements, T_nodal, k_per_element):
    """
    Compute the heat flux vector  q = -k * grad(T)  for each element.

    For a P1 (linear) triangle, grad(T) is constant inside each element.
    So the heat flux is also constant per element (piecewise constant field).

    The gradient is:
        dT/dx = b1*T1 + b2*T2 + b3*T3
        dT/dy = c1*T1 + c2*T2 + c3*T3

    where b_i, c_i are the same shape-function gradient coefficients
    computed in fem.py.

    Parameters
    ----------
    nodes          : array (N, 2)
    elements       : array (Ne, 3)
    T_nodal        : array (N,)     nodal temperatures
    k_per_element  : array (Ne,)    conductivity per element

    Returns
    -------
    flux_x : array (Ne,)   x-component of heat flux  [W/m2]
    flux_y : array (Ne,)   y-component of heat flux  [W/m2]
    """
    Ne = len(elements)
    flux_x = np.zeros(Ne)
    flux_y = np.zeros(Ne)

    for e in range(Ne):
        # Node indices
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])

        # Coordinates
        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]

        # Area
        two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

        # Shape function gradient coefficients
        b1 = (y2 - y3) / two_area
        b2 = (y3 - y1) / two_area
        b3 = (y1 - y2) / two_area

        c1 = (x3 - x2) / two_area
        c2 = (x1 - x3) / two_area
        c3 = (x2 - x1) / two_area

        # Nodal temperatures of this element
        T1 = T_nodal[g0]
        T2 = T_nodal[g1]
        T3 = T_nodal[g2]

        # grad(T) = B * T_e
        dT_dx = b1*T1 + b2*T2 + b3*T3
        dT_dy = c1*T1 + c2*T2 + c3*T3

        # Heat flux q = -k * grad(T)
        flux_x[e] = -k_per_element[e] * dT_dx
        flux_y[e] = -k_per_element[e] * dT_dy

    return flux_x, flux_y


def smooth_flux_to_nodes(nodes, elements, flux_x, flux_y, k_per_element):
    """
    Smooth the piecewise-constant element heat flux to nodal values
    using area-weighted averaging.

    For each node n:
        q_n = sum over all elements touching n of (area_e * q_e)
              -----------------------------------------------
              sum over all elements touching n of (area_e)

    Parameters
    ----------
    nodes         : array (N, 2)
    elements      : array (Ne, 3)
    flux_x, flux_y: arrays (Ne,)   element heat flux components
    k_per_element : array (Ne,)    (only needed for computing areas)

    Returns
    -------
    nodal_flux_x : array (N,)
    nodal_flux_y : array (N,)
    """
    N  = len(nodes)
    Ne = len(elements)

    sum_qx     = np.zeros(N)
    sum_qy     = np.zeros(N)
    sum_weight = np.zeros(N)

    for e in range(Ne):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])

        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]

        two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(two_area)

        # Add area-weighted contribution to each of the three nodes
        for g in [g0, g1, g2]:
            sum_qx[g]     = sum_qx[g]     + area * flux_x[e]
            sum_qy[g]     = sum_qy[g]     + area * flux_y[e]
            sum_weight[g] = sum_weight[g] + area

    nodal_flux_x = np.zeros(N)
    nodal_flux_y = np.zeros(N)
    for i in range(N):
        if sum_weight[i] > 0.0:
            nodal_flux_x[i] = sum_qx[i] / sum_weight[i]
            nodal_flux_y[i] = sum_qy[i] / sum_weight[i]

    return nodal_flux_x, nodal_flux_y


# =============================================================================
# Energy balance
# =============================================================================

def check_energy_balance(nodes, elements, T_nodal, q_per_element,
                          outer_bc_edges, h_coeff, T_inf):
    """
    Verify that heat generated = heat removed.

    Q_generated  = sum over all elements of (q_dot_e * area_e)
    Q_convection = sum over outer boundary edges of (h * (T_avg - T_inf) * L)

    Parameters
    ----------
    nodes          : array (N, 2)
    elements       : array (Ne, 3)
    T_nodal        : array (N,)
    q_per_element  : array (Ne,)    heat source per element [W/m3]
    outer_bc_edges : array (M, 2)   outer boundary edge node pairs
    h_coeff        : float
    T_inf          : float

    Returns
    -------
    Q_gen   : float   total heat generated [W/m]
    Q_conv  : float   total heat removed   [W/m]
    rel_err : float   |Q_gen - Q_conv| / Q_gen
    """
    # Total heat generated by all elements
    Q_gen = 0.0
    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])
        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]
        two_area = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        area = 0.5 * abs(two_area)
        Q_gen = Q_gen + q_per_element[e] * area

    # Total heat removed by convection on the outer boundary
    Q_conv = 0.0
    for e in range(len(outer_bc_edges)):
        i = int(outer_bc_edges[e, 0])
        j = int(outer_bc_edges[e, 1])
        xi = nodes[i, 0];  yi = nodes[i, 1]
        xj = nodes[j, 0];  yj = nodes[j, 1]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        T_avg = 0.5 * (T_nodal[i] + T_nodal[j])
        Q_conv = Q_conv + h_coeff * (T_avg - T_inf) * L

    if abs(Q_gen) > 1e-30:
        rel_err = abs(Q_gen - Q_conv) / abs(Q_gen)
    else:
        rel_err = 0.0

    return Q_gen, Q_conv, rel_err


# =============================================================================
# Plots
# =============================================================================

def plot_mesh(nodes, elements, element_tags, title, save_path, show=False):
    """Plot mesh coloured by material region tag."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Build triangle patches and colour by tag
    triangles = []
    colours   = []
    cmap_tags = {1: "#4c72b0", 2: "#dd8452", 3: "#55a868"}   # blue/orange/green

    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])
        tri = [nodes[g0], nodes[g1], nodes[g2]]
        triangles.append(tri)
        tag = int(element_tags[e])
        colours.append(cmap_tags.get(tag, "#888888"))

    poly = mc.PolyCollection(triangles, facecolors=colours,
                              edgecolors="none", linewidths=0.1)
    ax.add_collection(poly)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_temperature(nodes, elements, T_nodal, title, save_path, show=False):
    """Plot filled temperature contour map."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Build connectivity for tricontourf
    conn = np.zeros((len(elements), 3), dtype=np.int32)
    for e in range(len(elements)):
        conn[e, 0] = int(elements[e, 0])
        conn[e, 1] = int(elements[e, 1])
        conn[e, 2] = int(elements[e, 2])

    cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], conn, T_nodal, levels=50, cmap="hot_r")
    plt.colorbar(cf, ax=ax, label="Temperature [°C]")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    T_min = T_nodal.min()
    T_max = T_nodal.max()
    ax.set_title("{}\nT_min = {:.1f} °C,  T_max = {:.1f} °C".format(
        title, T_min, T_max))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_heat_flux(nodes, elements, T_nodal, flux_x_nodal, flux_y_nodal,
                   title, save_path, show=False):
    """Plot temperature contours with heat-flux arrows overlaid."""
    fig, ax = plt.subplots(figsize=(6, 6))

    conn = np.zeros((len(elements), 3), dtype=np.int32)
    for e in range(len(elements)):
        conn[e, 0] = int(elements[e, 0])
        conn[e, 1] = int(elements[e, 1])
        conn[e, 2] = int(elements[e, 2])

    ax.tricontourf(nodes[:, 0], nodes[:, 1], conn, T_nodal,
                   levels=20, cmap="hot_r", alpha=0.7)

    # Subsample nodes for arrows (plotting every node is too dense)
    step = max(1, len(nodes) // 400)
    x_q  = nodes[::step, 0]
    y_q  = nodes[::step, 1]
    qx   = flux_x_nodal[::step]
    qy   = flux_y_nodal[::step]

    ax.quiver(x_q, y_q, qx, qy, color="steelblue", scale=None,
              scale_units="xy", angles="xy", width=0.003, alpha=0.8)

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_radial(nodes, T_nodal, T_exact_fn, title, save_path, show=False):
    """
    Scatter plot of FEM nodal temperatures vs radius,
    with the exact analytical solution overlaid as a line.
    """
    r_nodes = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)

    r_line = np.linspace(r_nodes.min(), r_nodes.max(), 300)
    T_line = T_exact_fn(r_line)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r_nodes, T_nodal, s=3, alpha=0.4, color="#4c72b0",
               label="FEM nodes")
    ax.plot(r_line, T_line, color="red", linewidth=2, label="Exact solution")
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_convergence(mesh_sizes, L2_errors, H1_errors, title, save_path, show=False):
    """
    Log-log convergence plot.
    Shows measured L2 and H1 errors vs mesh size h,
    together with reference lines of slope 2 and 1.
    """
    h  = np.array(mesh_sizes)
    L2 = np.array(L2_errors)
    H1 = np.array(H1_errors)

    # Reference lines through the last data point
    h_ref = np.array([h.min() * 0.5, h.max() * 2.0])
    L2_ref = L2[-1] * (h_ref / h[-1])**2
    H1_ref = H1[-1] * (h_ref / h[-1])**1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, L2, "o-", color="#4c72b0", linewidth=2, label="L2 error")
    ax.loglog(h, H1, "s-", color="#dd8452", linewidth=2, label="H1 error")
    ax.loglog(h_ref, L2_ref, "--", color="#4c72b0", alpha=0.5, label="O(h^2) ref")
    ax.loglog(h_ref, H1_ref, "--", color="#dd8452", alpha=0.5, label="O(h)   ref")
    ax.set_xlabel("Mesh size h [m]")
    ax.set_ylabel("Error norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
