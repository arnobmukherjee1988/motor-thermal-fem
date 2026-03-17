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

check_energy_balance(...)
    -> backward-compatible single-Robin energy balance

check_energy_balance_multi(...)
    -> Q_gen, Q_conv_total, relative_error, per-boundary contributions

compute_boundary_conductive_flux(...)
    -> independent boundary flux integral from nodal flux vectors

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
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    k_per_element  : array (Ne,) for isotropic conductivity
                     OR array (Ne,2,2) for tensor conductivity

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

        # Heat flux q = -K * grad(T)
        k_e = k_per_element[e]
        if np.isscalar(k_e):
            flux_x[e] = -float(k_e) * dT_dx
            flux_y[e] = -float(k_e) * dT_dy
        else:
            k_mat = np.asarray(k_e, dtype=np.float64)
            flux_x[e] = -(k_mat[0, 0] * dT_dx + k_mat[0, 1] * dT_dy)
            flux_y[e] = -(k_mat[1, 0] * dT_dx + k_mat[1, 1] * dT_dy)

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

def _compute_total_generated_heat(nodes, elements, q_per_element):
    Q_gen = 0.0
    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])
        x1 = nodes[g0, 0]; y1 = nodes[g0, 1]
        x2 = nodes[g1, 0]; y2 = nodes[g1, 1]
        x3 = nodes[g2, 0]; y3 = nodes[g2, 1]
        two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * abs(two_area)
        Q_gen += q_per_element[e] * area
    return Q_gen


def _compute_robin_heat(nodes, T_nodal, bc_edges, h_coeff, T_inf):
    Q_conv = 0.0
    for e in range(len(bc_edges)):
        i = int(bc_edges[e, 0])
        j = int(bc_edges[e, 1])
        xi = nodes[i, 0]; yi = nodes[i, 1]
        xj = nodes[j, 0]; yj = nodes[j, 1]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)
        T_avg = 0.5 * (T_nodal[i] + T_nodal[j])
        Q_conv += h_coeff * (T_avg - T_inf) * L
    return Q_conv


def check_energy_balance_multi(nodes, elements, T_nodal, q_per_element, robin_sets):
    """
    Energy balance with multiple Robin boundaries.
    robin_sets is a list of dicts, each with:
      {"name": str, "edges": array(M,2), "h": float, "T_inf": float}
    """
    Q_gen = _compute_total_generated_heat(nodes, elements, q_per_element)
    contributions = {}
    Q_conv_total = 0.0

    for bc in robin_sets:
        q_bc = _compute_robin_heat(
            nodes=nodes,
            T_nodal=T_nodal,
            bc_edges=bc["edges"],
            h_coeff=bc["h"],
            T_inf=bc["T_inf"],
        )
        contributions[bc["name"]] = q_bc
        Q_conv_total += q_bc

    if abs(Q_gen) > 1e-30:
        rel_err = abs(Q_gen - Q_conv_total) / abs(Q_gen)
    else:
        rel_err = 0.0

    return Q_gen, Q_conv_total, rel_err, contributions


def check_energy_balance(nodes, elements, T_nodal, q_per_element,
                         outer_bc_edges, h_coeff, T_inf):
    """
    Backward-compatible single-Robin energy balance wrapper.
    """
    Q_gen, Q_conv_total, rel_err, _ = check_energy_balance_multi(
        nodes=nodes,
        elements=elements,
        T_nodal=T_nodal,
        q_per_element=q_per_element,
        robin_sets=[{
            "name": "outer",
            "edges": outer_bc_edges,
            "h": h_coeff,
            "T_inf": T_inf,
        }],
    )
    return Q_gen, Q_conv_total, rel_err


def compute_boundary_conductive_flux(nodes, bc_edges, flux_x_nodal, flux_y_nodal,
                                     normal_sign=1.0):
    """
    Integrate conductive heat flow across a boundary using nodal flux values.
    normal_sign: +1 for +r-hat normal (outer boundary),
                 -1 for -r-hat normal (inner-hole boundary).
    """
    total = 0.0
    for e in range(len(bc_edges)):
        i = int(bc_edges[e, 0])
        j = int(bc_edges[e, 1])
        xi = nodes[i, 0]; yi = nodes[i, 1]
        xj = nodes[j, 0]; yj = nodes[j, 1]
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2)

        mx = 0.5 * (xi + xj)
        my = 0.5 * (yi + yj)
        rm = np.sqrt(mx * mx + my * my)
        if rm < 1e-14:
            continue

        nx = normal_sign * mx / rm
        ny = normal_sign * my / rm

        qx = 0.5 * (flux_x_nodal[i] + flux_x_nodal[j])
        qy = 0.5 * (flux_y_nodal[i] + flux_y_nodal[j])
        total += (qx * nx + qy * ny) * L
    return total


# =============================================================================
# Plots
# =============================================================================

def _build_connectivity(elements):
    conn = np.zeros((len(elements), 3), dtype=np.int32)
    for e in range(len(elements)):
        conn[e, 0] = int(elements[e, 0])
        conn[e, 1] = int(elements[e, 1])
        conn[e, 2] = int(elements[e, 2])
    return conn


def _material_interface_segments(nodes, elements, element_tags):
    """
    Build line segments on interfaces between different material tags.
    """
    edge_tag = {}
    segments = []
    for e in range(len(elements)):
        tag = int(element_tags[e])
        tri = [int(elements[e, 0]), int(elements[e, 1]), int(elements[e, 2])]
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            key = (a, b) if a < b else (b, a)
            if key not in edge_tag:
                edge_tag[key] = tag
            elif edge_tag[key] != tag:
                na, nb = key
                segments.append([nodes[na], nodes[nb]])
    return segments


def _plot_material_interfaces(ax, nodes, elements, element_tags):
    if element_tags is None:
        return
    segments = _material_interface_segments(nodes, elements, element_tags)
    if len(segments) == 0:
        return
    lc = mc.LineCollection(segments, colors="white", linewidths=0.7, alpha=0.9)
    ax.add_collection(lc)


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


def plot_temperature(nodes, elements, T_nodal, title, save_path, show=False,
                     element_tags=None):
    """Plot filled temperature contour map."""
    fig, ax = plt.subplots(figsize=(6, 6))

    conn = _build_connectivity(elements)

    cf = ax.tricontourf(nodes[:, 0], nodes[:, 1], conn, T_nodal, levels=50, cmap="hot_r")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    fig.colorbar(cf, cax=cax, label=r"Temperature [$^\circ$C]")
    _plot_material_interfaces(ax, nodes, elements, element_tags)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    T_min = T_nodal.min()
    T_max = T_nodal.max()
    ax.set_title("{}\n$T_{{\\min}}$ = {:.1f} $^\\circ$C,  $T_{{\\max}}$ = {:.1f} $^\\circ$C".format(
        title, T_min, T_max))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_heat_flux(nodes, elements, T_nodal, flux_x_nodal, flux_y_nodal,
                   title, save_path, show=False, element_tags=None):
    """Plot flux magnitude contours with normalized heat-flux direction arrows."""
    fig, ax = plt.subplots(figsize=(6, 6))

    conn = _build_connectivity(elements)
    q_mag = np.sqrt(flux_x_nodal ** 2 + flux_y_nodal ** 2)

    cf = ax.tricontourf(
        nodes[:, 0], nodes[:, 1], conn, q_mag,
        levels=40, cmap="viridis"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    fig.colorbar(cf, cax=cax, label=r"$|q|$ [W/m$^2$]")
    _plot_material_interfaces(ax, nodes, elements, element_tags)

    # Subsample nodes for arrows and normalize vectors to show direction clearly.
    # A denser sample + dual-layer arrows improves visibility on PDF export.
    step = max(1, len(nodes) // 250)
    idx = np.arange(0, len(nodes), step)
    x_q = nodes[idx, 0]
    y_q = nodes[idx, 1]
    qx = flux_x_nodal[idx]
    qy = flux_y_nodal[idx]
    qn = np.sqrt(qx ** 2 + qy ** 2)
    # Avoid plotting near-zero vectors that clutter the field.
    mag_cut = max(1e-14, 0.05 * np.percentile(q_mag, 90))
    keep = qn >= mag_cut
    x_q = x_q[keep]
    y_q = y_q[keep]
    qx = qx[keep]
    qy = qy[keep]
    qn = qn[keep]
    qn[qn < 1e-14] = 1.0
    ux = qx / qn
    uy = qy / qn

    # White underlay + black overlay for contrast on dark/light regions.
    # ax.quiver(
    #     x_q, y_q, ux, uy, color="white",
    #     scale=115, scale_units="xy", angles="xy",
    #     width=0.006, alpha=0.85, pivot="mid",
    #     headwidth=3.6, headlength=5.0, headaxislength=4.4
    # )
    qv = ax.quiver(
        x_q, y_q, ux, uy, color="orange",
        scale=125, scale_units="xy", angles="xy",
        width=0.0044, alpha=0.99, pivot="mid",
        headwidth=3.6, headlength=5.0, headaxislength=4.4
    )
    # ax.quiverkey(qv, 0.88, 1.02, 1.0, "unit direction", coordinates="axes")

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
    ax.set_ylabel(r"Temperature [$^\circ$C]")
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
    ax.loglog(h_ref, L2_ref, "--", color="#4c72b0", alpha=0.5, label=r"$\mathcal{O}(h^2)$ ref")
    ax.loglog(h_ref, H1_ref, "--", color="#dd8452", alpha=0.5, label=r"$\mathcal{O}(h)$ ref")
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
