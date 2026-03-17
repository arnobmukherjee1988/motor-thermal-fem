"""
validation.py
=============
Analytical solution for the annulus and error norm calculations.

The governing PDE in polar coordinates (radially symmetric, uniform source):
    -k/r * d/dr(r * dT/dr) = q_dot

General solution:
    T(r) = -(q_dot / 4k) * r^2  +  C1 * ln(r)  +  C2

The constants C1, C2 are found from the two boundary conditions:
    T(r_inner) = T_inner
    T(r_outer) = T_outer

Functions
---------
exact_temperature(r_inner, r_outer, T_inner, T_outer, k, q_dot)
    Returns a function  T_exact(r)

exact_gradient(r_inner, r_outer, T_inner, T_outer, k, q_dot)
    Returns a function  dTdr(r)  -- needed for H1 error

compute_L2_error(nodes, elements, T_h, T_exact_fn)
compute_H1_error(nodes, elements, T_h, dTdr_fn)

run_convergence_study(mesh_sizes, r_inner, r_outer, T_inner, T_outer, k, q_dot)
    Returns: mesh_sizes, L2_errors, H1_errors, L2_rate, H1_rate
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh       import make_annulus_mesh
from materials  import PHASE1_TAGS, get_element_properties
from fem        import assemble, apply_dirichlet, solve


# =============================================================================
# Analytical solution
# =============================================================================

def exact_temperature(r_inner, r_outer, T_inner, T_outer, k, q_dot=0.0):
    """
    Build and return the exact temperature function T_exact(r) for the annulus.

    Parameters
    ----------
    r_inner, r_outer : inner and outer radii [m]
    T_inner, T_outer : boundary temperatures [deg C]
    k                : conductivity [W/(m.K)]
    q_dot            : heat source [W/m3], default 0

    Returns
    -------
    T_fn : callable  T_fn(r) -> array of temperatures
    """
    log_ratio = np.log(r_outer / r_inner)

    if abs(q_dot) < 1e-30:
        # Pure conduction (no source):  T = C1 * ln(r) + C2
        C1 = (T_outer - T_inner) / log_ratio
        C2 = T_inner - C1 * np.log(r_inner)
    else:
        # With uniform source:  T = -(q/4k)*r^2 + C1*ln(r) + C2
        C1 = ((T_outer - T_inner) + (q_dot / (4.0 * k)) * (r_outer**2 - r_inner**2)) \
             / log_ratio
        C2 = T_inner + (q_dot / (4.0 * k)) * r_inner**2 - C1 * np.log(r_inner)

    # Return a plain function that evaluates T(r) at any radius r
    def T_fn(r):
        r = np.asarray(r, dtype=np.float64)
        return -(q_dot / (4.0 * k)) * r**2 + C1 * np.log(r) + C2

    return T_fn


def exact_gradient(r_inner, r_outer, T_inner, T_outer, k, q_dot=0.0):
    """
    Return the exact radial temperature gradient dT/dr(r).
    Used for computing the H1 error norm.
    """
    log_ratio = np.log(r_outer / r_inner)

    if abs(q_dot) < 1e-30:
        C1 = (T_outer - T_inner) / log_ratio
    else:
        C1 = ((T_outer - T_inner) + (q_dot / (4.0 * k)) * (r_outer**2 - r_inner**2)) \
             / log_ratio

    def dTdr(r):
        r = np.asarray(r, dtype=np.float64)
        return -(q_dot / (2.0 * k)) * r + C1 / r

    return dTdr


def evaluate_exact_at_nodes(nodes, T_fn):
    """
    Evaluate T_fn(r) at the radial position of each mesh node.

    Parameters
    ----------
    nodes  : array (N, 2)
    T_fn   : callable returned by exact_temperature()

    Returns
    -------
    T_exact_nodes : array (N,)
    """
    r = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    return T_fn(r)


# =============================================================================
# Error norms
# =============================================================================

def compute_L2_error(nodes, elements, T_h, T_exact_fn):
    """
    Compute the L2 error norm:
        ||T_h - T_exact||_L2 = sqrt( integral over domain of (T_h - T_exact)^2 dA )

    For a P1 triangle the integral is computed exactly using:
        integral of (e1^2 + e2^2 + e3^2 + e1*e2 + e2*e3 + e1*e3) * Area / 6
    where e_i = T_h(node i) - T_exact(node i).

    Parameters
    ----------
    nodes      : array (N, 2)
    elements   : array (Ne, 3)
    T_h        : array (N,)   FEM solution
    T_exact_fn : callable

    Returns
    -------
    L2_error : float
    """
    L2_sq = 0.0

    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])

        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]

        two_area = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        area = 0.5 * abs(two_area)

        # Exact temperatures at the three nodes
        r1 = np.sqrt(x1*x1 + y1*y1);  Te1 = T_exact_fn(r1)
        r2 = np.sqrt(x2*x2 + y2*y2);  Te2 = T_exact_fn(r2)
        r3 = np.sqrt(x3*x3 + y3*y3);  Te3 = T_exact_fn(r3)

        # Errors at the three nodes
        err1 = T_h[g0] - Te1
        err2 = T_h[g1] - Te2
        err3 = T_h[g2] - Te3

        # Exact integral of linear interpolation of errors squared over triangle
        L2_sq = L2_sq + area / 6.0 * (
            err1*err1 + err2*err2 + err3*err3 +
            err1*err2 + err2*err3 + err1*err3
        )

    return np.sqrt(abs(L2_sq))


def compute_H1_error(nodes, elements, T_h, dTdr_fn):
    """
    Compute the H1 semi-norm error:
        ||grad(T_h) - grad(T_exact)||_L2

    For each element:
      - FEM gradient is constant (computed from shape functions)
      - Exact gradient is evaluated at the element centroid

    Parameters
    ----------
    nodes     : array (N, 2)
    elements  : array (Ne, 3)
    T_h       : array (N,)
    dTdr_fn   : callable  dT/dr(r)

    Returns
    -------
    H1_error : float
    """
    H1_sq = 0.0

    for e in range(len(elements)):
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])

        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]

        two_area = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        area = 0.5 * abs(two_area)

        # FEM gradient (constant inside the element)
        b1 = (y2 - y3) / two_area
        b2 = (y3 - y1) / two_area
        b3 = (y1 - y2) / two_area
        c1 = (x3 - x2) / two_area
        c2 = (x1 - x3) / two_area
        c3 = (x2 - x1) / two_area

        T1 = T_h[g0];  T2 = T_h[g1];  T3 = T_h[g2]
        dTh_dx = b1*T1 + b2*T2 + b3*T3
        dTh_dy = c1*T1 + c2*T2 + c3*T3

        # Exact gradient at the centroid
        cx = (x1 + x2 + x3) / 3.0
        cy = (y1 + y2 + y3) / 3.0
        r_c = np.sqrt(cx*cx + cy*cy)

        dT_dr_exact = dTdr_fn(r_c)
        # Convert radial gradient to Cartesian:  dT/dx = dT/dr * cos(theta)
        cos_th = cx / r_c
        sin_th = cy / r_c
        dTe_dx = dT_dr_exact * cos_th
        dTe_dy = dT_dr_exact * sin_th

        # Squared error in gradient
        ex = dTh_dx - dTe_dx
        ey = dTh_dy - dTe_dy
        H1_sq = H1_sq + area * (ex*ex + ey*ey)

    return np.sqrt(abs(H1_sq))


# =============================================================================
# Convergence study
# =============================================================================

def run_convergence_study(mesh_sizes, r_inner, r_outer, T_inner, T_outer, k, q_dot,
                           verbose=True):
    """
    Run the FEM at a series of mesh sizes and compute error norms at each level.

    Parameters
    ----------
    mesh_sizes : list of floats (decreasing)
    r_inner, r_outer, T_inner, T_outer, k, q_dot : problem parameters

    Returns
    -------
    mesh_sizes : list
    L2_errors  : list
    H1_errors  : list
    L2_rate    : float   (slope of log-log line)
    H1_rate    : float
    """
    T_fn    = exact_temperature(r_inner, r_outer, T_inner, T_outer, k, q_dot)
    dTdr_fn = exact_gradient(r_inner, r_outer, T_inner, T_outer, k, q_dot)

    L2_errors = []
    H1_errors = []

    for h in mesh_sizes:
        # Build mesh
        nodes, elements, element_tags, inner_bc, outer_bc, outer_edges = \
            make_annulus_mesh(r_inner, r_outer, h)

        # Material properties
        k_arr, q_arr = get_element_properties(element_tags, PHASE1_TAGS)
        for i in range(len(k_arr)):
            k_arr[i] = k
            q_arr[i] = q_dot

        # Assemble
        K, f = assemble(nodes, elements, k_arr, q_arr)

        # Dirichlet BCs: prescribe exact temperature at inner and outer rings
        T_dict = {}
        for n in inner_bc:
            n = int(n)
            r = np.sqrt(nodes[n, 0]**2 + nodes[n, 1]**2)
            T_dict[n] = float(T_fn(r))
        for n in outer_bc:
            n = int(n)
            r = np.sqrt(nodes[n, 0]**2 + nodes[n, 1]**2)
            T_dict[n] = float(T_fn(r))

        bc_nodes = np.concatenate([inner_bc, outer_bc])
        K, f = apply_dirichlet(K, f, bc_nodes, T_dict)

        # Solve
        T_h = solve(K, f)

        # Errors
        L2 = compute_L2_error(nodes, elements, T_h, T_fn)
        H1 = compute_H1_error(nodes, elements, T_h, dTdr_fn)

        L2_errors.append(L2)
        H1_errors.append(H1)

        if verbose:
            print("  h = {:.4f}  ->  L2 = {:.4e},  H1 = {:.4e}".format(h, L2, H1))

    # Fit convergence rates from log-log slope
    log_h  = np.log(mesh_sizes)
    log_L2 = np.log(L2_errors)
    log_H1 = np.log(H1_errors)

    L2_rate = float(np.polyfit(log_h, log_L2, 1)[0])
    H1_rate = float(np.polyfit(log_h, log_H1, 1)[0])

    return mesh_sizes, L2_errors, H1_errors, L2_rate, H1_rate
