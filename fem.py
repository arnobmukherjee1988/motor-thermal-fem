"""
fem.py
======
All finite element mathematics in one file.

Sections
--------
1. ELEMENT LEVEL     -- stiffness matrix and load vector for one triangle
2. ROBIN EDGE        -- convection BC contribution for one boundary edge
3. GLOBAL ASSEMBLY   -- scatter local matrices into the global system
4. BOUNDARY CONDITIONS
     apply_dirichlet  -- prescribed temperature
     apply_robin      -- convective cooling
5. SOLVER            -- solve the linear system K * T = f

No classes. Every function takes arrays as inputs and returns arrays.
Style: explicit for-loops, named intermediate variables -- like C or Fortran.
"""

import numpy as np
from scipy.sparse import coo_matrix, lil_matrix


# =============================================================================
# SECTION 1 -- Element stiffness matrix and load vector
# =============================================================================

def element_stiffness_and_load(x1, y1, x2, y2, x3, y3, k_e, q_e):
    """
    Compute the 3x3 local stiffness matrix and 3-element load vector
    for one linear (P1) triangular element.

    The triangle has nodes at (x1,y1), (x2,y2), (x3,y3) in CCW order.

    Physics recap
    -------------
    For steady-state heat conduction  -div(k * grad(T)) = q,
    the weak form gives these element integrals:

      K_e[i,j] = k_e * Area * (b_i*b_j + c_i*c_j)   (stiffness)
      f_e[i]   = q_e * Area / 3                       (load)

    where b_i, c_i are the constant shape-function gradient components:
      b_i = (y_j - y_k) / (2 * Area)
      c_i = (x_k - x_j) / (2 * Area)
    with (i,j,k) cycling as (1,2,3), (2,3,1), (3,1,2).

    Parameters
    ----------
    x1,y1, x2,y2, x3,y3 : coordinates of the three nodes [m]
    k_e : thermal conductivity of this element [W/(m.K)]
    q_e : volumetric heat source in this element [W/m3]

    Returns
    -------
    K_e  : 2-D numpy array, shape (3, 3)
    f_e  : 1-D numpy array, shape (3,)
    area : float, element area [m2]
    """

    # ── Element area ──────────────────────────────────────────────────────────
    # 2 * Area = determinant of the coordinate matrix
    two_area = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)

    if two_area <= 0.0:
        raise ValueError(
            "Triangle nodes are not in counter-clockwise order "
            "(or the triangle is degenerate). "
            "2*Area = {:.3e}".format(two_area)
        )

    area = 0.5 * two_area

    # ── Shape function gradient coefficients ──────────────────────────────────
    # b coefficients (related to y coordinates)
    b1 = (y2 - y3) / two_area
    b2 = (y3 - y1) / two_area
    b3 = (y1 - y2) / two_area

    # c coefficients (related to x coordinates)
    c1 = (x3 - x2) / two_area
    c2 = (x1 - x3) / two_area
    c3 = (x2 - x1) / two_area

    # Collect into arrays for convenient matrix multiplication
    b = np.array([b1, b2, b3])
    c = np.array([c1, c2, c3])

    # ── Local stiffness matrix: K_e = k_e * Area * (b * b^T + c * c^T) ───────
    K_e = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            K_e[i, j] = k_e * area * (b[i] * b[j] + c[i] * c[j])

    # ── Local load vector: f_e[i] = q_e * Area / 3 ───────────────────────────
    f_e = np.zeros(3)
    for i in range(3):
        f_e[i] = q_e * area / 3.0

    return K_e, f_e, area


# =============================================================================
# SECTION 2 -- Robin (convection) contribution for one boundary edge
# =============================================================================

def robin_edge(xi, yi, xj, yj, h, T_inf):
    """
    Compute the stiffness and load contributions from one convective edge.

    When the boundary condition is  -k * dT/dn = h * (T - T_inf),
    the weak form adds:

      K_R = (h * L / 6) * [[2, 1],    (2x2 matrix for nodes i and j)
                            [1, 2]]

      f_R = (h * T_inf * L / 2) * [1, 1]^T

    where L is the length of the edge.

    Parameters
    ----------
    xi, yi : coordinates of edge node i [m]
    xj, yj : coordinates of edge node j [m]
    h      : convection coefficient [W/(m2.K)]
    T_inf  : ambient (coolant) temperature [deg C or K]

    Returns
    -------
    K_R : 2-D numpy array, shape (2, 2)
    f_R : 1-D numpy array, shape (2,)
    """
    # Edge length
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2)

    K_R = np.zeros((2, 2))
    K_R[0, 0] = h * L / 3.0
    K_R[0, 1] = h * L / 6.0
    K_R[1, 0] = h * L / 6.0
    K_R[1, 1] = h * L / 3.0

    f_R = np.zeros(2)
    f_R[0] = h * T_inf * L / 2.0
    f_R[1] = h * T_inf * L / 2.0

    return K_R, f_R


# =============================================================================
# SECTION 3 -- Global assembly
# =============================================================================

def assemble(nodes, elements, k_per_element, q_per_element):
    """
    Assemble the global stiffness matrix K and load vector f
    by looping over all elements and scattering each local
    contribution into the global arrays.

    The key idea ("scatter-add"):
        For element e with global node indices [g0, g1, g2]:
            K[ g_i, g_j ]  +=  K_e[i, j]    for all i, j in 0..2
            f[ g_i       ]  +=  f_e[i]

    Parameters
    ----------
    nodes          : array (N, 2)   node coordinates
    elements       : array (Ne, 3)  element connectivity (0-based indices)
    k_per_element  : array (Ne,)    conductivity per element [W/(m.K)]
    q_per_element  : array (Ne,)    heat source per element [W/m3]

    Returns
    -------
    K : scipy CSR sparse matrix, shape (N, N)
    f : 1-D numpy array, shape (N,)
    """
    N  = nodes.shape[0]
    Ne = elements.shape[0]

    # Pre-allocate COO arrays.
    # A 3x3 element matrix has 9 entries, so we need 9 * Ne total slots.
    row_idx = np.zeros(9 * Ne, dtype=np.int32)
    col_idx = np.zeros(9 * Ne, dtype=np.int32)
    data    = np.zeros(9 * Ne, dtype=np.float64)

    f = np.zeros(N, dtype=np.float64)

    ptr = 0   # write position in the COO arrays

    for e in range(Ne):
        # Global node indices of this element
        g0 = int(elements[e, 0])
        g1 = int(elements[e, 1])
        g2 = int(elements[e, 2])

        # Node coordinates
        x1 = nodes[g0, 0];  y1 = nodes[g0, 1]
        x2 = nodes[g1, 0];  y2 = nodes[g1, 1]
        x3 = nodes[g2, 0];  y3 = nodes[g2, 1]

        # Local stiffness matrix and load vector
        K_e, f_e, area = element_stiffness_and_load(
            x1, y1, x2, y2, x3, y3,
            k_per_element[e], q_per_element[e]
        )

        # Global node indices as a list for easy indexing
        g = [g0, g1, g2]

        # Scatter K_e into COO triplet arrays
        for i in range(3):
            for j in range(3):
                row_idx[ptr] = g[i]
                col_idx[ptr] = g[j]
                data[ptr]    = K_e[i, j]
                ptr = ptr + 1

            # Scatter f_e into global f
            f[g[i]] = f[g[i]] + f_e[i]

    # Trim arrays to the number of entries actually written
    row_idx = row_idx[:ptr]
    col_idx = col_idx[:ptr]
    data    = data[:ptr]

    # Convert COO -> CSR.  scipy automatically sums duplicate (i,j) entries --
    # this is how overlapping element contributions get added together.
    K = coo_matrix((data, (row_idx, col_idx)), shape=(N, N)).tocsr()

    return K, f


# =============================================================================
# SECTION 4 -- Boundary conditions
# =============================================================================

def apply_dirichlet(K, f, dirichlet_nodes, T_values):
    """
    Enforce prescribed temperatures (Dirichlet BC) using row-and-column
    elimination.  This keeps K symmetric.

    For each Dirichlet node g_i with prescribed temperature T_D:
      Step 1.  For every other row j:  f[j] -= K[j, g_i] * T_D
      Step 2.  Zero out row g_i and column g_i in K.
      Step 3.  Set K[g_i, g_i] = 1  and  f[g_i] = T_D.
               (The equation now simply reads: 1 * T[g_i] = T_D.)

    Parameters
    ----------
    K               : scipy CSR sparse matrix (N x N)
    f               : 1-D array (N,)
    dirichlet_nodes : 1-D integer array   -- indices of prescribed nodes
    T_values        : float  (same T for all Dirichlet nodes)
                      OR  dictionary  { node_index : temperature }

    Returns
    -------
    K : modified sparse matrix (still symmetric)
    f : modified load vector
    """
    N = K.shape[0]

    # Build a full {node_index: temperature} dictionary either way
    if isinstance(T_values, (int, float)):
        T_dict = {}
        for n in dirichlet_nodes:
            T_dict[int(n)] = float(T_values)
    else:
        T_dict = {}
        for n in T_values:
            T_dict[int(n)] = float(T_values[n])

    # Convert to LIL format for fast row/column modification
    K_lil = K.tolil()

    for g_i in dirichlet_nodes:
        g_i  = int(g_i)
        T_D  = T_dict[g_i]

        # Step 1: modify RHS for all other rows
        for j in range(N):
            if j != g_i:
                val = K_lil[j, g_i]
                if val != 0.0:
                    f[j] = f[j] - val * T_D

        # Step 2: zero out row and column
        K_lil[g_i, :] = 0.0
        K_lil[:, g_i] = 0.0

        # Step 3: put identity on the diagonal and T_D in RHS
        K_lil[g_i, g_i] = 1.0
        f[g_i] = T_D

    return K_lil.tocsr(), f


def apply_robin(K, f, outer_bc_edges, nodes, h_coeff, T_inf):
    """
    Add Robin (convective cooling) contributions to K and f.

    This implements:  -k * dT/dn = h * (T - T_inf)  on each outer edge.

    Parameters
    ----------
    K              : scipy CSR sparse matrix (N x N)
    f              : 1-D array (N,)
    outer_bc_edges : array (M, 2)   pairs of adjacent outer-boundary nodes
    nodes          : array (N, 2)   node coordinates
    h_coeff        : float          convection coefficient [W/(m2.K)]
    T_inf          : float          ambient temperature [deg C or K]

    Returns
    -------
    K : modified sparse matrix
    f : modified load vector
    """
    K_lil = K.tolil()

    for e in range(len(outer_bc_edges)):
        i = int(outer_bc_edges[e, 0])
        j = int(outer_bc_edges[e, 1])

        xi = nodes[i, 0];  yi = nodes[i, 1]
        xj = nodes[j, 0];  yj = nodes[j, 1]

        K_R, f_R = robin_edge(xi, yi, xj, yj, h_coeff, T_inf)

        # Scatter the 2x2 K_R into the global K
        K_lil[i, i] = K_lil[i, i] + K_R[0, 0]
        K_lil[i, j] = K_lil[i, j] + K_R[0, 1]
        K_lil[j, i] = K_lil[j, i] + K_R[1, 0]
        K_lil[j, j] = K_lil[j, j] + K_R[1, 1]

        # Scatter f_R into the global f
        f[i] = f[i] + f_R[0]
        f[j] = f[j] + f_R[1]

    return K_lil.tocsr(), f


# =============================================================================
# SECTION 5 -- Linear system solver
# =============================================================================

def solve(K, f):
    """
    Solve the linear system  K * T = f  for the nodal temperatures T.

    Uses scipy's direct sparse LU solver (SuperLU).
    This is exact to machine precision and robust for all problem sizes
    in this project.

    Parameters
    ----------
    K : scipy CSR sparse matrix (N x N),  symmetric positive definite
    f : 1-D array (N,)

    Returns
    -------
    T : 1-D array (N,)   nodal temperatures [deg C or K]
    """
    from scipy.sparse.linalg import spsolve

    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be square, got shape {}".format(K.shape))
    if K.shape[0] != len(f):
        raise ValueError("K size {} and f size {} do not match".format(
            K.shape[0], len(f)))

    T = spsolve(K.tocsr(), f)
    return T
