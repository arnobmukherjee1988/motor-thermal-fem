2D Finite Element Method for Steady-State Heat Conduction
==========================================================

A plain Python implementation of the finite element method (FEM) for
solving 2D steady-state heat conduction problems.

Written without classes or object-oriented programming.
All functions take numpy arrays as input and return numpy arrays.


Problems solved
---------------

Phase 1 - Plain annular ring with Dirichlet boundary conditions.
Validated against the exact analytical solution. Convergence rates
measured: L2 norm rate 1.925, H1 norm rate 0.983 (theoretical: 2.0 and 1.0).

Phase 2 - Slotted electric motor cross-section with three material regions
(stator iron, winding-slot effective medium, air gap), convective cooling on
outer and inner boundaries, effective anisotropic iron conductivity, and
temperature-dependent copper loss. Includes parametric studies of peak winding
temperature versus heat load, convection, and slot effective conductivity.


Files
-----

materials.py       Material properties (conductivity, heat source)
mesh.py            Mesh generation for annulus and motor geometries
fem.py             Element stiffness, assembly, boundary conditions, solver
postprocessing.py  Heat flux computation and plots
validation.py      Analytical solution and error norms
phase1.py          Run Phase 1 (annulus)
phase2.py          Run Phase 2 (motor)


Dependencies
------------

Python 3.8 or later
numpy
scipy
matplotlib

Install with:

    pip install numpy scipy matplotlib


How to run
----------

    cd fem_simple
    python phase1.py
    python phase2.py

Results (plots) are saved to the results/ folder.


Background
----------

Element type    : P1 linear triangles
Assembly        : COO scatter-add, converted to CSR sparse matrix
Solver          : Direct sparse LU (scipy SuperLU)
Annulus mesh    : Structured polar O-grid
Motor mesh      : Interface-conforming structured polar grid
