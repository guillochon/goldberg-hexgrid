"""
Goldberg Hexgrid - A Python package for generating Goldberg polyhedra and hex grid coordinates.

This package provides utilities for:
- Hexagonal coordinate systems (axial/cube coordinates)
- Goldberg polyhedron generation for spherical hex grids
- 3D vertex operations on unit spheres
"""

from .hex_grid import (
    HexCoordinates,
    hex_distance,
    hex_neighbors,
    generate_hex_sphere,
    hex_ring,
)
from .goldberg_polyhedron import (
    Vertex3D,
    generate_goldberg_hex_sphere,
    icosahedron_vertices,
    icosahedron_faces,
    calculate_pentagon_threshold,
    is_pentagon_tile,
)

__version__ = "0.1.0"
__all__ = [
    # Hex grid
    "HexCoordinates",
    "hex_distance",
    "hex_neighbors",
    "generate_hex_sphere",
    "hex_ring",
    # Goldberg polyhedron
    "Vertex3D",
    "generate_goldberg_hex_sphere",
    "icosahedron_vertices",
    "icosahedron_faces",
    "calculate_pentagon_threshold",
    "is_pentagon_tile",
]
