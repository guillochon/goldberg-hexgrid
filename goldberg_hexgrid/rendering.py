"""
Rendering utilities for hex grids and Goldberg polyhedra.

This module provides functions to convert hex grid data structures
into visual representations using matplotlib.
"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .goldberg_polyhedron import Vertex3D


def calculate_hex_vertices(
    center: Vertex3D,
    neighbors: list[Vertex3D],
    radius: float = 1.0,
) -> list[Vertex3D]:
    """
    Calculate the vertices of a hexagon or pentagon on a sphere.

    Args:
        center: Center position of the hex/pentagon on the sphere
        neighbors: List of neighbor positions (5 for pentagon, 6 for hexagon)
        radius: Radius of the sphere (default 1.0 for unit sphere)

    Returns:
        List of vertex positions forming the polygon
    """
    center_normal = center.normalize()
    num_neighbors = len(neighbors)

    # Create a reference frame for the tangent plane
    world_x = Vertex3D(1, 0, 0)
    world_y = Vertex3D(0, 1, 0)
    world_z = Vertex3D(0, 0, 1)

    # Choose reference axis that's not too close to center normal
    if abs(center_normal.dot(world_x)) < 0.9:
        ref_axis = world_x
    elif abs(center_normal.dot(world_y)) < 0.9:
        ref_axis = world_y
    else:
        ref_axis = world_z

    # Create right and forward vectors for the tangent plane
    right = center_normal.cross(ref_axis).normalize()
    forward = right.cross(center_normal).normalize()

    # Project neighbors onto tangent plane and calculate angles
    neighbor_angles: list[tuple[int, float]] = []
    for i, neighbor in enumerate(neighbors):
        direction = Vertex3D(
            neighbor.x - center.x,
            neighbor.y - center.y,
            neighbor.z - center.z,
        )
        direction = direction.normalize()

        # Project onto tangent plane
        dir_right = direction.dot(right)
        dir_forward = direction.dot(forward)

        # Calculate angle
        angle = math.atan2(dir_forward, dir_right)
        neighbor_angles.append((i, angle))

    # Sort neighbors by angle
    neighbor_angles.sort(key=lambda x: x[1])

    # Calculate vertices using precise spherical geometry
    # For a regular hexagon on a sphere, vertices are at a specific angular distance from center
    # The vertex is the intersection point of the great circle arcs that form the boundaries
    # between the center tile and its adjacent neighbors
    
    # Calculate the angular distance from center to a neighbor (using dot product)
    # This gives us the base distance for calculating vertex positions
    center_normalized = center.normalize()
    neighbor_distances = []
    for neighbor in neighbors:
        neighbor_normalized = neighbor.normalize()
        # Angular distance = arccos(dot product) for unit vectors
        cos_angle = center_normalized.dot(neighbor_normalized)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
        angular_dist = math.acos(cos_angle)
        neighbor_distances.append(angular_dist)
    
    # Use average angular distance to neighbors (should be similar for regular hexagons)
    avg_angular_dist = sum(neighbor_distances) / len(neighbor_distances) if neighbor_distances else 0.1
    
    # For a regular hexagon, vertex angular distance from center is:
    # vertex_angle = neighbor_angle * sqrt(3)/2
    # This is the precise geometric relationship for regular hexagons
    vertex_angular_dist = avg_angular_dist * (math.sqrt(3) / 3)
    
    vertices = []
    for idx in range(num_neighbors):
        curr_idx = neighbor_angles[idx][0]
        next_idx = neighbor_angles[(idx + 1) % num_neighbors][0]

        curr_neighbor = neighbors[curr_idx]
        next_neighbor = neighbors[next_idx]

        # Calculate direction to vertex: direction toward midpoint of two neighbors
        midpoint = Vertex3D(
            (curr_neighbor.x + next_neighbor.x) / 2,
            (curr_neighbor.y + next_neighbor.y) / 2,
            (curr_neighbor.z + next_neighbor.z) / 2,
        )
        midpoint_normalized = midpoint.normalize()
        
        # Project midpoint direction onto tangent plane at center
        # Remove the component along the center normal to get tangent direction
        dot_with_center = midpoint_normalized.dot(center_normalized)
        tangent_dir = Vertex3D(
            midpoint_normalized.x - center_normalized.x * dot_with_center,
            midpoint_normalized.y - center_normalized.y * dot_with_center,
            midpoint_normalized.z - center_normalized.z * dot_with_center,
        )
        
        # Normalize the tangent direction
        tangent_length = math.sqrt(
            tangent_dir.x * tangent_dir.x +
            tangent_dir.y * tangent_dir.y +
            tangent_dir.z * tangent_dir.z
        )
        if tangent_length < 1e-10:
            # Fallback: use a perpendicular vector if tangent is too small
            # Find any vector perpendicular to center
            if abs(center_normalized.x) < 0.9:
                tangent_dir = Vertex3D(1, 0, 0)
            elif abs(center_normalized.y) < 0.9:
                tangent_dir = Vertex3D(0, 1, 0)
            else:
                tangent_dir = Vertex3D(0, 0, 1)
            # Project and normalize
            dot_with_center = tangent_dir.dot(center_normalized)
            tangent_dir = Vertex3D(
                tangent_dir.x - center_normalized.x * dot_with_center,
                tangent_dir.y - center_normalized.y * dot_with_center,
                tangent_dir.z - center_normalized.z * dot_with_center,
            )
            tangent_length = math.sqrt(
                tangent_dir.x * tangent_dir.x +
                tangent_dir.y * tangent_dir.y +
                tangent_dir.z * tangent_dir.z
            )
        
        tangent_dir_normalized = Vertex3D(
            tangent_dir.x / tangent_length,
            tangent_dir.y / tangent_length,
            tangent_dir.z / tangent_length,
        )
        
        # Calculate vertex position using spherical interpolation (slerp)
        # Vertex is at vertex_angular_dist from center in the tangent_dir direction
        cos_vertex_angle = math.cos(vertex_angular_dist)
        sin_vertex_angle = math.sin(vertex_angular_dist)
        
        # Slerp formula: center * cos(angle) + tangent_dir * sin(angle)
        vertex_pos = Vertex3D(
            center_normalized.x * cos_vertex_angle + tangent_dir_normalized.x * sin_vertex_angle,
            center_normalized.y * cos_vertex_angle + tangent_dir_normalized.y * sin_vertex_angle,
            center_normalized.z * cos_vertex_angle + tangent_dir_normalized.z * sin_vertex_angle,
        )
        
        # Scale to desired radius
        vertex = Vertex3D(
            vertex_pos.x * radius,
            vertex_pos.y * radius,
            vertex_pos.z * radius,
        )
        vertices.append(vertex)

    return vertices


def render_hex_grid_3d(
    positions: list[Vertex3D],
    neighbors_map: dict[int, list[int]],
    figsize: tuple[int, int] = (12, 12),
    alpha: float = 0.7,
    edge_color: str = "black",
    face_color: str = "lightblue",
    show_axes: bool = False,
    ax: Optional[Axes3D] = None,
) -> tuple[plt.Figure, Axes3D]:
    """
    Render a hex grid on a sphere in 3D.

    Args:
        positions: List of 3D positions for each tile
        neighbors_map: Dictionary mapping tile index to list of neighbor indices
        figsize: Figure size (width, height)
        alpha: Transparency of faces (0-1)
        edge_color: Color of edges
        face_color: Color of faces
        show_axes: Whether to show axes
        ax: Optional existing axes to plot on

    Returns:
        Tuple of (figure, axes)
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Calculate and draw each polygon
    for i, center in enumerate(positions):
        neighbor_indices = neighbors_map.get(i, [])
        if not neighbor_indices:
            continue

        # Get neighbor positions
        neighbor_positions = [positions[j] for j in neighbor_indices]

        # Calculate vertices
        vertices = calculate_hex_vertices(center, neighbor_positions)

        # Convert to numpy array for plotting
        verts = np.array([[v.x, v.y, v.z] for v in vertices])

        # Determine if this is a pentagon (5 neighbors) or hexagon (6 neighbors)
        is_pentagon = len(neighbor_indices) == 5
        tile_face_color = "red" if is_pentagon else face_color
        tile_edge_color = "darkred" if is_pentagon else edge_color

        # Create polygon
        poly = Poly3DCollection([verts], alpha=alpha, facecolor=tile_face_color, edgecolor=tile_edge_color)
        ax.add_collection3d(poly)

    # Set equal aspect ratio
    max_range = 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_aspect("equal")

    if not show_axes:
        ax.set_axis_off()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax


def render_hex_grid_2d_projection(
    positions: list[Vertex3D],
    neighbors_map: dict[int, list[int]],
    projection: str = "mollweide",
    figsize: tuple[int, int] = (16, 8),
    alpha: float = 0.7,
    edge_color: str = "black",
    face_color: str = "lightblue",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Render a hex grid using a 2D map projection.

    Args:
        positions: List of 3D positions for each tile
        neighbors_map: Dictionary mapping tile index to list of neighbor indices
        projection: Projection type. Supported: 'mollweide' (default), 'polar', or None for regular plot.
                    Note: For advanced geographic projections, consider using cartopy.
        figsize: Figure size (width, height)
        alpha: Transparency of faces (0-1)
        edge_color: Color of edges
        face_color: Color of faces
        ax: Optional existing axes to plot on

    Returns:
        Tuple of (figure, axes)
    """
    if ax is None:
        if projection:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
        else:
            fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Convert 3D positions to lat/lon (in radians for map projections)
    def vertex_to_latlon(vertex: Vertex3D) -> tuple[float, float]:
        """Convert 3D vertex to latitude and longitude in radians."""
        lat = math.asin(vertex.z)
        lon = math.atan2(vertex.y, vertex.x)
        return lat, lon

    # Draw each polygon
    for i, center in enumerate(positions):
        neighbor_indices = neighbors_map.get(i, [])
        if not neighbor_indices:
            continue

        # Get neighbor positions
        neighbor_positions = [positions[j] for j in neighbor_indices]

        # Calculate vertices
        vertices = calculate_hex_vertices(center, neighbor_positions)

        # Convert to lat/lon (in radians)
        latlons = [vertex_to_latlon(v) for v in vertices]
        lats = [lat for lat, _ in latlons]
        lons = [lon for _, lon in latlons]

        # Close the polygon
        lats.append(lats[0])
        lons.append(lons[0])

        # Determine if this is a pentagon (5 neighbors) or hexagon (6 neighbors)
        is_pentagon = len(neighbor_indices) == 5
        tile_face_color = "red" if is_pentagon else face_color
        tile_edge_color = "darkred" if is_pentagon else edge_color

        # Plot polygon
        # For matplotlib projections like 'mollweide', coordinates should be in radians
        # and the projection handles the transformation automatically
        ax.plot(lons, lats, color=tile_edge_color, linewidth=0.5)
        ax.fill(lons, lats, alpha=alpha, color=tile_face_color, edgecolor=tile_edge_color, linewidth=0.5)

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    return fig, ax


def render_hex_grid_interactive(
    positions: list[Vertex3D],
    neighbors_map: dict[int, list[int]],
    figsize: tuple[int, int] = (12, 12),
    alpha: float = 0.7,
    edge_color: str = "black",
    face_color: str = "lightblue",
) -> tuple[plt.Figure, Axes3D]:
    """
    Render a hex grid in 3D with interactive rotation.

    This is a convenience wrapper around render_hex_grid_3d that ensures
    the plot is interactive.

    Args:
        positions: List of 3D positions for each tile
        neighbors_map: Dictionary mapping tile index to list of neighbor indices
        figsize: Figure size (width, height)
        alpha: Transparency of faces (0-1)
        edge_color: Color of edges
        face_color: Color of faces

    Returns:
        Tuple of (figure, axes)
    """
    return render_hex_grid_3d(
        positions, neighbors_map, figsize, alpha, edge_color, face_color, show_axes=True
    )
