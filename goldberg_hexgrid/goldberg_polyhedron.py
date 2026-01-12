"""
Goldberg Polyhedron Generator
Based on https://github.com/shwuandwing/Goldberg-Polyhedron-Earth
Rewritten in Python from TypeScript implementation.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from .hex_grid import HexCoordinates

PHI = (1 + math.sqrt(5)) / 2


@dataclass
class Vertex3D:
    """3D vertex on sphere"""

    x: float
    y: float
    z: float

    def normalize(self) -> "Vertex3D":
        """Normalize to unit sphere"""
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length == 0:
            return Vertex3D(0, 0, 1)
        return Vertex3D(self.x / length, self.y / length, self.z / length)

    def distance_to(self, other: "Vertex3D") -> float:
        """Calculate 3D distance to another vertex"""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def dot(self, other: "Vertex3D") -> float:
        """Dot product"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vertex3D") -> "Vertex3D":
        """Cross product"""
        return Vertex3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def add_scaled(self, other: "Vertex3D", scale: float) -> "Vertex3D":
        """Add scaled vector"""
        return Vertex3D(
            self.x + other.x * scale, self.y + other.y * scale, self.z + other.z * scale
        )


def icosahedron_vertices() -> list[Vertex3D]:
    """Generate vertices of an icosahedron (matching TypeScript implementation)"""
    vertices = [
        Vertex3D(-1, PHI, 0),
        Vertex3D(1, PHI, 0),
        Vertex3D(-1, -PHI, 0),
        Vertex3D(1, -PHI, 0),
        Vertex3D(0, -1, PHI),
        Vertex3D(0, 1, PHI),
        Vertex3D(0, -1, -PHI),
        Vertex3D(0, 1, -PHI),
        Vertex3D(PHI, 0, -1),
        Vertex3D(PHI, 0, 1),
        Vertex3D(-PHI, 0, -1),
        Vertex3D(-PHI, 0, 1),
    ]
    return [v.normalize() for v in vertices]


def icosahedron_faces() -> list[tuple[int, int, int]]:
    """Return the 20 triangular faces of an icosahedron (matching TypeScript implementation)"""
    return [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]


def calculate_pentagon_threshold(radius: float, num_tiles: int) -> float:
    """
    Calculate the hexagon size threshold based on sphere radius and number of polygons.
    Uses the formula: sqrt(4 * radius^2 / num_tiles)
    This gives approximately the radius of each polygon if they were circles.
    For unit sphere (radius=1), this simplifies to: 2 / sqrt(num_tiles)
    """
    if num_tiles <= 0:
        return 0.15  # Fallback default
    return math.sqrt((4 * radius * radius) / num_tiles)


def is_pentagon_tile(pos: Vertex3D, ico_vertices: list[Vertex3D], threshold: float) -> bool:
    """
    Check if a tile position is close to an icosahedron vertex (i.e., is a pentagon).
    Pentagons always appear at the 12 fixed vertices of the icosahedron.

    Args:
        pos: Normalized position on unit sphere
        ico_vertices: List of 12 normalized icosahedron vertices
        threshold: Distance threshold (should be calculated based on hexagon size)

    Returns:
        True if the tile is a pentagon
    """
    pos_normalized = pos.normalize()

    # Check if position is close to any icosahedron vertex
    for vertex in ico_vertices:
        distance = pos_normalized.distance_to(vertex)
        if distance < threshold:
            return True

    return False


def generate_grid_points(m: int, n: int) -> list[tuple[int, int]]:
    """
    Generate grid points (u, v) for barycentric coordinates on a face.
    Based on the TypeScript implementation's grid generation.
    """
    B_grid = {"u": m, "v": n}
    C_grid = {"u": -n, "v": m + n}
    det = B_grid["u"] * C_grid["v"] - C_grid["u"] * B_grid["v"]

    minU = min(0, m, -n)
    maxU = max(0, m, -n)
    minV = min(0, n, m + n)
    maxV = max(0, n, m + n)

    grid_points = []
    eps = 0.000001

    for u in range(minU, maxU + 1):
        for v in range(minV, maxV + 1):
            wB = (u * C_grid["v"] - v * C_grid["u"]) / det
            wC = (B_grid["u"] * v - B_grid["v"] * u) / det
            wA = 1 - wB - wC
            if wA >= -eps and wB >= -eps and wC >= -eps:
                grid_points.append((u, v))

    return grid_points


def generate_goldberg_hex_sphere(
    m: int = 5, n: int = 5
) -> tuple[list[HexCoordinates], list[Vertex3D], dict[int, list[int]]]:
    """
    Generate a Goldberg polyhedron using the grid-based approach.
    m, n: Goldberg polyhedron parameters (GP(m, n))
    Returns hex coordinates, their 3D positions on the sphere, and neighbor map.

    Returns:
        - hex_coords_list: List of hex coordinates for each tile
        - positions_list: List of 3D positions for each tile
        - neighbors_map: Dictionary mapping tile index to list of neighbor tile indices

    Note: We use a simple sequential hex coordinate assignment since the reference
    implementation uses face-based coordinates, not hex grid coordinates.
    """
    ico_vertices = icosahedron_vertices()
    ico_faces = icosahedron_faces()

    B_grid = {"u": m, "v": n}
    C_grid = {"u": -n, "v": m + n}
    det = B_grid["u"] * C_grid["v"] - C_grid["u"] * B_grid["v"]

    grid_points = generate_grid_points(m, n)

    # Generate unique centers using spatial bucketing for deduplication
    unique_centers: list[Vertex3D] = []
    center_metadata: list[dict] = []  # Store face index and u, v for each center

    # Spatial bucketing for deduplication
    spatial_buckets: dict[str, list[int]] = {}
    bucket_size = 0.05

    for face_idx, face in enumerate(tqdm(ico_faces, desc="Processing faces")):
        vA = ico_vertices[face[0]]
        vB = ico_vertices[face[1]]
        vC = ico_vertices[face[2]]

        for u, v in grid_points:
            wB = (u * C_grid["v"] - v * C_grid["u"]) / det
            wC = (B_grid["u"] * v - B_grid["v"] * u) / det
            wA = 1 - wB - wC

            # Barycentric interpolation
            pos = Vertex3D(0, 0, 0)
            pos = pos.add_scaled(vA, wA)
            pos = pos.add_scaled(vB, wB)
            pos = pos.add_scaled(vC, wC)
            pos = pos.normalize()

            # Spatial bucketing for deduplication
            bx = int(pos.x / bucket_size)
            by = int(pos.y / bucket_size)
            bz = int(pos.z / bucket_size)

            found = False
            for x in range(bx - 1, bx + 2):
                for y in range(by - 1, by + 2):
                    for z in range(bz - 1, bz + 2):
                        key = f"{x},{y},{z}"
                        bucket = spatial_buckets.get(key, [])
                        for idx in bucket:
                            if unique_centers[idx].distance_to(pos) < 0.005:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

            if not found:
                new_idx = len(unique_centers)
                unique_centers.append(pos)
                center_metadata.append({"face": face_idx, "u": u, "v": v})
                key = f"{bx},{by},{bz}"
                if key not in spatial_buckets:
                    spatial_buckets[key] = []
                spatial_buckets[key].append(new_idx)

    # Convert centers to NumPy array and build KDTree for fast neighbor search
    centers_array = np.array([[v.x, v.y, v.z] for v in unique_centers])
    tree = cKDTree(centers_array)

    # Find neighbors for each center using cKDTree
    # Calculate dynamic distance threshold based on tile density
    # For a sphere, neighbor distance scales inversely with sqrt(number of tiles)
    # Base threshold of 0.4 works for ~720 tiles (world size 5)
    # Scale it down for larger world sizes
    num_tiles = len(unique_centers)
    base_threshold = 0.4
    base_tile_count = 720  # Approximate tiles for world size 5
    # Scale threshold: more tiles = smaller threshold
    neighbor_distance_threshold = base_threshold * math.sqrt(base_tile_count / max(num_tiles, 1))
    # Clamp to reasonable bounds (0.15 to 0.5)
    neighbor_distance_threshold = max(0.15, min(0.5, neighbor_distance_threshold))

    # Calculate pentagon threshold using the same formula as frontend
    # For unit sphere (radius=1), this is: sqrt(4 * 1^2 / num_tiles) = 2 / sqrt(num_tiles)
    # Multiply by 0.5 to not include neighboring hexagon tiles as pentagons
    pentagon_threshold = 0.5 * calculate_pentagon_threshold(1.0, num_tiles)

    # Get icosahedron vertices for pentagon detection
    ico_vertices = icosahedron_vertices()

    neighbors_map: dict[int, list[int]] = {}

    for i, center in enumerate(tqdm(unique_centers, desc="Finding neighbors")):
        # First, determine if this is a pentagon by checking proximity to icosahedron vertices
        # This is more reliable than counting neighbors
        is_pentagon = is_pentagon_tile(center, ico_vertices, pentagon_threshold)

        # Use cKDTree to find all neighbors within threshold
        neighbors = tree.query_ball_point(
            [center.x, center.y, center.z], r=neighbor_distance_threshold
        )
        # Remove self and calculate distances
        candidates: list[tuple[int, float]] = []
        for j in neighbors:
            if j == i:
                continue
            dist = center.distance_to(unique_centers[j])
            candidates.append((j, dist))

        # Sort by distance
        candidates.sort(key=lambda x: x[1])

        if len(candidates) < 5:
            raise ValueError(
                f"Cell {i} has only {len(candidates)} neighbors. Spatial search failed."
            )

        # Determine number of neighbors based on pentagon detection
        # Pentagons have 5 neighbors, hexagons have 6
        if is_pentagon:
            num_neighbors = 5
        else:
            num_neighbors = 6

        # Take exactly the number of neighbors determined
        # For pentagons, take 5 closest; for hexagons, take 6 closest
        neighbor_indices = [candidates[i][0] for i in range(min(num_neighbors, len(candidates)))]

        # Sort neighbors angularly around the center
        # This is critical for correct vertex calculation, especially at larger world sizes
        # The frontend expects neighbors in angular order for proper hex/pentagon rendering
        center_pos = unique_centers[i]
        center_normal = Vertex3D(center_pos.x, center_pos.y, center_pos.z).normalize()

        # Create a stable reference frame for angular sorting
        # Use a fixed world-space axis that's not too close to the center normal
        world_x = Vertex3D(1, 0, 0)
        world_y = Vertex3D(0, 1, 0)
        world_z = Vertex3D(0, 0, 1)

        # Choose reference axis
        if abs(center_normal.dot(world_x)) < 0.9:
            ref_axis = world_x
        elif abs(center_normal.dot(world_y)) < 0.9:
            ref_axis = world_y
        else:
            ref_axis = world_z

        # Create right and forward vectors for the tangent plane
        # Match frontend order: right = center_normal × ref_axis
        right = center_normal.cross(ref_axis).normalize()
        # Forward = right × center_normal (completes the orthonormal basis)
        forward = right.cross(center_normal).normalize()

        # Sort neighbors by angle around center
        def get_angle(neighbor_idx: int) -> float:
            neighbor_pos = unique_centers[neighbor_idx]
            # Direction from center to neighbor
            direction = Vertex3D(
                neighbor_pos.x - center_pos.x,
                neighbor_pos.y - center_pos.y,
                neighbor_pos.z - center_pos.z,
            )
            direction = direction.normalize()

            # Project onto tangent plane
            dir_right = direction.dot(right)
            dir_forward = direction.dot(forward)

            # Return angle using atan2 (matches frontend calculation)
            return math.atan2(dir_forward, dir_right)

        # Sort by angle
        neighbor_indices.sort(key=get_angle)

        neighbors_map[i] = neighbor_indices

    # Assign hex coordinates using BFS from first center
    hex_coords: dict[int, HexCoordinates] = {}
    assigned_coords: set[tuple[int, int, int]] = set()
    visited: set[int] = set()
    queue: deque[tuple[int, HexCoordinates]] = deque([(0, HexCoordinates(0, 0, 0))])

    directions = [
        HexCoordinates(1, 0, -1),
        HexCoordinates(1, -1, 0),
        HexCoordinates(0, -1, 1),
        HexCoordinates(-1, 0, 1),
        HexCoordinates(-1, 1, 0),
        HexCoordinates(0, 1, -1),
    ]

    while queue:
        center_idx, hex_coord = queue.popleft()
        if center_idx in visited:
            continue

        visited.add(center_idx)
        hex_coords[center_idx] = hex_coord
        coord_key = (hex_coord.q, hex_coord.r, hex_coord.s)
        assigned_coords.add(coord_key)

        # Assign coordinates to neighbors
        neighbors = neighbors_map.get(center_idx, [])
        for neighbor_idx in neighbors:
            if neighbor_idx in visited or neighbor_idx in hex_coords:
                continue

            # Try to assign an adjacent hex coordinate
            assigned = False
            for direction in directions:
                candidate = HexCoordinates(
                    hex_coord.q + direction.q,
                    hex_coord.r + direction.r,
                    hex_coord.s + direction.s,
                )

                # Check if coordinate is available
                coord_key = (candidate.q, candidate.r, candidate.s)
                if coord_key not in assigned_coords:
                    hex_coords[neighbor_idx] = candidate
                    assigned_coords.add(coord_key)
                    queue.append((neighbor_idx, candidate))
                    assigned = True
                    break

            if not assigned:
                # Fallback: assign a coordinate based on index
                offset = len(hex_coords)
                layer = int(math.sqrt(offset))
                pos_in_layer = offset - layer * layer
                if layer == 0:
                    candidate = HexCoordinates(0, 0, 0)
                else:
                    side_length = layer * 2
                    side = pos_in_layer // side_length if side_length > 0 else 0
                    pos_in_side = pos_in_layer % side_length if side_length > 0 else 0
                    dir_idx = side % 6
                    dir_coord = directions[dir_idx]
                    next_dir_coord = directions[(dir_idx + 1) % 6]
                    q = layer * dir_coord.q + pos_in_side * (next_dir_coord.q - dir_coord.q)
                    r = layer * dir_coord.r + pos_in_side * (next_dir_coord.r - dir_coord.r)
                    s = -q - r
                    candidate = HexCoordinates(q, r, s)
                hex_coords[neighbor_idx] = candidate
                coord_key = (candidate.q, candidate.r, candidate.s)
                assigned_coords.add(coord_key)
                queue.append((neighbor_idx, candidate))

    # Handle any remaining unvisited centers
    remaining = set(range(len(unique_centers))) - visited
    if remaining:
        for center_idx in tqdm(remaining, desc="Handling remaining centers"):
            # Find a visited neighbor
            neighbors = neighbors_map.get(center_idx, [])
            visited_neighbors = [n for n in neighbors if n in visited]

            if visited_neighbors:
                nn_idx = visited_neighbors[0]
                nn_coord = hex_coords[nn_idx]
                for direction in directions:
                    candidate = HexCoordinates(
                        nn_coord.q + direction.q,
                        nn_coord.r + direction.r,
                        nn_coord.s + direction.s,
                    )
                    coord_key = (candidate.q, candidate.r, candidate.s)
                    if coord_key not in assigned_coords:
                        hex_coords[center_idx] = candidate
                        assigned_coords.add(coord_key)
                        break
                else:
                    # Fallback
                    offset = len(hex_coords)
                    candidate = HexCoordinates(offset, 0, -offset)
                    hex_coords[center_idx] = candidate
                    coord_key = (candidate.q, candidate.r, candidate.s)
                    assigned_coords.add(coord_key)
            else:
                offset = len(hex_coords)
                candidate = HexCoordinates(offset, 0, -offset)
                hex_coords[center_idx] = candidate
                coord_key = (candidate.q, candidate.r, candidate.s)
                assigned_coords.add(coord_key)

    # Return in order
    hex_coords_list = []
    positions_list = []
    for i in range(len(unique_centers)):
        hex_coords_list.append(hex_coords.get(i, HexCoordinates(i, 0, -i)))
        positions_list.append(unique_centers[i])

    return hex_coords_list, positions_list, neighbors_map
