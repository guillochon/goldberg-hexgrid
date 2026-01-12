from dataclasses import dataclass


@dataclass
class HexCoordinates:
    q: int
    r: int
    s: int

    def __hash__(self):
        return hash((self.q, self.r, self.s))

    def __eq__(self, other):
        if not isinstance(other, HexCoordinates):
            return False
        return self.q == other.q and self.r == other.r and self.s == other.s


def hex_distance(a: HexCoordinates, b: HexCoordinates) -> int:
    return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) // 2


def hex_neighbors(hex_coord: HexCoordinates) -> list[HexCoordinates]:
    return [
        HexCoordinates(hex_coord.q + 1, hex_coord.r, hex_coord.s - 1),
        HexCoordinates(hex_coord.q + 1, hex_coord.r - 1, hex_coord.s),
        HexCoordinates(hex_coord.q, hex_coord.r - 1, hex_coord.s + 1),
        HexCoordinates(hex_coord.q - 1, hex_coord.r, hex_coord.s + 1),
        HexCoordinates(hex_coord.q - 1, hex_coord.r + 1, hex_coord.s),
        HexCoordinates(hex_coord.q, hex_coord.r + 1, hex_coord.s - 1),
    ]


def generate_hex_sphere(radius: int = 3) -> list[HexCoordinates]:
    hexes = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            s = -q - r
            hexes.append(HexCoordinates(q, r, s))
    return hexes


def hex_ring(center: HexCoordinates, radius: int) -> list[HexCoordinates]:
    if radius == 0:
        return [center]

    results = []
    hex_coord = HexCoordinates(
        center.q + hex_neighbors(center)[4].q * radius,
        center.r + hex_neighbors(center)[4].r * radius,
        center.s + hex_neighbors(center)[4].s * radius,
    )

    for i in range(6):
        for j in range(radius):
            results.append(hex_coord)
            hex_coord = hex_neighbors(hex_coord)[i]
    return results
