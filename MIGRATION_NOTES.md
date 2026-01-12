# Migration Notes

This package was extracted from `backend/hex_grid.py` and `backend/goldberg_polyhedron.py`.

## Changes Made

1. **Package Structure**: Created a proper Python package with:
   - `goldberg_hexgrid/` - Main package directory
   - `__init__.py` - Exports all public APIs
   - `hex_grid.py` - Hex coordinate system utilities
   - `goldberg_polyhedron.py` - Goldberg polyhedron generation (updated to use relative import)

2. **Import Changes**: 
   - Changed `from hex_grid import HexCoordinates` to relative import `from .hex_grid import HexCoordinates` in `goldberg_polyhedron.py`
   - Updated `backend/world_generator.py` to use `from goldberg_hexgrid import ...`

3. **Package Metadata**:
   - `pyproject.toml` - Modern Python package configuration
   - `README.md` - Package documentation
   - `LICENSE` - MIT License
   - `.gitignore` - Standard Python gitignore
   - `.gitattributes` - Line ending normalization

## Next Steps

1. **Initialize as Git Repository**:
   ```bash
   cd goldberg-hexgrid
   git init
   git add .
   git commit -m "Initial commit: Goldberg hexgrid package"
   ```

2. **Create Remote Repository**:
   - Create a new repository on GitHub/GitLab/etc.
   - Add the remote and push:
   ```bash
   git remote add origin <repository-url>
   git push -u origin main
   ```

3. **Add as Submodule to Main Project**:
   ```bash
   cd ..  # Back to main project root
   git submodule add <repository-url> goldberg-hexgrid
   ```

4. **Add to Backend Dependencies**:
   
   Add to `backend/pyproject.toml`:
   ```toml
   dependencies = [
       # ... existing dependencies ...
       "goldberg-hexgrid @ file:///${PROJECT_ROOT}/../goldberg-hexgrid",
   ]
   ```
   
   Note: `${PROJECT_ROOT}` in uv resolves to the directory containing the `pyproject.toml` file (the `backend/` directory), so `../goldberg-hexgrid` goes up one level to find the package.

5. **Sync dependencies with uv**:
   ```bash
   cd backend
   uv sync
   ```
   
   This will install the package and update `uv.lock`.

6. **Remove Old Files** (after verifying everything works):
   ```bash
   # From main project root
   rm backend/hex_grid.py
   rm backend/goldberg_polyhedron.py
   ```

## Testing

After installation, verify the package works:

```python
from goldberg_hexgrid import HexCoordinates, generate_goldberg_hex_sphere

# Test hex coordinates
coord = HexCoordinates(1, 2, -3)
print(coord)

# Test Goldberg polyhedron generation
hex_coords, positions, neighbors = generate_goldberg_hex_sphere(m=5, n=5)
print(f"Generated {len(hex_coords)} tiles")
```
