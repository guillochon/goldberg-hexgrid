# Setup Guide

This package can be used as a standalone Python package or as a git submodule.

## As a Git Submodule (Recommended with uv)

To use this package as a git submodule in your project:

1. **Initialize the submodule** (from your main project root):
   ```bash
   git submodule add <repository-url> goldberg-hexgrid
   ```

2. **Add to your project's dependencies** (recommended):
   
   Add to your `pyproject.toml` (e.g., in `backend/pyproject.toml`):
   
   If `pyproject.toml` is in the same directory as `goldberg-hexgrid`:
   ```toml
   dependencies = [
       # ... other dependencies ...
       "goldberg-hexgrid @ file:///${PROJECT_ROOT}/goldberg-hexgrid",
   ]
   ```
   
   If `pyproject.toml` is in a subdirectory (e.g., `backend/pyproject.toml`):
   ```toml
   dependencies = [
       # ... other dependencies ...
       "goldberg-hexgrid @ file:///${PROJECT_ROOT}/../goldberg-hexgrid",
   ]
   ```
   
   Note: `${PROJECT_ROOT}` in uv resolves to the directory containing the `pyproject.toml` file.

3. **Sync dependencies with uv**:
   ```bash
   cd backend  # or wherever your pyproject.toml is
   uv sync
   ```

4. **Or install directly with uv** (alternative):
   ```bash
   cd goldberg-hexgrid
   uv pip install -e .
   ```

5. **Use in your code**:
   ```python
   from goldberg_hexgrid import HexCoordinates, generate_goldberg_hex_sphere
   ```

## As a Standalone Package

1. **Clone the repository**:
   ```bash
   git clone <repository-url> goldberg-hexgrid
   cd goldberg-hexgrid
   ```

2. **Install in development mode with uv**:
   ```bash
   uv pip install -e .
   ```

3. **Or install in production mode**:
   ```bash
   uv pip install .
   ```

## Development Setup

For development with linting and type checking:

```bash
uv pip install -e ".[dev]"
ruff check .
mypy goldberg_hexgrid
```
