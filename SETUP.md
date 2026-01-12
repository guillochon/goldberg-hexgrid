# Setup Guide

This package can be used as a standalone Python package or as a git submodule.

## As a Git Submodule

To use this package as a git submodule in your project:

1. **Initialize the submodule** (from your main project root):
   ```bash
   git submodule add <repository-url> goldberg-hexgrid
   ```

2. **Install the package**:
   ```bash
   cd goldberg-hexgrid
   pip install -e .
   cd ..
   ```

3. **Update your project's dependencies**:
   Add `goldberg-hexgrid` to your `requirements.txt` or `pyproject.toml`:
   ```
   goldberg-hexgrid @ file:///${PROJECT_ROOT}/goldberg-hexgrid
   ```
   Or simply install it in editable mode as shown above.

4. **Use in your code**:
   ```python
   from goldberg_hexgrid import HexCoordinates, generate_goldberg_hex_sphere
   ```

## As a Standalone Package

1. **Clone the repository**:
   ```bash
   git clone <repository-url> goldberg-hexgrid
   cd goldberg-hexgrid
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

3. **Or install in production mode**:
   ```bash
   pip install .
   ```

## Development Setup

For development with linting and type checking:

```bash
pip install -e ".[dev]"
ruff check .
mypy goldberg_hexgrid
```
