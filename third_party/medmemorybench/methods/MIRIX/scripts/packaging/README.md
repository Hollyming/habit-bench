# Mirix Package Building Guide

This directory (`scripts/packaging/`) contains all scripts and documentation for building and publishing the Mirix client and server packages to PyPI.

## Directory Contents

This packaging directory contains:
- `setup_client.py` - Client package setup script
- `setup_server.py` - Server package setup script  
- `build_packages.sh` - Automated build script
- `requirements_client.txt` - Client-only dependencies
- `README.md` - This guide
- `PACKAGE_STRUCTURE.md` - Complete directory structure documentation
- `PACKAGING_SUMMARY.md` - Implementation summary

## Overview

Mirix is split into two separate PyPI packages:
1. **mirix-client** - Lightweight client library
2. **mirix-server** - Full server with agents, database, and LLM integrations

## Quick Start

### Build Both Packages

```bash
# From project root
cd <YOUR_MIRIX_CODE_FOLDER_FULL_PATH>
./scripts/packaging/build_packages.sh

# Or from packaging directory
cd scripts/packaging
./build_packages.sh
```

This will:
- Clean previous builds
- Install build tools
- Build both client and server packages
- Create distribution files in `dist/`

### Build Individual Packages

#### Client Package Only
```bash
# From project root
python scripts/packaging/setup_client.py sdist bdist_wheel

# Or from packaging directory
cd scripts/packaging
python setup_client.py sdist bdist_wheel
```

#### Server Package Only
```bash
# From project root
python scripts/packaging/setup_server.py sdist bdist_wheel

# Or from packaging directory
cd scripts/packaging
python setup_server.py sdist bdist_wheel
```

## Scripts Reference

### `setup_client.py`
Builds the **mirix-client** package.

**Includes:**
- `mirix/client/` - Remote client implementation
- `mirix/schemas/` - Pydantic schemas
- `mirix/helpers/` - Utility helpers

**Dependencies:**
- requests, httpx (HTTP client)
- pydantic (validation)
- python-dotenv, rich, pytz

**Output:**
- `dist/mirix-client-<version>.tar.gz`
- `dist/mirix_client-<version>-py3-none-any.whl`

---

### `setup_server.py`
Builds the **mirix-server** package.

**Includes:**
- `mirix/server/` - FastAPI REST API
- `mirix/agent/` - All agent implementations
- `mirix/database/` - PostgreSQL + Redis clients
- `mirix/orm/` - SQLAlchemy models
- `mirix/services/` - Business logic
- `mirix/llm_api/` - LLM integrations
- `mirix/functions/` - Function tools
- `mirix/prompts/` - System prompts
- `mirix/queue/` - Message queue
- All other server-side code

**Dependencies:**
- FastAPI, uvicorn (REST API)
- SQLAlchemy, psycopg2, redis (Database)
- openai, google-genai, anthropic (LLMs)
- 50+ other packages

**Output:**
- `dist/mirix-server-<version>.tar.gz`
- `dist/mirix_server-<version>-py3-none-any.whl`

---

### `build_packages.sh`
Automated build script for both packages.

**Features:**
- ✅ Cleans previous builds
- ✅ Installs build dependencies
- ✅ Builds both packages
- ✅ Shows build summary with file sizes
- ✅ Provides installation and publishing commands

**Usage:**
```bash
# From project root
./scripts/packaging/build_packages.sh

# Or from packaging directory
cd scripts/packaging && ./build_packages.sh
```

---

## Installation Testing

### Test Client Package Locally
```bash
# Create test environment
python -m venv test_client_env
source test_client_env/bin/activate  # On Windows: test_client_env\Scripts\activate

# Install from wheel
pip install dist/mirix_client-0.5.0-py3-none-any.whl

# Test import
python -c "from mirix.client import MirixClient; print('✓ Client package works!')"
```

### Test Server Package Locally
```bash
# Create test environment
python -m venv test_server_env
source test_server_env/bin/activate

# Install from wheel
pip install dist/mirix_server-0.5.0-py3-none-any.whl

# Test import
python -c "from mirix.server.rest_api import app; print('✓ Server package works!')"
```

---

## Publishing to PyPI

### Prerequisites

1. **Install Twine**
```bash
pip install twine
```

2. **Configure PyPI Credentials**
```bash
# Option A: Use token (recommended)
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...your-token...

# Option B: Use .pypirc file
cat > ~/.pypirc << EOF
[pypi]
username = __token__
password = pypi-...your-token...

[testpypi]
username = __token__
password = pypi-...your-token...
EOF
```

### Publish to Test PyPI (Recommended First)

```bash
# Upload client
twine upload --repository testpypi dist/mirix-client-*

# Upload server
twine upload --repository testpypi dist/mirix-server-*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ mirix-client
pip install --index-url https://test.pypi.org/simple/ mirix-server
```

### Publish to Production PyPI

```bash
# Check packages first
twine check dist/*

# Upload client
twine upload dist/mirix-client-*

# Upload server
twine upload dist/mirix-server-*
```

---

## Version Management

Both packages share the same version number from `mirix/__init__.py`:

```python
# mirix/__init__.py
__version__ = "0.5.0"
```

### Update Version

1. Edit `mirix/__init__.py`
2. Update `__version__` string
3. Rebuild packages
4. Publish with new version

### Version Compatibility

Client and server should maintain version compatibility:

```
mirix-client==0.5.0  ←→  mirix-server==0.5.x
mirix-client==0.6.0  ←→  mirix-server==0.6.x
```

**Rule**: Major and minor versions must match, patch versions can differ.

---

## Package Structure

### Client Package (~5 MB)

```
mirix-client/
├── mirix/
│   ├── client/          # HTTP client
│   ├── schemas/         # Pydantic models
│   └── helpers/         # Utilities
└── requirements_client.txt
```

### Server Package (~150 MB)

```
mirix-server/
├── mirix/
│   ├── server/          # FastAPI REST API
│   ├── agent/           # Agent implementations
│   ├── database/        # PostgreSQL + Redis
│   ├── orm/             # SQLAlchemy models
│   ├── services/        # Business logic
│   ├── llm_api/         # LLM integrations
│   ├── functions/       # Function tools
│   ├── prompts/         # System prompts
│   ├── queue/           # Message queue
│   ├── client/          # (included for internal use)
│   ├── schemas/         # (included for internal use)
│   └── helpers/         # (included for internal use)
└── requirements.txt
```

---

## Troubleshooting

### Build Errors

#### "No module named 'setuptools'"
```bash
pip install --upgrade setuptools wheel
```

#### "Unable to find version string"
Check that `mirix/__init__.py` contains:
```python
__version__ = "X.Y.Z"
```

#### "Package directory not found"
Make sure you're running from the project root:
```bash
cd /Users/jliao2/src/MIRIX_Intuit
python scripts/setup_client.py bdist_wheel
```

### Upload Errors

#### "Invalid credentials"
```bash
# Check your token
echo $TWINE_PASSWORD

# Re-export with correct token
export TWINE_PASSWORD=pypi-your-actual-token
```

#### "Filename already exists"
You're trying to upload a version that already exists. Update the version number in `mirix/__init__.py`.

#### "Invalid distribution file"
```bash
# Check package validity
twine check dist/*
```

### Installation Errors

#### "No matching distribution found"
- Make sure the package was uploaded successfully
- Check you're using the correct package name: `mirix-client` or `mirix-server`
- For TestPyPI, use `--index-url https://test.pypi.org/simple/`

#### "Dependency conflicts"
```bash
# Use a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install mirix-client  # or mirix-server
```

---

## Best Practices

### Before Publishing

1. ✅ **Run tests**: `pytest tests/`
2. ✅ **Check linting**: `ruff check mirix/`
3. ✅ **Update version**: Edit `mirix/__init__.py`
4. ✅ **Update changelog**: Document changes
5. ✅ **Build packages**: `./scripts/packaging/build_packages.sh`
6. ✅ **Validate packages**: `twine check dist/*`
7. ✅ **Test on TestPyPI**: Upload and install
8. ✅ **Publish to PyPI**: Upload production packages

### Semantic Versioning

Follow [SemVer](https://semver.org/):
- **MAJOR** (0.x.0): Breaking changes
- **MINOR** (x.1.0): New features (backward compatible)
- **PATCH** (x.x.1): Bug fixes

Examples:
- `0.5.0` → `0.5.1` (bug fix)
- `0.5.1` → `0.6.0` (new feature)
- `0.6.0` → `1.0.0` (breaking change)

### Release Checklist

- [ ] Update version in `mirix/__init__.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Build packages: `./scripts/packaging/build_packages.sh`
- [ ] Check packages: `twine check dist/*`
- [ ] Upload to TestPyPI
- [ ] Test installation from TestPyPI
- [ ] Upload to production PyPI
- [ ] Create GitHub release
- [ ] Update documentation
- [ ] Announce on Discord/Twitter

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install setuptools wheel twine
      
      - name: Build packages
        run: |
          ./scripts/packaging/build_packages.sh
      
      - name: Check packages
        run: |
          twine check dist/*
      
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: |
          twine upload dist/*
```

---

## Support

For questions or issues with package building:

- **GitHub Issues**: https://github.com/Mirix-AI/MIRIX/issues
- **Documentation**: https://docs.mirix.io
- **Email**: support@mirix.io

---

**Last Updated**: October 31, 2025  
**Mirix Version**: 0.5.0

