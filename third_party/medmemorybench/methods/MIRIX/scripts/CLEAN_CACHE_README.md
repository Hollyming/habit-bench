# Mirix Cache Cleanup Scripts

This directory contains scripts to clean Python cache files from the Mirix project.

## Available Scripts

### 1. Python Script (Cross-Platform) - **RECOMMENDED**

**File:** `clean_cache.py`

**Usage:**
```bash
# From project root
python3 scripts/clean_cache.py

# Or make it executable and run directly
chmod +x scripts/clean_cache.py
./scripts/clean_cache.py
```

**Features:**
- ✅ Works on Windows, macOS, and Linux
- ✅ Color-coded output
- ✅ Shows detailed statistics
- ✅ Safe error handling

### 2. Bash Script (Unix/Linux/macOS)

**File:** `clean_cache.sh`

**Usage:**
```bash
# From project root
./scripts/clean_cache.sh

# Or
bash scripts/clean_cache.sh
```

**Features:**
- ✅ Fast execution
- ✅ Color-coded output
- ✅ Works on Unix-like systems

## What Gets Cleaned?

Both scripts remove the following cache files and directories:

### Python Bytecode
- `__pycache__/` directories
- `*.pyc` files (compiled Python)
- `*.pyo` files (optimized Python)

### Testing
- `.pytest_cache/` directories
- `.coverage` files
- `.coverage.*` files
- `htmlcov/` directories (coverage reports)

### Type Checking
- `.mypy_cache/` directories

### Package Distribution
- `.eggs/` directories
- `*.egg-info/` directories
- `build/` directories
- `dist/` directories

### Linting
- `.ruff_cache/` directories

### Jupyter Notebooks
- `.ipynb_checkpoints/` directories

## When to Run

Run the cleanup script:

1. **Before committing code** - Ensure no cache files are accidentally committed
2. **After switching branches** - Clean stale cache from different code versions
3. **When experiencing weird import errors** - Stale cache can cause issues
4. **Before building distributions** - Ensure clean build
5. **To free up disk space** - Cache files can accumulate over time

## Example Output

```
========================================
Mirix Python Cache Cleanup
========================================

Project root: /Users/username/MIRIX_Intuit

Cleaning __pycache__ directories...
✓ Removed 454 __pycache__ directories
Cleaning .pyc files...
✓ Removed 0 .pyc files
...

========================================
✓ Python cache cleanup complete!
========================================

Total items removed: 458

Cache types cleaned:
  ✓ __pycache__ directories: 454
  ✓ build directories: 2
  ✓ dist directories: 2
  ...
```

## Safety

- Both scripts use safe removal methods with error handling
- They only remove cache files, never source code
- Failed removals generate warnings but don't stop the script
- All operations are non-destructive to actual code

## Troubleshooting

### Permission Errors
If you get permission errors, you may need to run with elevated privileges:
```bash
sudo python3 scripts/clean_cache.py
```

### Script Not Executable
Make the script executable:
```bash
chmod +x scripts/clean_cache.py
chmod +x scripts/clean_cache.sh
```

## Integration with Git

Consider adding a git hook to run cleanup before commits:

```bash
# .git/hooks/pre-commit
#!/bin/bash
python3 scripts/clean_cache.py
```

## Notes

- The Python script (`clean_cache.py`) is recommended for cross-platform use
- Both scripts are safe to run multiple times
- They will skip removing files that don't exist
- Run from any directory - they automatically find the project root

