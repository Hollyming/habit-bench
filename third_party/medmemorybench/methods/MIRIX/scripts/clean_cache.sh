#!/bin/bash
# Mirix Python Cache Cleanup Script
# This script removes all Python cache files and directories from the project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Mirix Python Cache Cleanup${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Get the project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo -e "${YELLOW}Project root: ${PROJECT_ROOT}${NC}"
echo ""

cd "$PROJECT_ROOT"

# Function to count items before deletion
count_items() {
    local pattern=$1
    local name=$2
    local count=0
    
    if [ "$pattern" = "-name" ]; then
        count=$(find . -type d -name "$name" 2>/dev/null | wc -l)
    else
        count=$(find . -type f -name "$name" 2>/dev/null | wc -l)
    fi
    
    echo "$count"
}

# 1. Remove __pycache__ directories
echo -e "${YELLOW}Cleaning __pycache__ directories...${NC}"
PYCACHE_COUNT=$(count_items "-name" "__pycache__")
if [ "$PYCACHE_COUNT" -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $PYCACHE_COUNT __pycache__ directories${NC}"
else
    echo -e "${GREEN}✓ No __pycache__ directories found${NC}"
fi

# 2. Remove .pyc files
echo -e "${YELLOW}Cleaning .pyc files...${NC}"
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)
if [ "$PYC_COUNT" -gt 0 ]; then
    find . -type f -name "*.pyc" -delete
    echo -e "${GREEN}✓ Removed $PYC_COUNT .pyc files${NC}"
else
    echo -e "${GREEN}✓ No .pyc files found${NC}"
fi

# 3. Remove .pyo files
echo -e "${YELLOW}Cleaning .pyo files...${NC}"
PYO_COUNT=$(find . -type f -name "*.pyo" 2>/dev/null | wc -l)
if [ "$PYO_COUNT" -gt 0 ]; then
    find . -type f -name "*.pyo" -delete
    echo -e "${GREEN}✓ Removed $PYO_COUNT .pyo files${NC}"
else
    echo -e "${GREEN}✓ No .pyo files found${NC}"
fi

# 4. Remove .pytest_cache directories
echo -e "${YELLOW}Cleaning .pytest_cache directories...${NC}"
PYTEST_COUNT=$(count_items "-name" ".pytest_cache")
if [ "$PYTEST_COUNT" -gt 0 ]; then
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $PYTEST_COUNT .pytest_cache directories${NC}"
else
    echo -e "${GREEN}✓ No .pytest_cache directories found${NC}"
fi

# 5. Remove .mypy_cache directories
echo -e "${YELLOW}Cleaning .mypy_cache directories...${NC}"
MYPY_COUNT=$(count_items "-name" ".mypy_cache")
if [ "$MYPY_COUNT" -gt 0 ]; then
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $MYPY_COUNT .mypy_cache directories${NC}"
else
    echo -e "${GREEN}✓ No .mypy_cache directories found${NC}"
fi

# 6. Remove .coverage files
echo -e "${YELLOW}Cleaning .coverage files...${NC}"
COVERAGE_COUNT=$(find . -type f -name ".coverage" 2>/dev/null | wc -l)
if [ "$COVERAGE_COUNT" -gt 0 ]; then
    find . -type f -name ".coverage" -delete
    echo -e "${GREEN}✓ Removed $COVERAGE_COUNT .coverage files${NC}"
else
    echo -e "${GREEN}✓ No .coverage files found${NC}"
fi

# 7. Remove .coverage.* files
echo -e "${YELLOW}Cleaning .coverage.* files...${NC}"
COVERAGE_SUB_COUNT=$(find . -type f -name ".coverage.*" 2>/dev/null | wc -l)
if [ "$COVERAGE_SUB_COUNT" -gt 0 ]; then
    find . -type f -name ".coverage.*" -delete
    echo -e "${GREEN}✓ Removed $COVERAGE_SUB_COUNT .coverage.* files${NC}"
else
    echo -e "${GREEN}✓ No .coverage.* files found${NC}"
fi

# 8. Remove htmlcov directories (coverage HTML reports)
echo -e "${YELLOW}Cleaning htmlcov directories...${NC}"
HTMLCOV_COUNT=$(count_items "-name" "htmlcov")
if [ "$HTMLCOV_COUNT" -gt 0 ]; then
    find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $HTMLCOV_COUNT htmlcov directories${NC}"
else
    echo -e "${GREEN}✓ No htmlcov directories found${NC}"
fi

# 9. Remove .eggs directories
echo -e "${YELLOW}Cleaning .eggs directories...${NC}"
EGGS_COUNT=$(count_items "-name" ".eggs")
if [ "$EGGS_COUNT" -gt 0 ]; then
    find . -type d -name ".eggs" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $EGGS_COUNT .eggs directories${NC}"
else
    echo -e "${GREEN}✓ No .eggs directories found${NC}"
fi

# 10. Remove *.egg-info directories
echo -e "${YELLOW}Cleaning *.egg-info directories...${NC}"
EGG_INFO_COUNT=$(find . -type d -name "*.egg-info" 2>/dev/null | wc -l)
if [ "$EGG_INFO_COUNT" -gt 0 ]; then
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $EGG_INFO_COUNT *.egg-info directories${NC}"
else
    echo -e "${GREEN}✓ No *.egg-info directories found${NC}"
fi

# 11. Remove build directories
echo -e "${YELLOW}Cleaning build directories...${NC}"
BUILD_COUNT=$(count_items "-name" "build")
if [ "$BUILD_COUNT" -gt 0 ]; then
    find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $BUILD_COUNT build directories${NC}"
else
    echo -e "${GREEN}✓ No build directories found${NC}"
fi

# 12. Remove dist directories
echo -e "${YELLOW}Cleaning dist directories...${NC}"
DIST_COUNT=$(count_items "-name" "dist")
if [ "$DIST_COUNT" -gt 0 ]; then
    find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $DIST_COUNT dist directories${NC}"
else
    echo -e "${GREEN}✓ No dist directories found${NC}"
fi

# 13. Remove .ruff_cache directories
echo -e "${YELLOW}Cleaning .ruff_cache directories...${NC}"
RUFF_COUNT=$(count_items "-name" ".ruff_cache")
if [ "$RUFF_COUNT" -gt 0 ]; then
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $RUFF_COUNT .ruff_cache directories${NC}"
else
    echo -e "${GREEN}✓ No .ruff_cache directories found${NC}"
fi

# 14. Remove .ipynb_checkpoints directories (Jupyter)
echo -e "${YELLOW}Cleaning .ipynb_checkpoints directories...${NC}"
IPYNB_COUNT=$(count_items "-name" ".ipynb_checkpoints")
if [ "$IPYNB_COUNT" -gt 0 ]; then
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    echo -e "${GREEN}✓ Removed $IPYNB_COUNT .ipynb_checkpoints directories${NC}"
else
    echo -e "${GREEN}✓ No .ipynb_checkpoints directories found${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Python cache cleanup complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "Cache types cleaned:"
echo -e "  • __pycache__ directories"
echo -e "  • .pyc compiled files"
echo -e "  • .pyo optimized files"
echo -e "  • .pytest_cache directories"
echo -e "  • .mypy_cache directories"
echo -e "  • .coverage files"
echo -e "  • htmlcov directories"
echo -e "  • .eggs directories"
echo -e "  • *.egg-info directories"
echo -e "  • build directories"
echo -e "  • dist directories"
echo -e "  • .ruff_cache directories"
echo -e "  • .ipynb_checkpoints directories"
echo ""

