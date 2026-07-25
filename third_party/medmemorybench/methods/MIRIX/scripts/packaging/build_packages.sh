#!/bin/bash
# Build script for Mirix client and server packages

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default package names
CLIENT_PKG_NAME="mirix-client"
SERVER_PKG_NAME="mirix-server"
VERSION=""

# Parse command line arguments
usage() {
    echo "Usage: $0 -v VERSION [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -v, --version VERSION     Package version (e.g., 0.6.5, 1.0.0)"
    echo ""
    echo "Options:"
    echo "  -c, --client-name NAME    Client package name (default: mirix-client)"
    echo "  -s, --server-name NAME    Server package name (default: mirix-server)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -v 1.0.0"
    echo "  $0 -v 2.0.0 -c jl-ecms-client -s jl-ecms-server"
    echo "  $0 --version 0.6.6 --client-name custom-client --server-name custom-server"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -c|--client-name)
            CLIENT_PKG_NAME="$2"
            shift 2
            ;;
        -s|--server-name)
            SERVER_PKG_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version is required${NC}"
    echo ""
    usage
fi

# Function to restore files on exit
restore_files() {
    if [ -f "pyproject.toml.tmp.hidden" ]; then
        echo -e "${BLUE}Restoring pyproject.toml...${NC}"
        mv pyproject.toml.tmp.hidden pyproject.toml
    fi
    if [ -f "pyproject.toml.backup" ]; then
        mv pyproject.toml.backup pyproject.toml
    fi
    if [ -f "mirix/__init__.py.backup" ]; then
        echo -e "${BLUE}Restoring mirix/__init__.py...${NC}"
        mv mirix/__init__.py.backup mirix/__init__.py
    fi
}

# Set trap to restore files on exit (success or failure)
trap restore_files EXIT

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  Mirix Package Build Script${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Version:        ${GREEN}${VERSION}${NC}"
echo -e "  Client package: ${GREEN}${CLIENT_PKG_NAME}${NC}"
echo -e "  Server package: ${GREEN}${SERVER_PKG_NAME}${NC}"
echo ""

# Get script directory (scripts/packaging/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up two levels: packaging/ -> scripts/ -> project_root/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# Update version in source files
echo -e "${BLUE}[0/6] Updating version in source files...${NC}"

# Backup __init__.py
cp mirix/__init__.py mirix/__init__.py.backup

# Update version in mirix/__init__.py
sed -i.tmp "s/^__version__ = .*/__version__ = \"${VERSION}\"/" mirix/__init__.py
rm -f mirix/__init__.py.tmp
echo -e "${GREEN}✓ Updated mirix/__init__.py to version ${VERSION}${NC}"

# Backup pyproject.toml (before we move it later)
cp pyproject.toml pyproject.toml.backup

# Update version in pyproject.toml
sed -i.tmp "s/^version = .*/version = \"${VERSION}\"/" pyproject.toml
rm -f pyproject.toml.tmp
echo -e "${GREEN}✓ Updated pyproject.toml to version ${VERSION}${NC}"
echo ""

# Clean previous builds
echo -e "${BLUE}[1/6] Cleaning previous builds...${NC}"
# Convert package names to underscore format for egg-info directories
CLIENT_EGG_INFO=$(echo "${CLIENT_PKG_NAME}" | tr '-' '_')
SERVER_EGG_INFO=$(echo "${SERVER_PKG_NAME}" | tr '-' '_')
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf "${CLIENT_EGG_INFO}.egg-info"
rm -rf "${SERVER_EGG_INFO}.egg-info"
echo -e "${GREEN}✓ Cleaned${NC}"
echo ""

# Temporarily rename pyproject.toml to prevent it from overriding setup scripts
echo -e "${BLUE}[2/6] Temporarily moving pyproject.toml...${NC}"
if [ -f "pyproject.toml" ]; then
    mv pyproject.toml pyproject.toml.tmp.hidden
    echo -e "${GREEN}✓ Moved pyproject.toml${NC}"
else
    echo -e "${BLUE}  pyproject.toml already moved${NC}"
fi
echo ""

# Install build dependencies
echo -e "${BLUE}[3/6] Installing build dependencies...${NC}"
pip install --upgrade setuptools wheel twine
echo -e "${GREEN}✓ Build tools ready${NC}"
echo ""

# Build client package
echo -e "${BLUE}[4/6] Building ${CLIENT_PKG_NAME} package...${NC}"
python scripts/packaging/setup_client.py --package-name "${CLIENT_PKG_NAME}" --version "${VERSION}" sdist bdist_wheel
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Client package built successfully${NC}"
    # Convert package name to underscore format for file matching
    CLIENT_FILE_NAME=$(echo "${CLIENT_PKG_NAME}" | tr '-' '_')
    echo -e "   Version: ${GREEN}${VERSION}${NC}"
else
    echo -e "${RED}✗ Client package build failed${NC}"
    exit 1
fi
echo ""

# Build server package
echo -e "${BLUE}[5/6] Building ${SERVER_PKG_NAME} package...${NC}"
python scripts/packaging/setup_server.py --package-name "${SERVER_PKG_NAME}" --version "${VERSION}" sdist bdist_wheel
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Server package built successfully${NC}"
    # Convert package name to underscore format for file matching
    SERVER_FILE_NAME=$(echo "${SERVER_PKG_NAME}" | tr '-' '_')
    echo -e "   Version: ${GREEN}${VERSION}${NC}"
else
    echo -e "${RED}✗ Server package build failed${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${BLUE}[6/6] Build Summary${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "Built packages in: ${GREEN}dist/${NC}"
echo ""
echo -e "Client Package (${CLIENT_PKG_NAME}):"
ls -lh dist/${CLIENT_FILE_NAME}-* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "Server Package (${SERVER_PKG_NAME}):"
ls -lh dist/${SERVER_FILE_NAME}-* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✓ All packages built successfully!${NC}"
echo ""
echo "To install locally:"
echo -e "  ${BLUE}pip install dist/${CLIENT_FILE_NAME}-${VERSION}-py3-none-any.whl${NC}"
echo -e "  ${BLUE}pip install dist/${SERVER_FILE_NAME}-${VERSION}-py3-none-any.whl${NC}"
echo ""
echo "To publish to PyPI:"
echo -e "  ${BLUE}twine upload dist/${CLIENT_FILE_NAME}-${VERSION}*${NC}"
echo -e "  ${BLUE}twine upload dist/${SERVER_FILE_NAME}-${VERSION}*${NC}"
echo ""
echo "To test packages:"
echo -e "  ${BLUE}twine check dist/*${NC}"
echo ""

# Note: Files are automatically restored by the EXIT trap

