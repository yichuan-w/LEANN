#!/bin/bash

# Manual build and test script for local testing

PACKAGE=${1:-"all"}  # Default to all packages

echo "Building package: $PACKAGE"

# Ensure we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Please activate a virtual environment first"
    echo "Run: source .venv/bin/activate (or .venv/bin/activate.fish for fish shell)"
    exit 1
fi

# Install build tools
uv pip install build twine delocate auditwheel scikit-build-core cmake pybind11 numpy

build_package() {
    local package_dir=$1
    local package_name=$(basename $package_dir)

    echo "Building $package_name..."
    cd $package_dir

    # Clean previous builds
    rm -rf dist/ build/ _skbuild/

    # Build directly with pip wheel (avoids sdist issues)
    pip wheel . --no-deps -w dist

    # Repair wheel for binary packages
    if [[ "$package_name" != "leann-core" ]] && [[ "$package_name" != "leann" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # For macOS
            for wheel in dist/*.whl; do
                if [[ -f "$wheel" ]]; then
                    delocate-wheel -w dist_repaired -v "$wheel"
                fi
            done
            if [[ -d dist_repaired ]]; then
                rm -rf dist/*.whl
                mv dist_repaired/*.whl dist/
                rmdir dist_repaired
            fi
        else
            # For Linux
            for wheel in dist/*.whl; do
                if [[ -f "$wheel" ]]; then
                    auditwheel repair "$wheel" -w dist_repaired
                fi
            done
            if [[ -d dist_repaired ]]; then
                rm -rf dist/*.whl
                mv dist_repaired/*.whl dist/
                rmdir dist_repaired
            fi
        fi
    fi

    echo "Built wheels in $package_dir/dist/"
    ls -la dist/
    cd - > /dev/null
}

# Build specific package or all
if [ "$PACKAGE" == "diskann" ]; then
    build_package "packages/leann-backend-diskann"
elif [ "$PACKAGE" == "hnsw" ]; then
    build_package "packages/leann-backend-hnsw"
elif [ "$PACKAGE" == "core" ]; then
    build_package "packages/leann-core"
elif [ "$PACKAGE" == "meta" ]; then
    build_package "packages/leann"
elif [ "$PACKAGE" == "all" ]; then
    build_package "packages/leann-core"
    build_package "packages/leann-backend-hnsw"
    build_package "packages/leann-backend-diskann"
    build_package "packages/leann"
else
    echo "Unknown package: $PACKAGE"
    echo "Usage: $0 [diskann|hnsw|core|meta|all]"
    exit 1
fi

echo -e "\nBuild complete! Test with:"
echo "uv pip install packages/*/dist/*.whl"
