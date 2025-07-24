#!/bin/bash

# Manual upload script for testing

TARGET=${1:-"test"}  # Default to test pypi

if [ "$TARGET" != "test" ] && [ "$TARGET" != "prod" ]; then
    echo "Usage: $0 [test|prod]"
    exit 1
fi

# Check for built packages
if ! ls packages/*/dist/*.whl >/dev/null 2>&1; then
    echo "No built packages found. Run ./scripts/build_and_test.sh first"
    exit 1
fi

if [ "$TARGET" == "test" ]; then
    echo "Uploading to Test PyPI..."
    twine upload --repository testpypi packages/*/dist/*
else
    echo "Uploading to PyPI..."
    echo "Are you sure? (y/N)"
    read -r response
    if [ "$response" == "y" ]; then
        twine upload packages/*/dist/*
    else
        echo "Cancelled"
    fi
fi 