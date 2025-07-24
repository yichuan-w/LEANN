#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

NEW_VERSION=$1

# Update all pyproject.toml files
# Use different sed syntax for macOS vs Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' "s/version = \".*\"/version = \"$NEW_VERSION\"/" packages/*/pyproject.toml
else
    sed -i "s/version = \".*\"/version = \"$NEW_VERSION\"/" packages/*/pyproject.toml
fi

echo "Version updated to $NEW_VERSION" 