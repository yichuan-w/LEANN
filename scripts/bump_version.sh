#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

NEW_VERSION=$1

# Update all pyproject.toml files
sed -i '' "s/version = \".*\"/version = \"$NEW_VERSION\"/" packages/*/pyproject.toml

echo "Version updated to $NEW_VERSION" 