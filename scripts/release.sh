#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.1"
    exit 1
fi

VERSION=$1

# Update version
./scripts/bump_version.sh $VERSION

# Commit and push
git add . && git commit -m "chore: bump version to $VERSION" && git push

# Create release (triggers CI)
gh release create v$VERSION --generate-notes
