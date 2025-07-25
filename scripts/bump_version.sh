#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

NEW_VERSION=$1

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Update all pyproject.toml files
echo "Updating versions in $PROJECT_ROOT/packages/"

# Use different sed syntax for macOS vs Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Update version fields
    find "$PROJECT_ROOT/packages" -name "pyproject.toml" -exec sed -i '' "s/version = \".*\"/version = \"$NEW_VERSION\"/" {} \;
    # Update leann-core dependencies
    find "$PROJECT_ROOT/packages" -name "pyproject.toml" -exec sed -i '' "s/leann-core==[0-9.]*/leann-core==$NEW_VERSION/" {} \;
else
    # Update version fields
    find "$PROJECT_ROOT/packages" -name "pyproject.toml" -exec sed -i "s/version = \".*\"/version = \"$NEW_VERSION\"/" {} \;
    # Update leann-core dependencies
    find "$PROJECT_ROOT/packages" -name "pyproject.toml" -exec sed -i "s/leann-core==[0-9.]*/leann-core==$NEW_VERSION/" {} \;
fi

echo "✅ Version updated to $NEW_VERSION"
echo "✅ Dependencies updated to use leann-core==$NEW_VERSION" 