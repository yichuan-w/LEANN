"""
Unit tests for leann.analysis.CodeAnalyzer.
Tests the core metadata extraction logic (imports, skeleton, main detection)
independent of the chunking mechanism.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add paths for local modules
try:
    TEST_FILE_PATH = Path(__file__).resolve()
    LEANN_FORK_DIR = TEST_FILE_PATH.parent.parent

    LEANN_CORE_SRC = LEANN_FORK_DIR / "packages" / "leann-core" / "src"
    ASTCHUNK_SRC = LEANN_FORK_DIR / "packages" / "astchunk-leann" / "src"
    APPS_DIR = LEANN_FORK_DIR / "apps"

    sys.path.insert(0, str(LEANN_CORE_SRC))
    sys.path.insert(0, str(ASTCHUNK_SRC))
    sys.path.insert(0, str(APPS_DIR))
except Exception:
    pass

# Mock Backend Dependencies causing import issues in some environments
sys.modules["leann_backend_hnsw"] = MagicMock()
sys.modules["leann_backend_hnsw.convert_to_csr"] = MagicMock()
sys.modules["leann_backend_faiss"] = MagicMock()

from leann.analysis import TREE_SITTER_AVAILABLE, CodeAnalyzer  # noqa: E402


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not installed")
class TestCodeAnalyzerPython:
    """Test CodeAnalyzer with Python code."""

    def setup_method(self):
        self.analyzer = CodeAnalyzer("python")

    def test_imports_extraction(self):
        code = """
import os
import sys
from typing import List, Optional
from .local import submodule
import numpy as np
        """
        result = self.analyzer.analyze(code, "test.py")
        imports = result["imports"]

        # Test basic presence
        assert "os" in imports
        assert "sys" in imports
        assert "typing" in imports
        assert len(imports) >= 3

    def test_main_module_detection_filename(self):
        assert self.analyzer._detect_main_module(None, "", "main.py") is True
        assert self.analyzer._detect_main_module(None, "", "app.py") is True
        assert self.analyzer._detect_main_module(None, "", "utils.py") is False

    def test_main_module_detection_content(self):
        code_main = """
def main(): pass

if __name__ == "__main__":
    main()
"""
        code_lib = "def foo(): pass"

        # Check analyze() integration
        res_main = self.analyzer.analyze(code_main, "script.py")
        assert res_main["is_main_module"] is True

        res_lib = self.analyzer.analyze(code_lib, "lib.py")
        assert res_lib["is_main_module"] is False

    def test_skeleton_generation(self):
        code = """
def hello():
    '''Docstring.'''
    pass

class MyClass:
    def method(self):
        pass
"""
        res = self.analyzer.analyze(code, "test.py")
        skeleton = res["skeleton"]

        # If tree-sitter is available this should be populated
        # but locally it might be missing. The class skipif handles that.
        assert "def hello" in skeleton
        assert "class MyClass" in skeleton
        assert "Docstring" in skeleton
        assert "# Line" in skeleton


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="Tree-sitter not installed")
class TestCodeAnalyzerTypeScript:
    """Test CodeAnalyzer with TypeScript code."""

    def setup_method(self):
        self.analyzer = CodeAnalyzer("typescript")

    def test_imports_extraction_es6(self):
        code = """
import React from 'react';
import { useState } from 'react';
const fs = require('fs');
import './styles.css';
"""
        result = self.analyzer.analyze(code, "App.tsx")
        imports = result["imports"]

        # Logic captures 'source' string in import_statement
        assert "react" in imports
        assert "./styles.css" in imports

        # Logic captures 'require' arguments
        assert "fs" in imports

    def test_skeleton_generation_ts(self):
        code = """
interface Props {
  name: string;
}

export const MyComp = (props: Props) => {
  return <div />;
}

function helper() {}
"""
        res = self.analyzer.analyze(code, "App.tsx")
        skeleton = res["skeleton"]

        assert "interface Props" in skeleton
        assert "function helper" in skeleton
