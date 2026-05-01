"""
Tests for code symbol-aware hybrid search.

This module verifies that the SymbolAwareBM25Scorer improves code search
by better matching exact function names, class names, variable names,
file paths, and other code-specific identifiers.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "leann-core" / "src"))

from leann.api import (
    BM25Scorer,
    CodeSymbolExtractor,
    SearchResult,
    SymbolAwareBM25Scorer,
)


class TestCodeSymbolExtractor:
    """Test the code symbol extraction utility."""

    def test_camel_case_splitting(self):
        """Test that camelCase identifiers are split correctly."""
        result = CodeSymbolExtractor._split_identifier("getUserData")
        
        assert "get" in result
        assert "user" in result
        assert "data" in result
        assert "getUserData" in result

    def test_snake_case_splitting(self):
        """Test that snake_case identifiers are split correctly."""
        result = CodeSymbolExtractor._split_identifier("get_user_data")
        
        assert "get" in result
        assert "user" in result
        assert "data" in result
        assert "get_user_data" in result

    def test_constant_case_splitting(self):
        """Test that CONSTANT_CASE identifiers are handled."""
        result = CodeSymbolExtractor._split_identifier("API_KEY")
        
        assert "api" in result or "API" in result
        assert "key" in result or "KEY" in result

    def test_extract_imports(self):
        """Test that import statements are extracted correctly."""
        code = """
from leann.api import LeannSearcher, LeannBuilder
import numpy as np
from typing import Optional, Union
"""
        result = CodeSymbolExtractor._extract_imports(code)
        
        assert "leann" in result
        assert "api" in result
        assert "leann.api" in result
        assert "LeannSearcher" in result
        assert "LeannBuilder" in result
        assert "numpy" in result
        assert "Optional" in result
        assert "Union" in result

    def test_extract_metadata_symbols(self):
        """Test that file path metadata is extracted correctly."""
        metadata = {
            "file_path": "/src/models/user.py",
            "file_name": "user.py",
            "language": "python",
        }
        result = CodeSymbolExtractor._extract_metadata_symbols(metadata)
        
        assert "src" in result
        assert "models" in result
        assert "user.py" in result
        assert "user" in result
        assert "python" in result

    def test_full_extraction(self):
        """Test full symbol extraction from code and metadata."""
        code = """
def get_user_profile(user_id: int) -> dict:
    API_KEY = os.getenv("AUTH_TOKEN")
    return {"id": user_id, "profile": {}}
"""
        metadata = {"file_path": "/src/services/auth.py"}
        
        result = CodeSymbolExtractor.extract_symbols(code, metadata)
        result_lower = [s.lower() for s in result]
        
        assert "get" in result_lower
        assert "user" in result_lower
        assert "profile" in result_lower
        assert "get_user_profile" in result_lower
        assert "api_key" in result_lower
        assert "auth_token" in result_lower
        assert "auth.py" in result_lower
        assert "auth" in result_lower


class TestSymbolAwareBM25Scorer:
    """Test the enhanced BM25 scorer with code symbol awareness."""

    @pytest.fixture
    def code_documents(self):
        """Create a set of code documents for testing."""
        return [
            {
                "id": "0",
                "text": """
def get_user_profile(user_id):
    \"\"\"Fetch user profile data.\"\"\"
    API_ENDPOINT = "https://api.example.com/users"
    response = requests.get(f"{API_ENDPOINT}/{user_id}")
    return response.json()
""",
                "metadata": {
                    "file_path": "/src/services/user_service.py",
                    "file_name": "user_service.py",
                    "language": "python",
                },
            },
            {
                "id": "1",
                "text": """
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.DB_PASSWORD = os.getenv("DB_PASS")
    
    def process_batch(self, items):
        return [self._transform(item) for item in items]
""",
                "metadata": {
                    "file_path": "/src/processing/data_pipeline.py",
                    "file_name": "data_pipeline.py",
                    "language": "python",
                },
            },
            {
                "id": "2",
                "text": """
import requests
from utils import format_response

def fetch_external_api(url, headers=None):
    \"\"\"Generic API fetcher.\"\"\"
    default_headers = {"Content-Type": "application/json"}
    actual_headers = {**default_headers, **(headers or {})}
    return requests.get(url, headers=actual_headers)
""",
                "metadata": {
                    "file_path": "/src/utils/api_client.py",
                    "file_name": "api_client.py",
                    "language": "python",
                },
            },
            {
                "id": "3",
                "text": """
The quick brown fox jumps over the lazy dog.
This is a regular text document about animals and nature.
No code here, just plain English text.
""",
                "metadata": {"file_path": "/docs/animals.txt", "file_name": "animals.txt"},
            },
        ]

    def test_basic_search_works(self, code_documents):
        """Test that basic search functionality works."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("user profile", top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score > 0 for r in results)

    def test_exact_function_name_match(self, code_documents):
        """Test that exact function name matches get higher priority."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("get_user_profile", top_k=3)
        
        assert results[0].id == "0", f"Expected id '0' for get_user_profile, got '{results[0].id}'"
        
        traditional_scorer = BM25Scorer()
        traditional_scorer.fit(code_documents)
        traditional_results = traditional_scorer.search("get_user_profile", top_k=3)
        
        traditional_first = traditional_results[0].id if traditional_results else None
        enhanced_first = results[0].id if results else None
        
        print(f"Traditional BM25 first: {traditional_first}")
        print(f"Enhanced BM25 first: {enhanced_first}")
        
        if traditional_first != "0":
            assert enhanced_first == "0", "Enhanced scorer should find exact function name match better"

    def test_class_name_match(self, code_documents):
        """Test that class name searches work better."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("DataProcessor", top_k=3)
        
        assert results[0].id == "1", f"Expected id '1' for DataProcessor, got '{results[0].id}'"
        
        assert "DataProcessor" in code_documents[1]["text"]
        assert results[0].score > 0

    def test_constant_name_match(self, code_documents):
        """Test that constant/environment variable names are found."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("DB_PASSWORD", top_k=3)
        
        assert results[0].id == "1", f"Expected id '1' for DB_PASSWORD, got '{results[0].id}'"
        
        assert "DB_PASSWORD" in code_documents[1]["text"]

    def test_file_path_match(self, code_documents):
        """Test that file path components help in search."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("user_service.py", top_k=3)
        
        assert results[0].id == "0", f"Expected id '0' for user_service.py, got '{results[0].id}'"
        
        results2 = scorer.search("data_pipeline", top_k=3)
        assert results2[0].id == "1", f"Expected id '1' for data_pipeline, got '{results2[0].id}'"

    def test_import_name_match(self, code_documents):
        """Test that import names are searchable."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("format_response", top_k=3)
        
        found_in_metadata = any("format_response" in doc["text"] for doc in code_documents)
        if found_in_metadata:
            for i, doc in enumerate(code_documents):
                if "format_response" in doc["text"]:
                    expected_id = str(i)
                    assert any(
                        r.id == expected_id for r in results
                    ), f"format_response should find document {expected_id}"

    def test_partial_identifier_match(self, code_documents):
        """Test that partial identifier matches work (e.g., 'user' for 'get_user_profile')."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("user", top_k=3)
        
        assert any(
            r.id == "0" for r in results
        ), "Searching for 'user' should find the user_service.py document"

    def test_mixed_case_query(self, code_documents):
        """Test that mixed case queries work."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results1 = scorer.search("getUserProfile", top_k=3)
        results2 = scorer.search("get_user_profile", top_k=3)
        
        assert results1[0].id == results2[0].id, "Both queries should find the same document"
        assert results1[0].id == "0", f"Expected id '0', got '{results1[0].id}'"

    def test_env_var_pattern_detection(self, code_documents):
        """Test that environment variable patterns are detected."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(code_documents)
        
        results = scorer.search("API_ENDPOINT", top_k=3)
        
        assert results[0].id == "0", f"Expected id '0' for API_ENDPOINT, got '{results[0].id}'"
        
        results2 = scorer.search("DB_PASS", top_k=3)
        assert any(
            r.id == "1" for r in results2
        ), "Searching for DB_PASS should find document with DB_PASSWORD"


class TestSymbolSearchVsTraditional:
    """Compare SymbolAwareBM25Scorer with traditional BM25Scorer."""

    @pytest.fixture
    def targeted_documents(self):
        """Documents designed to test specific symbol matching scenarios."""
        return [
            {
                "id": "target_func",
                "text": "def calculateOrderTotal(items, taxRate):\n    return sum(i.price for i in items) * (1 + taxRate)",
                "metadata": {"file_path": "/checkout/pricing.py"},
            },
            {
                "id": "similar_text",
                "text": "The order total will be calculated after adding all items and applying the tax rate.",
                "metadata": {"file_path": "/docs/order_guide.txt"},
            },
            {
                "id": "unrelated",
                "text": "This is a completely unrelated document about cooking recipes.",
                "metadata": {"file_path": "/docs/cooking.txt"},
            },
        ]

    def test_exact_function_name_precision(self, targeted_documents):
        """
        Test that SymbolAwareBM25Scorer prioritizes exact function name matches
        over semantic similarity in text.
        """
        symbol_scorer = SymbolAwareBM25Scorer()
        symbol_scorer.fit(targeted_documents)
        
        traditional_scorer = BM25Scorer()
        traditional_scorer.fit(targeted_documents)
        
        query = "calculateOrderTotal"
        
        symbol_results = symbol_scorer.search(query, top_k=3)
        traditional_results = traditional_scorer.search(query, top_k=3)
        
        print(f"\nSymbol-aware results for '{query}':")
        for r in symbol_results:
            print(f"  ID: {r.id}, Score: {r.score:.4f}")
        
        print(f"\nTraditional BM25 results for '{query}':")
        for r in traditional_results:
            print(f"  ID: {r.id}, Score: {r.score:.4f}")
        
        assert (
            symbol_results[0].id == "target_func"
        ), f"Symbol-aware should rank target_func first, got {symbol_results[0].id}"
        
        if traditional_results[0].id != "target_func":
            assert (
                symbol_results[0].id == "target_func"
            ), "Symbol-aware should outperform traditional on exact function names"

    def test_snake_case_vs_camel_case(self, targeted_documents):
        """Test that snake_case and camelCase variants both find the same document."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(targeted_documents)
        
        results_camel = scorer.search("calculateOrderTotal", top_k=2)
        results_snake = scorer.search("calculate_order_total", top_k=2)
        
        assert (
            results_camel[0].id == results_snake[0].id
        ), "Both casing variants should find the same document"
        assert results_camel[0].id == "target_func"


class TestMixedQueryTypes:
    """Test various query types that benefit from symbol awareness."""

    @pytest.fixture
    def comprehensive_docs(self):
        """Comprehensive test documents covering multiple symbol types."""
        return [
            {
                "id": "auth_service",
                "text": """
import jwt
from datetime import datetime, timedelta

SECRET_KEY = os.environ.get("JWT_SECRET", "default-secret")
ALGORITHM = "HS256"

class AuthService:
    def create_access_token(self, user_id: str) -> str:
        expire = datetime.utcnow() + timedelta(hours=24)
        payload = {"sub": user_id, "exp": expire}
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    def verify_token(self, token: str) -> dict:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
""",
                "metadata": {
                    "file_path": "/src/core/auth_service.py",
                    "file_name": "auth_service.py",
                    "language": "python",
                },
            },
            {
                "id": "payment_gateway",
                "text": """
import stripe
from typing import Optional

STRIPE_API_KEY = os.getenv("STRIPE_SECRET")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

class PaymentProcessor:
    def __init__(self):
        stripe.api_key = STRIPE_API_KEY
    
    def create_payment_intent(self, amount: int, currency: str = "usd"):
        return stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            automatic_payment_methods={"enabled": True},
        )
""",
                "metadata": {
                    "file_path": "/src/payments/stripe_gateway.py",
                    "file_name": "stripe_gateway.py",
                    "language": "python",
                },
            },
            {
                "id": "text_about_auth",
                "text": """
Authentication is an important part of any web application.
You need to create access tokens and verify them securely.
Consider using industry-standard algorithms like JWT.
""",
                "metadata": {"file_path": "/docs/security_guide.txt"},
            },
        ]

    def test_class_name_search(self, comprehensive_docs):
        """Test searching for class names."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(comprehensive_docs)
        
        results = scorer.search("AuthService", top_k=3)
        assert results[0].id == "auth_service"
        
        results2 = scorer.search("PaymentProcessor", top_k=3)
        assert results2[0].id == "payment_gateway"

    def test_method_name_search(self, comprehensive_docs):
        """Test searching for method names."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(comprehensive_docs)
        
        results = scorer.search("create_access_token", top_k=3)
        assert results[0].id == "auth_service"
        
        results2 = scorer.search("create_payment_intent", top_k=3)
        assert results2[0].id == "payment_gateway"

    def test_env_var_search(self, comprehensive_docs):
        """Test searching for environment variable names."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(comprehensive_docs)
        
        results = scorer.search("JWT_SECRET", top_k=3)
        assert results[0].id == "auth_service"
        
        results2 = scorer.search("STRIPE_API_KEY", top_k=3)
        assert results2[0].id == "payment_gateway"

    def test_file_name_search(self, comprehensive_docs):
        """Test searching by file name."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(comprehensive_docs)
        
        results = scorer.search("auth_service.py", top_k=3)
        assert results[0].id == "auth_service"
        
        results2 = scorer.search("stripe_gateway", top_k=3)
        assert results2[0].id == "payment_gateway"

    def test_import_search(self, comprehensive_docs):
        """Test searching for imported module names."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(comprehensive_docs)
        
        results = scorer.search("jwt", top_k=3)
        assert any(r.id == "auth_service" for r in results)
        
        results2 = scorer.search("stripe", top_k=3)
        assert any(r.id == "payment_gateway" for r in results2)
