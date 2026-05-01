"""
Unit tests for CodeSymbolExtractor.

These tests verify the symbol extraction logic works correctly
without requiring the full LEANN dependencies.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SearchResult:
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeSymbolExtractor:
    """
    Lightweight code symbol extractor for better keyword matching in code search.
    """

    CAMEL_CASE_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
    SNAKE_CASE_PATTERN = re.compile(r"_")
    IMPORT_PATTERN = re.compile(r"(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)")
    ENV_VAR_PATTERN = re.compile(r"\b[A-Z_][A-Z0-9_]*\b")

    @classmethod
    def extract_symbols(cls, text: str, metadata: Optional[dict[str, Any]] = None) -> list[str]:
        """
        Extract code-aware symbols from text and metadata.
        """
        symbols = []
        
        words = re.findall(r"\b\w+\b", text)
        
        for word in words:
            if not word:
                continue
            
            symbols.extend(cls._split_identifier(word))
            
            if cls.ENV_VAR_PATTERN.fullmatch(word) and len(word) > 1:
                symbols.append(word.lower())
                symbols.append(word)
        
        symbols.extend(cls._extract_imports(text))
        
        if metadata:
            symbols.extend(cls._extract_metadata_symbols(metadata))
        
        return [s.lower() for s in symbols if s and len(s) > 0]

    @classmethod
    def _split_identifier(cls, identifier: str) -> list[str]:
        """
        Split a code identifier into its components.
        """
        parts = []
        
        s1 = cls.CAMEL_CASE_PATTERN.sub(r"\1 \2", identifier)
        s2 = cls.SNAKE_CASE_PATTERN.sub(r" ", s1)
        
        sub_parts = [p for p in s2.split() if p]
        parts.extend(sub_parts)
        
        if len(sub_parts) > 1:
            parts.append(identifier)
        
        if "_" in identifier:
            parts.append(identifier)
        
        return parts

    @classmethod
    def _extract_imports(cls, text: str) -> list[str]:
        """Extract module and symbol names from import statements."""
        imports = []
        
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            from_match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", line)
            if from_match:
                from_module = from_match.group(1)
                import_symbols = from_match.group(2)
                
                imports.extend(from_module.split("."))
                imports.append(from_module)
                
                for sym in import_symbols.split(","):
                    sym = sym.strip()
                    if sym:
                        if " as " in sym.lower():
                            sym = sym.split()[0]
                        imports.append(sym)
                continue
            
            import_match = re.match(r"import\s+(.+)", line)
            if import_match:
                import_part = import_match.group(1)
                for mod_part in import_part.split(","):
                    mod_part = mod_part.strip()
                    if " as " in mod_part.lower():
                        mod = mod_part.split(" as ")[0].strip()
                    else:
                        mod = mod_part
                    
                    if mod:
                        imports.extend(mod.split("."))
                        imports.append(mod)
        
        return imports

    @classmethod
    def _extract_metadata_symbols(cls, metadata: dict[str, Any]) -> list[str]:
        """Extract searchable symbols from metadata."""
        symbols = []
        
        file_path = metadata.get("file_path", "") or metadata.get("filepath", "")
        if file_path:
            path_parts = [p for p in re.split(r"[\\/]", file_path) if p]
            symbols.extend(path_parts)
            
            if path_parts:
                file_name = path_parts[-1]
                symbols.append(file_name)
                name_without_ext = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
                symbols.append(name_without_ext)
        
        file_name = metadata.get("file_name", "")
        if file_name and file_name not in symbols:
            symbols.append(file_name)
            name_without_ext = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
            symbols.append(name_without_ext)
        
        language = metadata.get("language", "")
        if language:
            symbols.append(language)
        
        return symbols


class BM25Scorer:
    """Original BM25Scorer for comparison."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = None
        self.doc_lengths = {}
        self.word_counts = {}
        self.avg_doc_length = None
        self.corpus_size = None
        self.idlist = set()

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^\w\s]", "", text).lower().split()

    def fit(self, documents: list[dict[str, Any]]):
        self.corpus_size = len(documents)
        self.doc_lengths = {}
        self.word_counts = {}
        self.idlist = set()
        doc_freqs = defaultdict(int)

        for doc_data in documents:
            doc_id = doc_data["id"]
            words = self._tokenize(doc_data["text"])
            doc_length = len(words)
            self.doc_lengths[doc_id] = doc_length

            unique_words = set(words)
            for word in unique_words:
                doc_freqs[word] += 1
            self.word_counts[doc_id] = dict(Counter(words))
            self.idlist.add(doc_id)

        self.doc_freqs = dict(doc_freqs)
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def score(self, query_words: list[str], document_id: str) -> float:
        if (
            self.doc_freqs is None
            or self.doc_lengths == {}
            or self.word_counts == {}
            or self.avg_doc_length is None
            or self.corpus_size is None
        ):
            raise ValueError("BM25 model not fitted. Call fit() before scoring.")

        passage_words = self.word_counts[document_id]
        passage_length = sum(passage_words.values())
        score = 0.0
        for word in query_words:
            if word not in self.doc_freqs:
                continue
            word_freq = passage_words[word] if word in passage_words else 0
            idf = np.log(
                (self.corpus_size - self.doc_freqs[word] + 0.5) / (self.doc_freqs[word] + 0.5) + 1
            )
            tf = (word_freq * (self.k1 + 1)) / (
                word_freq + self.k1 * (1 - self.b + self.b * (passage_length / self.avg_doc_length))
            )
            score += idf * tf
        return score

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_words = self._tokenize(query)
        scores = {doc_id: self.score(query_words, doc_id) for doc_id in self.idlist}
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(id=doc_id, score=score, text="", metadata={})
            for doc_id, score in sorted_scores[:top_k]
        ]


class SymbolAwareBM25Scorer(BM25Scorer):
    """Enhanced BM25 scorer optimized for code search."""

    EXACT_MATCH_BOOST = 3.0
    SYMBOL_MATCH_BOOST = 1.5

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        super().__init__(k1, b)
        self.doc_metadata: dict[str, dict[str, Any]] = {}
        self.raw_doc_texts: dict[str, str] = {}

    def _extract_all_terms(self, text: str, metadata: Optional[dict[str, Any]] = None) -> list[str]:
        """Extract both traditional words and code-aware symbols."""
        traditional_words = re.sub(r"[^\w\s]", "", text).lower().split()
        code_symbols = CodeSymbolExtractor.extract_symbols(text, metadata)
        
        all_terms = traditional_words + code_symbols
        
        return all_terms

    def fit(self, documents: list[dict[str, Any]]):
        self.corpus_size = len(documents)
        self.doc_lengths = {}
        self.word_counts = {}
        self.idlist = set()
        self.doc_metadata = {}
        self.raw_doc_texts = {}
        doc_freqs = defaultdict(int)

        for doc_data in documents:
            doc_id = doc_data["id"]
            text = doc_data.get("text", "")
            metadata = doc_data.get("metadata", {})
            
            self.doc_metadata[doc_id] = metadata
            self.raw_doc_texts[doc_id] = text
            
            terms = self._extract_all_terms(text, metadata)
            doc_length = len(terms)
            self.doc_lengths[doc_id] = doc_length

            unique_terms = set(terms)
            for term in unique_terms:
                doc_freqs[term] += 1
            self.word_counts[doc_id] = dict(Counter(terms))
            self.idlist.add(doc_id)

        self.doc_freqs = dict(doc_freqs)
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def _compute_exact_match_boost(self, query: str, doc_id: str) -> float:
        """Compute boost score for exact symbol matches."""
        boost = 0.0
        text = self.raw_doc_texts.get(doc_id, "")
        metadata = self.doc_metadata.get(doc_id, {})
        
        query_lower = query.lower().strip()
        query_terms = query_lower.split()
        
        for query_term in query_terms:
            if len(query_term) < 2:
                continue
            
            if re.search(rf"\b{re.escape(query_term)}\b", text, re.IGNORECASE):
                boost += 0.5
            
            exact_pattern = rf"\b{re.escape(query_term)}\b"
            if re.search(exact_pattern, text):
                boost += 1.0
            
            file_path = metadata.get("file_path", "") or metadata.get("filepath", "")
            file_name = metadata.get("file_name", "")
            
            if query_term.lower() in file_path.lower():
                boost += 1.5
            if query_term.lower() == file_name.lower() or query_term.lower() in file_name.lower():
                boost += 2.0
            
            if "_" in query_term or re.search(r"[A-Z]", query_term):
                if query_term in text:
                    boost += self.EXACT_MATCH_BOOST
        
        return boost

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search with code-aware symbol matching and exact match boosting."""
        query_terms = self._extract_all_terms(query)
        scores = {}
        
        for doc_id in self.idlist:
            base_score = self.score(query_terms, doc_id)
            exact_boost = self._compute_exact_match_boost(query, doc_id)
            scores[doc_id] = base_score + exact_boost
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(id=doc_id, score=score, text="", metadata={})
            for doc_id, score in sorted_scores[:top_k]
        ]


class TestCodeSymbolExtractor:
    """Test the code symbol extraction utility."""

    def test_camel_case_splitting(self):
        """Test that camelCase identifiers are split correctly."""
        result = CodeSymbolExtractor._split_identifier("getUserData")
        result_lower = [s.lower() for s in result]
        
        assert "get" in result_lower
        assert "user" in result_lower
        assert "data" in result_lower
        assert "getuserdata" in result_lower

    def test_snake_case_splitting(self):
        """Test that snake_case identifiers are split correctly."""
        result = CodeSymbolExtractor._split_identifier("get_user_data")
        result_lower = [s.lower() for s in result]
        
        assert "get" in result_lower
        assert "user" in result_lower
        assert "data" in result_lower
        assert "get_user_data" in result_lower

    def test_constant_case_splitting(self):
        """Test that CONSTANT_CASE identifiers are handled."""
        result = CodeSymbolExtractor._split_identifier("API_KEY")
        result_lower = [s.lower() for s in result]
        
        assert "api" in result_lower
        assert "key" in result_lower

    def test_extract_imports(self):
        """Test that import statements are extracted correctly."""
        code = """
from leann.api import LeannSearcher, LeannBuilder
import numpy as np
from typing import Optional, Union
"""
        result = CodeSymbolExtractor._extract_imports(code)
        result_lower = [s.lower() for s in result]
        
        assert "leann" in result_lower
        assert "api" in result_lower
        assert "leann.api" in result_lower
        assert "leannsearcher" in result_lower
        assert "leannbuilder" in result_lower
        assert "numpy" in result_lower
        assert "optional" in result_lower
        assert "union" in result_lower

    def test_extract_metadata_symbols(self):
        """Test that file path metadata is extracted correctly."""
        metadata = {
            "file_path": "/src/models/user.py",
            "file_name": "user.py",
            "language": "python",
        }
        result = CodeSymbolExtractor._extract_metadata_symbols(metadata)
        result_lower = [s.lower() for s in result]
        
        assert "src" in result_lower
        assert "models" in result_lower
        assert "user.py" in result_lower
        assert "user" in result_lower
        assert "python" in result_lower

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

    @property
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

    def test_basic_search_works(self):
        """Test that basic search functionality works."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(self.code_documents)
        
        results = scorer.search("user profile", top_k=3)
        
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        for i in range(1, len(results)):
            assert results[i - 1].score >= results[i].score, "Results should be sorted by score descending"

    def test_exact_function_name_match(self):
        """Test that exact function name matches get higher priority."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(self.code_documents)
        
        results = scorer.search("get_user_profile", top_k=3)
        
        assert results[0].id == "0", f"Expected id '0' for get_user_profile, got '{results[0].id}'"

    def test_class_name_match(self):
        """Test that class name searches work better."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(self.code_documents)
        
        results = scorer.search("DataProcessor", top_k=3)
        
        assert results[0].id == "1", f"Expected id '1' for DataProcessor, got '{results[0].id}'"

    def test_constant_name_match(self):
        """Test that constant/environment variable names are found."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(self.code_documents)
        
        results = scorer.search("DB_PASSWORD", top_k=3)
        
        assert results[0].id == "1", f"Expected id '1' for DB_PASSWORD, got '{results[0].id}'"

    def test_file_path_match(self):
        """Test that file path components help in search."""
        scorer = SymbolAwareBM25Scorer()
        scorer.fit(self.code_documents)
        
        results = scorer.search("user_service.py", top_k=3)
        
        assert results[0].id == "0", f"Expected id '0' for user_service.py, got '{results[0].id}'"
        
        results2 = scorer.search("data_pipeline", top_k=3)
        assert results2[0].id == "1", f"Expected id '1' for data_pipeline, got '{results2[0].id}'"


class TestSymbolSearchVsTraditional:
    """Compare SymbolAwareBM25Scorer with traditional BM25Scorer."""

    @property
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

    def test_exact_function_name_precision(self):
        """
        Test that SymbolAwareBM25Scorer prioritizes exact function name matches
        over semantic similarity in text.
        """
        symbol_scorer = SymbolAwareBM25Scorer()
        symbol_scorer.fit(self.targeted_documents)
        
        traditional_scorer = BM25Scorer()
        traditional_scorer.fit(self.targeted_documents)
        
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

    def test_symbol_aware_outperforms_traditional(self):
        """
        Verify that symbol-aware scorer gives higher scores to exact matches.
        """
        symbol_scorer = SymbolAwareBM25Scorer()
        symbol_scorer.fit(self.targeted_documents)
        
        traditional_scorer = BM25Scorer()
        traditional_scorer.fit(self.targeted_documents)
        
        query = "calculateOrderTotal"
        
        symbol_results = symbol_scorer.search(query, top_k=3)
        traditional_results = traditional_scorer.search(query, top_k=3)
        
        symbol_scores = {r.id: r.score for r in symbol_results}
        traditional_scores = {r.id: r.score for r in traditional_results}
        
        symbol_boost = symbol_scores.get("target_func", 0) - traditional_scores.get("target_func", 0)
        
        print(f"\nScore comparison for 'target_func':")
        print(f"  Traditional: {traditional_scores.get('target_func', 0):.4f}")
        print(f"  Symbol-aware: {symbol_scores.get('target_func', 0):.4f}")
        print(f"  Boost: {symbol_boost:.4f}")
        
        assert symbol_boost > 0, "Symbol-aware should provide a boost for exact function names"


if __name__ == "__main__":
    import sys
    
    tester = TestCodeSymbolExtractor()
    print("Testing CodeSymbolExtractor...")
    tester.test_camel_case_splitting()
    print("  ✓ camelCase splitting")
    tester.test_snake_case_splitting()
    print("  ✓ snake_case splitting")
    tester.test_constant_case_splitting()
    print("  ✓ CONSTANT_CASE splitting")
    tester.test_extract_imports()
    print("  ✓ import extraction")
    tester.test_extract_metadata_symbols()
    print("  ✓ metadata symbol extraction")
    tester.test_full_extraction()
    print("  ✓ full extraction")
    
    tester2 = TestSymbolAwareBM25Scorer()
    print("\nTesting SymbolAwareBM25Scorer...")
    tester2.test_basic_search_works()
    print("  ✓ basic search works")
    tester2.test_exact_function_name_match()
    print("  ✓ exact function name match")
    tester2.test_class_name_match()
    print("  ✓ class name match")
    tester2.test_constant_name_match()
    print("  ✓ constant name match")
    tester2.test_file_path_match()
    print("  ✓ file path match")
    
    tester3 = TestSymbolSearchVsTraditional()
    print("\nTesting SymbolAware vs Traditional BM25...")
    tester3.test_exact_function_name_precision()
    print("  ✓ exact function name precision")
    tester3.test_symbol_aware_outperforms_traditional()
    print("  ✓ symbol-aware outperforms traditional")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    sys.exit(0)
