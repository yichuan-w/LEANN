from unittest.mock import MagicMock, patch

from leann.web_search import WebSearcher


def test_search_uses_bounded_post_request():
    response = MagicMock()
    response.json.return_value = {
        "organic": [
            {"title": "A", "link": "https://example.com/a", "snippet": "alpha"},
            {"title": "B", "link": "https://example.com/b", "snippet": "beta"},
        ]
    }

    with patch("leann.web_search.requests.post", return_value=response) as mock_post:
        searcher = WebSearcher(api_key="key", timeout_seconds=3.5)
        results = searcher.search("leann", top_k=1)

    mock_post.assert_called_once_with(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": "key", "Content-Type": "application/json"},
        json={"q": "leann", "num": 1},
        timeout=3.5,
    )
    response.raise_for_status.assert_called_once()
    assert results == [{"title": "A", "link": "https://example.com/a", "snippet": "alpha"}]


def test_get_page_content_rejects_non_http_urls_without_request():
    searcher = WebSearcher(api_key="key")

    with patch("leann.web_search.requests.get") as mock_get:
        content = searcher.get_page_content("file:///etc/passwd")

    mock_get.assert_not_called()
    assert "only http and https URLs" in content


def test_get_page_content_uses_timeout_and_jina_headers():
    response = MagicMock()
    response.text = "# Page"

    with patch("leann.web_search.requests.get", return_value=response) as mock_get:
        searcher = WebSearcher(api_key="key", jina_api_key="jina", timeout_seconds=2)
        content = searcher.get_page_content("https://example.com/docs")

    mock_get.assert_called_once_with(
        "https://r.jina.ai/https://example.com/docs",
        headers={"X-Return-Format": "markdown", "Authorization": "Bearer jina"},
        timeout=2,
    )
    response.raise_for_status.assert_called_once()
    assert content == "# Page"
