from leann.api import Fts5BM25Index


def test_fts5_bm25_matches_chinese_substrings_after_reopen(tmp_path):
    db_path = tmp_path / "passages.sqlite"
    index = Fts5BM25Index(str(db_path))
    index.fit(
        [
            {"id": "database", "text": "数据库检索系统"},
            {"id": "image", "text": "图像分类系统"},
        ]
    )
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        assert [result.id for result in reopened.search("数据库")] == ["database"]
    finally:
        reopened.close()


def test_fts5_bm25_keeps_legacy_database_query_format(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    index = Fts5BM25Index(str(db_path), cjk_ngrams=False)
    index.fit([{"id": "database", "text": "database retrieval"}])
    index.close()

    reopened = Fts5BM25Index(str(db_path))
    try:
        assert [result.id for result in reopened.search("database")] == ["database"]
    finally:
        reopened.close()
