# Indexed Literal and Regex Search

LEANN supports exact text search over indexed passages without shelling out to
system `grep`. Each build writes a local SQLite trigram sidecar:

```text
<index>.regex.sqlite
```

The sidecar lives next to the normal LEANN artifacts:

```text
<index>.passages.jsonl
<index>.passages.idx
<index>.bm25.sqlite
<index>.index
```

## Python API

Use `use_grep=True` for the existing case-insensitive literal search behavior:

```python
from leann.api import LeannSearcher

searcher = LeannSearcher("your_index_path")
results = searcher.search("def authenticate_user", use_grep=True, top_k=5)
```

Use `use_regex=True` for regex search:

```python
results = searcher.search(r"class .*Retriever", use_regex=True, top_k=5)
```

Regex search is case-sensitive by default. Pass `regex_case_sensitive=False` to
make verification case-insensitive:

```python
results = searcher.search(
    r"class .*retriever",
    use_regex=True,
    regex_case_sensitive=False,
)
```

## CLI

```bash
leann search my-index "def authenticate_user" --grep
leann search my-index "class .*Retriever" --regex
leann search my-index "class .*retriever" --regex --regex-ignore-case
```

`leann ask` accepts the same retrieval flags:

```bash
leann ask my-index "Where is the retriever implemented?" --regex
```

## How It Works

During `leann build`, LEANN extracts ordinary lowercase trigrams from each
passage and stores `trigram -> passage_id` postings in SQLite. At query time:

1. Literal queries extract trigrams from the literal text.
2. Regex queries extract only trigrams that are provably required by the pattern.
3. LEANN intersects postings to get candidate passage IDs.
4. LEANN loads only those passages through `PassageManager`.
5. Python `re` verifies the literal or regex match deterministically.
6. Results are ranked by match count and can still use metadata filters.

If a regex contains constructs where required trigrams are not provable, such as
alternation, groups, or character classes, LEANN falls back to scanning the live
passage store and still verifies with Python `re`. This preserves correctness:
the trigram index is only a candidate filter, never the final matcher.

## Compatibility

`use_grep=True` remains supported as a compatibility alias for literal exact
search. It no longer depends on a platform `grep` binary, so it works the same on
macOS, Linux, and Windows.

Older indexes that do not have `<index>.regex.sqlite` are upgraded lazily on the
first `use_grep=True` or `use_regex=True` search when the index directory is
writable. If the sidecar cannot be written, LEANN falls back to scanning live
passages for that query.
