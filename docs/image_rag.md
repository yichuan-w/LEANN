# CLIP Image RAG

`apps.image_rag` indexes local image folders with CLIP image embeddings and lets you search them
with text queries.

```bash
python -m apps.image_rag --image-dir ./screenshots --query "architecture diagram"
```

The app stores one LEANN passage per image with:

- a stable `image-<hash>` passage ID based on the image path relative to the indexed folder
- source metadata (`image_path`, `relative_path`, `image_name`, extension, byte size)
- image dimensions (`width`, `height`)
- `media_type=image` and the CLIP embedding model used for the vector

## Build and Search

```bash
# Build or rebuild an index
python -m apps.image_rag \
  --image-dir ./photos \
  --index-dir ./image_index \
  --force-rebuild

# Search the existing index
python -m apps.image_rag \
  --image-dir ./photos \
  --index-dir ./image_index \
  --query "sunset over mountains"
```

Use `--max-items` for a small first run, `--batch-size` to tune CLIP encoding batches, and
`--image-extensions` to restrict file types.

## Notes

The app uses `SentenceTransformer` with `clip-ViT-L-14` by default and builds from precomputed
image embeddings, so the LEANN index uses cosine distance with embedding recomputation disabled.
Tests mock the encoder and use generated tiny images, so CI does not download CLIP weights.
