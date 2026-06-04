# OCR PDF Indexing

LEANN can OCR scanned and image-heavy PDFs while building document indexes. OCR is
opt-in because it needs extra Python packages and a local Tesseract installation.

## Install

```bash
pip install "leann[ocr]"
```

From a source checkout, use:

```bash
uv sync --extra ocr
```

You also need the `tesseract` command available on `PATH`.

## CLI

Use `--enable-ocr` with PDF indexing:

```bash
leann build scanned-docs --docs ./pdfs --file-types .pdf --enable-ocr
leann build scanned-doc --docs ./scan.pdf --enable-ocr
```

LEANN first reads embedded PDF text with PyMuPDF. When `--enable-ocr` is set, only
pages with no embedded text are rendered and passed to Tesseract. Normal text PDFs
keep the same behavior as before.

## Example App

```bash
python -m apps.ocr_rag --data-dir ./pdfs --query "What is on the scanned invoice?"
```

Indexed chunks include metadata fields:

- `source`: PDF path
- `file_path`: PDF path
- `file_name`: PDF filename
- `ocr_enabled`: `true`

## Failure Behavior

If OCR is enabled without the optional Python dependencies, LEANN raises an error
explaining how to install `leann[ocr]`. If the Tesseract binary is unavailable,
the OCR call fails with the page number and PDF path so the bad input is traceable.
