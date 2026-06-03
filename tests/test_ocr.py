import inspect
import sys
from pathlib import Path

from leann.cli import LeannCLI, extract_pdf_text_with_pymupdf


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self):
        return self._text


class _FakeDoc:
    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self):
        self.closed = True


class _FakeFitz:
    def __init__(self, doc):
        self.doc = doc

    def open(self, _file_path):
        return self.doc


def test_extract_pdf_text_with_pymupdf_ocr_is_opt_in(monkeypatch):
    doc = _FakeDoc([_FakePage("embedded text")])
    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz(doc))

    text = extract_pdf_text_with_pymupdf("sample.pdf")

    assert text == "embedded text"
    assert doc.closed is True


def test_extract_pdf_text_with_pymupdf_ocrs_only_blank_pages(monkeypatch):
    from leann import cli

    doc = _FakeDoc([_FakePage("embedded "), _FakePage("   ")])
    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz(doc))

    ocr_calls = []

    def fake_ocr_page(page, page_number, file_path):
        ocr_calls.append((page, page_number, file_path))
        return "ocr text"

    monkeypatch.setattr(cli, "_extract_pdf_page_ocr_text", fake_ocr_page)

    text = extract_pdf_text_with_pymupdf("sample.pdf", use_ocr=True)

    assert text == "embedded ocr text"
    assert [call[1:] for call in ocr_calls] == [(2, "sample.pdf")]
    assert doc.closed is True


def test_extract_pdf_text_with_pymupdf_default_signature_is_backward_compatible():
    signature = inspect.signature(extract_pdf_text_with_pymupdf)

    assert list(signature.parameters) == ["file_path", "use_ocr"]
    assert signature.parameters["use_ocr"].default is False


def test_build_parser_accepts_enable_ocr_flag():
    parser = LeannCLI().create_parser()

    args = parser.parse_args(["build", "test-index", "--docs", "/tmp/docs", "--enable-ocr"])

    assert args.enable_ocr is True


def test_build_parser_enable_ocr_defaults_false():
    parser = LeannCLI().create_parser()

    args = parser.parse_args(["build", "test-index", "--docs", "/tmp/docs"])

    assert args.enable_ocr is False


def test_load_ocr_pdf_documents_preserves_metadata(tmp_path, monkeypatch):
    import apps.ocr_rag as ocr_rag

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF placeholder")

    calls = []

    def fake_extract(path, use_ocr=False):
        calls.append((Path(path).name, use_ocr))
        return "OCR text"

    monkeypatch.setattr(ocr_rag, "extract_pdf_text_with_pymupdf", fake_extract)

    documents = ocr_rag.load_ocr_pdf_documents(tmp_path)

    assert calls == [("scan.pdf", True)]
    assert len(documents) == 1
    assert documents[0].text == "OCR text"
    assert documents[0].metadata["source"] == str(pdf_path)
    assert documents[0].metadata["file_path"] == str(pdf_path)
    assert documents[0].metadata["file_name"] == "scan.pdf"
    assert documents[0].metadata["ocr_enabled"] is True
