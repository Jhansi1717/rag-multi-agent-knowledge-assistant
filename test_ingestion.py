"""test_ingestion.py

Phase 3.1 unit tests — Document model and ingestion interfaces.

Covers:
  - Document dataclass creation (valid and invalid)
  - make_document() factory
  - validate_file_path() for missing file, directory, unsupported extension,
    and all four supported extensions
  - extract_document() with a real TXT file

Run:
    python test_ingestion.py
"""

import os
import sys
import tempfile
import shutil

# Ensure the project root is on sys.path when running from any working dir.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ingestion.models import (
    Document,
    SUPPORTED_EXTENSIONS,
    make_document,
    validate_file_path,
)
from ingestion.extractor import extract_document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0
_ERRORS = []


def _run(name: str, fn):
    global _PASS, _FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        _PASS += 1
    except Exception as exc:
        print(f"  FAIL  {name}")
        print(f"        {type(exc).__name__}: {exc}")
        _FAIL += 1
        _ERRORS.append((name, exc))


def _assert(condition: bool, msg: str = "assertion failed") -> None:
    if not condition:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Test: Document model creation
# ---------------------------------------------------------------------------

def test_document_model_creation_valid():
    doc = Document(
        document_id="test-id-001",
        filename="report.txt",
        file_type=".txt",
        source="/data/report.txt",
        created_at="2026-08-29T10:00:00+00:00",
        text="Hello world.",
        metadata={"key": "value"},
    )
    _assert(doc.document_id == "test-id-001")
    _assert(doc.filename == "report.txt")
    _assert(doc.file_type == ".txt")
    _assert(doc.source == "/data/report.txt")
    _assert(doc.text == "Hello world.")
    _assert(doc.metadata == {"key": "value"})
    _assert(doc.extension == ".txt", "extension property should match file_type")
    _assert(doc.char_count == len("Hello world."), "char_count mismatch")
    _assert(doc.word_count == 2, "word_count mismatch")


def test_document_model_empty_metadata_defaults():
    """Metadata defaults to empty dict when not supplied via make_document."""
    doc = make_document(
        filename="notes.txt",
        file_type=".txt",
        source="/tmp/notes.txt",
        text="Some text.",
    )
    _assert(isinstance(doc.metadata, dict), "metadata must be a dict")
    _assert(isinstance(doc.document_id, str) and len(doc.document_id) > 0)
    _assert(doc.file_type == ".txt")


def test_document_model_file_type_lowercased():
    """make_document must lower-case the file_type."""
    doc = make_document(
        filename="report.PDF",
        file_type=".PDF",
        source="/tmp/report.PDF",
        text="content",
    )
    _assert(doc.file_type == ".pdf", f"Expected '.pdf', got {doc.file_type!r}")


def test_document_model_to_dict():
    """to_dict() must return all seven required keys."""
    doc = make_document(
        filename="api.docx",
        file_type=".docx",
        source="/docs/api.docx",
        text="API reference.",
        document_id="fixed-id-42",
        created_at="2026-01-01T00:00:00+00:00",
    )
    d = doc.to_dict()
    _assert(d["document_id"] == "fixed-id-42")
    _assert(d["filename"] == "api.docx")
    _assert(d["file_type"] == ".docx")
    _assert(d["source"] == "/docs/api.docx")
    _assert(d["created_at"] == "2026-01-01T00:00:00+00:00")
    _assert(d["text"] == "API reference.")
    _assert(isinstance(d["metadata"], dict))


def test_document_model_invalid_empty_id():
    """Document must raise ValueError when document_id is empty."""
    try:
        Document(
            document_id="",
            filename="x.txt",
            file_type=".txt",
            source="/x.txt",
            created_at="2026-01-01T00:00:00+00:00",
            text="",
        )
        raise AssertionError("Expected ValueError for empty document_id")
    except ValueError:
        pass  # expected


def test_document_model_invalid_file_type_no_dot():
    """Document must raise ValueError when file_type lacks leading dot."""
    try:
        Document(
            document_id="id-1",
            filename="x.txt",
            file_type="txt",          # missing dot
            source="/x.txt",
            created_at="2026-01-01T00:00:00+00:00",
            text="",
        )
        raise AssertionError("Expected ValueError for file_type without dot")
    except ValueError:
        pass  # expected


# ---------------------------------------------------------------------------
# Test: validate_file_path
# ---------------------------------------------------------------------------

def test_validate_missing_file():
    """Raise FileNotFoundError for a path that does not exist."""
    try:
        validate_file_path("/nonexistent/path/phantom.txt")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


def test_validate_directory_path():
    """Raise IsADirectoryError when the path is a directory."""
    tmp_dir = tempfile.mkdtemp()
    try:
        # Give it a .txt extension in its name so extension check doesn't fire first
        named_dir = os.path.join(tempfile.gettempdir(), "fakefile.txt_dir")
        os.makedirs(named_dir, exist_ok=True)
        try:
            validate_file_path(named_dir)
            raise AssertionError("Expected IsADirectoryError")
        except IsADirectoryError:
            pass  # expected
        finally:
            shutil.rmtree(named_dir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_validate_unsupported_extension():
    """Raise ValueError for a real file with an unsupported extension."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"# Hello")
        tmp_path = f.name
    try:
        validate_file_path(tmp_path)
        raise AssertionError("Expected ValueError for .md extension")
    except ValueError as exc:
        _assert(".md" in str(exc), "Error message should mention the bad extension")
    finally:
        os.unlink(tmp_path)


def test_validate_supported_extensions_all_pass():
    """validate_file_path must not raise for all four supported extensions."""
    for ext in (".txt", ".pdf", ".docx", ".csv"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(b"placeholder")
            tmp_path = f.name
        try:
            # Should complete without raising.
            validate_file_path(tmp_path)
        finally:
            os.unlink(tmp_path)


def test_supported_extensions_constant():
    """SUPPORTED_EXTENSIONS must contain exactly the four agreed extensions."""
    _assert(".pdf" in SUPPORTED_EXTENSIONS)
    _assert(".docx" in SUPPORTED_EXTENSIONS)
    _assert(".txt" in SUPPORTED_EXTENSIONS)
    _assert(".csv" in SUPPORTED_EXTENSIONS)
    _assert(len(SUPPORTED_EXTENSIONS) == 4, "Unexpected extra entries")


# ---------------------------------------------------------------------------
# Test: extract_document — unified entry point
# ---------------------------------------------------------------------------

def test_extract_document_returns_document_instance():
    """extract_document() must return a Document for a valid TXT file."""
    content = (
        "This is a Phase 3 integration test document.\n"
        "It contains multiple sentences to validate extraction.\n"
        "The Document model should capture the full text."
    )
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
        tmp_path = f.name
    try:
        doc = extract_document(tmp_path)
        _assert(isinstance(doc, Document), "Return value must be a Document")
        _assert(doc.filename == os.path.basename(tmp_path))
        _assert(doc.file_type == ".txt")
        _assert(doc.source == tmp_path)
        _assert(content in doc.text, "Full text must be preserved")
        _assert(isinstance(doc.document_id, str) and len(doc.document_id) > 0)
        _assert("created_at" in doc.to_dict())
        _assert("file_size_bytes" in doc.metadata)
        _assert("format" in doc.metadata)
        _assert(doc.metadata["format"] == "txt")
    finally:
        os.unlink(tmp_path)


def test_extract_document_missing_file_raises():
    """extract_document() must raise FileNotFoundError for missing path."""
    try:
        extract_document("/no/such/file/document.txt")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


def test_extract_document_unsupported_ext_raises():
    """extract_document() must raise ValueError for unsupported extension."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(b"{}")
        tmp_path = f.name
    try:
        extract_document(tmp_path)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass  # expected
    finally:
        os.unlink(tmp_path)


def test_extract_document_unique_ids():
    """Two calls on the same file must produce different document_ids."""
    with tempfile.NamedTemporaryFile(
        suffix=".txt", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write("Unique ID test content.")
        tmp_path = f.name
    try:
        doc_a = extract_document(tmp_path)
        doc_b = extract_document(tmp_path)
        _assert(
            doc_a.document_id != doc_b.document_id,
            "Each extraction call must yield a distinct document_id (UUID4).",
        )
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Phase 3.2 — PDF extraction tests
# ---------------------------------------------------------------------------

# Path to the sample PDF generated by generate_samples.py.
# The file is committed to the repository under data/hr/.
_SAMPLE_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "hr", "leave_policy.pdf",
)


def _require_sample_pdf():
    """Skip-like guard: raise RuntimeError if the sample PDF is absent."""
    if not os.path.isfile(_SAMPLE_PDF):
        raise RuntimeError(
            f"Sample PDF not found: {_SAMPLE_PDF}. "
            "Run `python generate_samples.py` first."
        )


# --- PdfExtractor unit tests (direct class usage) ---------------------------

def test_pdf_extractor_returns_dict_keys():
    """PdfExtractor.extract() must return a dict with 'text' and 'metadata'."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    _assert(isinstance(result, dict), "result must be a dict")
    _assert("text" in result, "result must contain 'text'")
    _assert("metadata" in result, "result must contain 'metadata'")


def test_pdf_extractor_text_non_empty():
    """Extracted text from the sample PDF must be non-empty."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    _assert(
        len(result["text"].strip()) > 0,
        "Sample PDF must yield non-empty text.",
    )


def test_pdf_extractor_page_count_captured():
    """page_count must be present in metadata and be a positive integer."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    meta = result["metadata"]
    _assert("page_count" in meta, "metadata must contain 'page_count'")
    _assert(
        isinstance(meta["page_count"], int) and meta["page_count"] >= 1,
        f"page_count must be a positive int, got {meta.get('page_count')!r}",
    )


def test_pdf_extractor_metadata_required_keys():
    """All required metadata keys must be present for a PDF extraction."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    meta = result["metadata"]
    required_keys = [
        "source_location",
        "document_name",
        "file_size_bytes",
        "format",
        "page_count",
        "empty_page_count",
        "empty_page_numbers",
        "pdf_producer",
        "pdf_creator",
        "pdf_creation_date",
        "has_text",
    ]
    for key in required_keys:
        _assert(key in meta, f"metadata missing required key: {key!r}")


def test_pdf_extractor_format_is_pdf():
    """metadata['format'] must equal 'pdf' (lower-case, no dot)."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    _assert(
        result["metadata"]["format"] == "pdf",
        f"Expected 'pdf', got {result['metadata']['format']!r}",
    )


def test_pdf_extractor_has_text_true():
    """has_text must be True for the sample text-based PDF."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    _assert(
        result["metadata"]["has_text"] is True,
        "Sample PDF should have has_text=True.",
    )


def test_pdf_extractor_known_content():
    """Extracted text must contain phrases from the generated leave policy."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    text = result["text"]
    _assert("Annual Leave" in text, "Text must contain 'Annual Leave'")
    _assert("20 days" in text or "20" in text, "Text must reference leave quantity")


def test_pdf_extractor_empty_page_list_type():
    """empty_page_numbers must be a list (possibly empty for a normal PDF)."""
    from ingestion.pdf_extractor import PdfExtractor
    _require_sample_pdf()
    result = PdfExtractor().extract(_SAMPLE_PDF)
    _assert(
        isinstance(result["metadata"]["empty_page_numbers"], list),
        "empty_page_numbers must be a list",
    )


def test_pdf_extractor_missing_file_raises():
    """PdfExtractor must raise FileNotFoundError for a non-existent path."""
    from ingestion.pdf_extractor import PdfExtractor
    try:
        PdfExtractor().extract("/no/such/file/phantom.pdf")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


# --- extract_document() PDF integration tests --------------------------------

def test_extract_document_pdf_returns_document():
    """extract_document() on the sample PDF must return a Document."""
    from ingestion.models import Document
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(isinstance(doc, Document), "Must return a Document instance")


def test_extract_document_pdf_file_type():
    """Document.file_type must be '.pdf'."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(doc.file_type == ".pdf", f"Expected '.pdf', got {doc.file_type!r}")


def test_extract_document_pdf_filename_preserved():
    """Document.filename must match the basename of the input path."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(
        doc.filename == os.path.basename(_SAMPLE_PDF),
        f"filename mismatch: {doc.filename!r}",
    )


def test_extract_document_pdf_source_preserved():
    """Document.source must equal the full path passed to extract_document."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(doc.source == _SAMPLE_PDF, f"source mismatch: {doc.source!r}")


def test_extract_document_pdf_text_non_empty():
    """Document.text must not be empty for the text-based sample PDF."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(len(doc.text.strip()) > 0, "Document.text must not be empty")


def test_extract_document_pdf_page_count_in_metadata():
    """Document.metadata must carry page_count from PdfExtractor."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(
        "page_count" in doc.metadata,
        "page_count must be in Document.metadata for PDF",
    )
    _assert(
        doc.metadata["page_count"] >= 1,
        "page_count must be at least 1",
    )


def test_extract_document_pdf_document_id_present():
    """Document.document_id must be a non-empty string."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(
        isinstance(doc.document_id, str) and len(doc.document_id) > 0,
        "document_id must be a non-empty string",
    )


def test_extract_document_pdf_created_at_present():
    """Document.created_at must be a non-empty ISO 8601 timestamp string."""
    _require_sample_pdf()
    doc = extract_document(_SAMPLE_PDF)
    _assert(
        isinstance(doc.created_at, str) and len(doc.created_at) > 0,
        "created_at must be a non-empty string",
    )
    # Quick ISO 8601 sanity: should contain 'T'
    _assert("T" in doc.created_at, f"created_at does not look like ISO 8601: {doc.created_at!r}")


# ---------------------------------------------------------------------------
# Phase 3.3 — DOCX extraction tests
# ---------------------------------------------------------------------------

_SAMPLE_DOCX = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "technical", "api_documentation.docx",
)


def _require_sample_docx():
    """Skip-like guard: raise RuntimeError if the sample DOCX is absent."""
    if not os.path.isfile(_SAMPLE_DOCX):
        raise RuntimeError(
            f"Sample DOCX not found: {_SAMPLE_DOCX}. "
            "Run `python generate_samples.py` first."
        )


# --- DocxExtractor unit tests -----------------------------------------------

def test_docx_extractor_returns_dict_keys():
    """DocxExtractor.extract() must return a dict with 'text' and 'metadata'."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    _assert(isinstance(result, dict), "result must be a dict")
    _assert("text" in result, "result must contain 'text'")
    _assert("metadata" in result, "result must contain 'metadata'")


def test_docx_extractor_text_non_empty():
    """Extracted text from the sample DOCX must be non-empty."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    _assert(
        len(result["text"].strip()) > 0,
        "Sample DOCX must yield non-empty text.",
    )


def test_docx_extractor_metadata_required_keys():
    """All required metadata keys must be present for a DOCX extraction."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    meta = result["metadata"]
    required_keys = [
        "source_location",
        "document_name",
        "file_size_bytes",
        "format",
        "paragraph_count",
        "table_count",
        "has_text",
    ]
    for key in required_keys:
        _assert(key in meta, f"metadata missing required key: {key!r}")


def test_docx_extractor_paragraph_count_captured():
    """paragraph_count must be present in metadata and be >= 1."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    meta = result["metadata"]
    _assert("paragraph_count" in meta, "metadata must contain 'paragraph_count'")
    _assert(
        isinstance(meta["paragraph_count"], int) and meta["paragraph_count"] >= 1,
        f"paragraph_count must be a positive int, got {meta.get('paragraph_count')!r}",
    )


def test_docx_extractor_table_count_captured():
    """table_count must be present in metadata as an integer."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    meta = result["metadata"]
    _assert("table_count" in meta, "metadata must contain 'table_count'")
    _assert(
        isinstance(meta["table_count"], int) and meta["table_count"] >= 0,
        f"table_count must be a non-negative int, got {meta.get('table_count')!r}",
    )


def test_docx_extractor_format_is_docx():
    """metadata['format'] must equal 'docx' (lower-case, no dot)."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    _assert(
        result["metadata"]["format"] == "docx",
        f"Expected 'docx', got {result['metadata']['format']!r}",
    )


def test_docx_extractor_has_text_true():
    """has_text must be True for the sample DOCX."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    _assert(
        result["metadata"]["has_text"] is True,
        "Sample DOCX should have has_text=True.",
    )


def test_docx_extractor_known_content():
    """Extracted text must contain expected phrases from api_documentation.docx."""
    from ingestion.docx_extractor import DocxExtractor
    _require_sample_docx()
    result = DocxExtractor().extract(_SAMPLE_DOCX)
    text = result["text"]
    _assert("API Integration Guidelines" in text, "Text must contain heading")
    _assert("REST" in text, "Text must mention REST")
    _assert("OAuth 2.0" in text, "Text must mention OAuth 2.0")


def test_docx_extractor_table_extraction():
    """DocxExtractor must extract table content formatted as rows."""
    import docx
    from ingestion.docx_extractor import DocxExtractor
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        tmp_path = f.name

    try:
        doc = docx.Document()
        doc.add_paragraph("Pre-table introductory paragraph.")
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Header 1"
        table.cell(0, 1).text = "Header 2"
        table.cell(1, 0).text = "Value A"
        table.cell(1, 1).text = "Value B"
        doc.add_paragraph("Post-table concluding paragraph.")
        doc.save(tmp_path)

        result = DocxExtractor().extract(tmp_path)
        meta = result["metadata"]
        text = result["text"]

        _assert(meta["table_count"] == 1, f"Expected table_count=1, got {meta['table_count']}")
        _assert(meta["paragraph_count"] == 2, f"Expected paragraph_count=2, got {meta['paragraph_count']}")
        _assert("Header 1 | Header 2" in text, "Table row 1 text mismatch")
        _assert("Value A | Value B" in text, "Table row 2 text mismatch")
        # Verify body order: pre-table -> table -> post-table
        pos_pre = text.find("Pre-table")
        pos_table = text.find("Header 1")
        pos_post = text.find("Post-table")
        _assert(pos_pre < pos_table < pos_post, "Document body ordering should be preserved")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_docx_extractor_missing_file_raises():
    """DocxExtractor must raise FileNotFoundError for a non-existent path."""
    from ingestion.docx_extractor import DocxExtractor
    try:
        DocxExtractor().extract("/no/such/file/phantom.docx")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


# --- extract_document() DOCX integration tests ------------------------------

def test_extract_document_docx_returns_document():
    """extract_document() on the sample DOCX must return a Document."""
    from ingestion.models import Document
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(isinstance(doc, Document), "Must return a Document instance")


def test_extract_document_docx_file_type():
    """Document.file_type must be '.docx'."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(doc.file_type == ".docx", f"Expected '.docx', got {doc.file_type!r}")


def test_extract_document_docx_filename_preserved():
    """Document.filename must match the basename of the input path."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(
        doc.filename == os.path.basename(_SAMPLE_DOCX),
        f"filename mismatch: {doc.filename!r}",
    )


def test_extract_document_docx_source_preserved():
    """Document.source must equal the full path passed to extract_document."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(doc.source == _SAMPLE_DOCX, f"source mismatch: {doc.source!r}")


def test_extract_document_docx_text_non_empty():
    """Document.text must not be empty for the sample DOCX."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(len(doc.text.strip()) > 0, "Document.text must not be empty")


def test_extract_document_docx_paragraph_count_in_metadata():
    """Document.metadata must carry paragraph_count from DocxExtractor."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(
        "paragraph_count" in doc.metadata,
        "paragraph_count must be in Document.metadata for DOCX",
    )
    _assert(
        doc.metadata["paragraph_count"] >= 1,
        "paragraph_count must be at least 1",
    )


def test_extract_document_docx_table_count_in_metadata():
    """Document.metadata must carry table_count from DocxExtractor."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(
        "table_count" in doc.metadata,
        "table_count must be in Document.metadata for DOCX",
    )


def test_extract_document_docx_document_id_present():
    """Document.document_id must be a non-empty string."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(
        isinstance(doc.document_id, str) and len(doc.document_id) > 0,
        "document_id must be a non-empty string",
    )


def test_extract_document_docx_created_at_present():
    """Document.created_at must be a non-empty ISO 8601 timestamp string."""
    _require_sample_docx()
    doc = extract_document(_SAMPLE_DOCX)
    _assert(
        isinstance(doc.created_at, str) and len(doc.created_at) > 0,
        "created_at must be a non-empty string",
    )
    _assert("T" in doc.created_at, f"created_at does not look like ISO 8601: {doc.created_at!r}")


# ---------------------------------------------------------------------------
# Phase 3.4 — TXT and CSV extraction tests
# ---------------------------------------------------------------------------

_SAMPLE_TXT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "technical", "deployment_guide.txt",
)

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "hr", "leave_allowance.csv",
)


def _require_sample_txt():
    """Skip-like guard: raise RuntimeError if the sample TXT is absent."""
    if not os.path.isfile(_SAMPLE_TXT):
        raise RuntimeError(
            f"Sample TXT not found: {_SAMPLE_TXT}. "
            "Run `python generate_samples.py` first."
        )


def _require_sample_csv():
    """Skip-like guard: raise RuntimeError if the sample CSV is absent."""
    if not os.path.isfile(_SAMPLE_CSV):
        raise RuntimeError(
            f"Sample CSV not found: {_SAMPLE_CSV}. "
            "Run `python generate_samples.py` first."
        )


# --- TxtExtractor unit tests ------------------------------------------------

def test_txt_extractor_returns_dict_keys():
    """TxtExtractor.extract() must return a dict with 'text' and 'metadata'."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    _assert(isinstance(result, dict), "result must be a dict")
    _assert("text" in result, "result must contain 'text'")
    _assert("metadata" in result, "result must contain 'metadata'")


def test_txt_extractor_text_non_empty():
    """Extracted text from the sample TXT must be non-empty."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    _assert(len(result["text"].strip()) > 0, "Sample TXT must yield non-empty text.")


def test_txt_extractor_metadata_required_keys():
    """All required metadata keys must be present for a TXT extraction."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    meta = result["metadata"]
    required_keys = [
        "source_location",
        "document_name",
        "file_size_bytes",
        "format",
        "line_count",
        "has_text",
    ]
    for key in required_keys:
        _assert(key in meta, f"metadata missing required key: {key!r}")


def test_txt_extractor_line_count_captured():
    """line_count must be present in metadata and be >= 1."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    meta = result["metadata"]
    _assert("line_count" in meta, "metadata must contain 'line_count'")
    _assert(
        isinstance(meta["line_count"], int) and meta["line_count"] >= 1,
        f"line_count must be a positive int, got {meta.get('line_count')!r}",
    )


def test_txt_extractor_format_is_txt():
    """metadata['format'] must equal 'txt' (lower-case, no dot)."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    _assert(result["metadata"]["format"] == "txt", f"Expected 'txt', got {result['metadata']['format']!r}")


def test_txt_extractor_has_text_true():
    """has_text must be True for the sample TXT."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    _assert(result["metadata"]["has_text"] is True, "Sample TXT should have has_text=True.")


def test_txt_extractor_known_content():
    """Extracted text must contain expected phrases from deployment_guide.txt."""
    from ingestion.txt_extractor import TxtExtractor
    _require_sample_txt()
    result = TxtExtractor().extract(_SAMPLE_TXT)
    text = result["text"]
    _assert("Deployment Procedure" in text, "Text must contain heading")
    _assert("docker pull" in text, "Text must mention docker pull")


def test_txt_extractor_decoding_safe():
    """TxtExtractor must safely handle invalid UTF-8 bytes without crashing."""
    from ingestion.txt_extractor import TxtExtractor
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        # Write bytes that are invalid in UTF-8
        f.write(b"Valid prefix. \xff\xfe Invalid bytes. Valid suffix.\nSecond line.")
        tmp_path = f.name
    try:
        result = TxtExtractor().extract(tmp_path)
        _assert("Valid prefix." in result["text"], "Valid text portion should be preserved")
        _assert(result["metadata"]["line_count"] == 2, "Line count should be 2")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_txt_extractor_missing_file_raises():
    """TxtExtractor must raise FileNotFoundError for a non-existent path."""
    from ingestion.txt_extractor import TxtExtractor
    try:
        TxtExtractor().extract("/no/such/file/phantom.txt")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


# --- extract_document() TXT integration tests -------------------------------

def test_extract_document_txt_returns_document():
    """extract_document() on the sample TXT must return a Document."""
    from ingestion.models import Document
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(isinstance(doc, Document), "Must return a Document instance")


def test_extract_document_txt_file_type():
    """Document.file_type must be '.txt'."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(doc.file_type == ".txt", f"Expected '.txt', got {doc.file_type!r}")


def test_extract_document_txt_filename_preserved():
    """Document.filename must match the basename of the input path."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(doc.filename == os.path.basename(_SAMPLE_TXT), f"filename mismatch: {doc.filename!r}")


def test_extract_document_txt_source_preserved():
    """Document.source must equal the full path passed to extract_document."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(doc.source == _SAMPLE_TXT, f"source mismatch: {doc.source!r}")


def test_extract_document_txt_text_non_empty():
    """Document.text must not be empty for the sample TXT."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(len(doc.text.strip()) > 0, "Document.text must not be empty")


def test_extract_document_txt_line_count_in_metadata():
    """Document.metadata must carry line_count from TxtExtractor."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert("line_count" in doc.metadata, "line_count must be in Document.metadata for TXT")
    _assert(doc.metadata["line_count"] >= 1, "line_count must be at least 1")


def test_extract_document_txt_document_id_present():
    """Document.document_id must be a non-empty string."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(isinstance(doc.document_id, str) and len(doc.document_id) > 0, "document_id must be a non-empty string")


def test_extract_document_txt_created_at_present():
    """Document.created_at must be a non-empty ISO 8601 timestamp string."""
    _require_sample_txt()
    doc = extract_document(_SAMPLE_TXT)
    _assert(isinstance(doc.created_at, str) and len(doc.created_at) > 0, "created_at must be a non-empty string")
    _assert("T" in doc.created_at, f"created_at does not look like ISO 8601: {doc.created_at!r}")


# --- CsvExtractor unit tests ------------------------------------------------

def test_csv_extractor_returns_dict_keys():
    """CsvExtractor.extract() must return a dict with 'text' and 'metadata'."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    _assert(isinstance(result, dict), "result must be a dict")
    _assert("text" in result, "result must contain 'text'")
    _assert("metadata" in result, "result must contain 'metadata'")


def test_csv_extractor_text_non_empty():
    """Extracted text from the sample CSV must be non-empty."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    _assert(len(result["text"].strip()) > 0, "Sample CSV must yield non-empty text.")


def test_csv_extractor_metadata_required_keys():
    """All required metadata keys must be present for a CSV extraction."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    meta = result["metadata"]
    required_keys = [
        "source_location",
        "document_name",
        "file_size_bytes",
        "format",
        "row_count",
        "column_count",
        "column_names",
        "has_text",
    ]
    for key in required_keys:
        _assert(key in meta, f"metadata missing required key: {key!r}")


def test_csv_extractor_row_count_captured():
    """row_count must match the number of data rows in leave_allowance.csv (5)."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    meta = result["metadata"]
    _assert(meta["row_count"] == 5, f"Expected row_count=5, got {meta['row_count']}")


def test_csv_extractor_column_count_captured():
    """column_count must match the number of columns in leave_allowance.csv (4)."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    meta = result["metadata"]
    _assert(meta["column_count"] == 4, f"Expected column_count=4, got {meta['column_count']}")


def test_csv_extractor_column_names_captured():
    """column_names must be a list containing all original header names."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    cols = result["metadata"]["column_names"]
    _assert("Leave_Type" in cols, "Leave_Type missing from column_names")
    _assert("Days_Allowed" in cols, "Days_Allowed missing from column_names")
    _assert("Carry_Forward_Limit" in cols, "Carry_Forward_Limit missing from column_names")
    _assert("Approval_Required" in cols, "Approval_Required missing from column_names")


def test_csv_extractor_format_is_csv():
    """metadata['format'] must equal 'csv' (lower-case, no dot)."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    _assert(result["metadata"]["format"] == "csv", f"Expected 'csv', got {result['metadata']['format']!r}")


def test_csv_extractor_has_text_true():
    """has_text must be True for the sample CSV."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    _assert(result["metadata"]["has_text"] is True, "Sample CSV should have has_text=True.")


def test_csv_extractor_semantic_row_formatting():
    """Extracted text must format each row with column labels and row indices."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    text = result["text"]
    # Check row-level formatting
    _assert("Row 1:" in text, "Text must contain 'Row 1:' prefix")
    _assert("Leave_Type: Annual" in text, "Text must contain 'Leave_Type: Annual'")
    _assert("Days_Allowed: 20" in text, "Text must contain 'Days_Allowed: 20'")
    _assert("Row 2:" in text, "Text must contain 'Row 2:' prefix")
    _assert("Leave_Type: Sick" in text, "Text must contain 'Leave_Type: Sick'")


def test_csv_extractor_known_content():
    """Extracted text must contain all expected leave categories."""
    from ingestion.csv_extractor import CsvExtractor
    _require_sample_csv()
    result = CsvExtractor().extract(_SAMPLE_CSV)
    text = result["text"]
    for category in ["Annual", "Sick", "Maternity", "Paternity", "Bereavement"]:
        _assert(category in text, f"Text must mention leave category: {category}")


def test_csv_extractor_missing_file_raises():
    """CsvExtractor must raise FileNotFoundError for a non-existent path."""
    from ingestion.csv_extractor import CsvExtractor
    try:
        CsvExtractor().extract("/no/such/file/phantom.csv")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass  # expected


# --- extract_document() CSV integration tests -------------------------------

def test_extract_document_csv_returns_document():
    """extract_document() on the sample CSV must return a Document."""
    from ingestion.models import Document
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(isinstance(doc, Document), "Must return a Document instance")


def test_extract_document_csv_file_type():
    """Document.file_type must be '.csv'."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(doc.file_type == ".csv", f"Expected '.csv', got {doc.file_type!r}")


def test_extract_document_csv_filename_preserved():
    """Document.filename must match the basename of the input path."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(doc.filename == os.path.basename(_SAMPLE_CSV), f"filename mismatch: {doc.filename!r}")


def test_extract_document_csv_source_preserved():
    """Document.source must equal the full path passed to extract_document."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(doc.source == _SAMPLE_CSV, f"source mismatch: {doc.source!r}")


def test_extract_document_csv_text_non_empty():
    """Document.text must not be empty for the sample CSV."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(len(doc.text.strip()) > 0, "Document.text must not be empty")


def test_extract_document_csv_row_count_in_metadata():
    """Document.metadata must carry row_count from CsvExtractor."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert("row_count" in doc.metadata, "row_count must be in Document.metadata for CSV")
    _assert(doc.metadata["row_count"] == 5, f"Expected row_count=5, got {doc.metadata['row_count']}")


def test_extract_document_csv_column_count_in_metadata():
    """Document.metadata must carry column_count from CsvExtractor."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert("column_count" in doc.metadata, "column_count must be in Document.metadata for CSV")
    _assert(doc.metadata["column_count"] == 4, f"Expected column_count=4, got {doc.metadata['column_count']}")


def test_extract_document_csv_column_names_in_metadata():
    """Document.metadata must carry column_names list from CsvExtractor."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert("column_names" in doc.metadata, "column_names must be in Document.metadata for CSV")
    _assert(len(doc.metadata["column_names"]) == 4, "column_names must have 4 entries")


def test_extract_document_csv_document_id_present():
    """Document.document_id must be a non-empty string."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(isinstance(doc.document_id, str) and len(doc.document_id) > 0, "document_id must be a non-empty string")


def test_extract_document_csv_created_at_present():
    """Document.created_at must be a non-empty ISO 8601 timestamp string."""
    _require_sample_csv()
    doc = extract_document(_SAMPLE_CSV)
    _assert(isinstance(doc.created_at, str) and len(doc.created_at) > 0, "created_at must be a non-empty string")
    _assert("T" in doc.created_at, f"created_at does not look like ISO 8601: {doc.created_at!r}")


# ---------------------------------------------------------------------------
# Phase 3.5 — Text cleaning and normalization tests
# ---------------------------------------------------------------------------

def test_clean_text_repeated_whitespace():
    """clean_text must collapse multiple horizontal spaces/tabs to a single space."""
    from ingestion.cleaner import clean_text
    raw = "This    is   a \t  sentence    with    erratic   spacing."
    expected = "This is a sentence with erratic spacing."
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_excessive_blank_lines():
    """clean_text must collapse 3 or more newlines to a standard double newline."""
    from ingestion.cleaner import clean_text
    raw = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
    expected = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_mixed_line_endings():
    """clean_text must normalize CRLF and CR line endings to LF."""
    from ingestion.cleaner import clean_text
    raw = "Line 1\r\nLine 2\rLine 3\nLine 4"
    expected = "Line 1\nLine 2\nLine 3\nLine 4"
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_preserves_headings_and_numbering():
    """clean_text must preserve heading hierarchy and numbered lists."""
    from ingestion.cleaner import clean_text
    raw = "  # 1. Main Policy Heading \n\n  1.1 Sub-item: 20 days entitlement. \n  1.2 Sub-item: 10 days sick leave.  "
    expected = "# 1. Main Policy Heading\n\n1.1 Sub-item: 20 days entitlement.\n1.2 Sub-item: 10 days sick leave."
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_preserves_punctuation_and_symbols():
    """clean_text must not strip punctuation, quotes, hyphens, or brackets."""
    from ingestion.cleaner import clean_text
    raw = 'Standard HTTP verbs: ["GET", "POST", "PUT", "DELETE"] & OAuth 2.0 (Bearer token).'
    expected = 'Standard HTTP verbs: ["GET", "POST", "PUT", "DELETE"] & OAuth 2.0 (Bearer token).'
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_preserves_table_rows():
    """clean_text must preserve table row pipes and field representations."""
    from ingestion.cleaner import clean_text
    raw = "Row 1: Leave_Type: Annual   |   Days_Allowed: 20   |   Carry_Forward_Limit: 5"
    expected = "Row 1: Leave_Type: Annual | Days_Allowed: 20 | Carry_Forward_Limit: 5"
    _assert(clean_text(raw) == expected, f"Expected {expected!r}, got {clean_text(raw)!r}")


def test_clean_text_idempotence():
    """Running clean_text twice must produce identical results."""
    from ingestion.cleaner import clean_text
    raw = "  Heading 1 \r\n\r\n\r\n  Body text   with   multiple    spaces.  \n"
    first_pass = clean_text(raw)
    second_pass = clean_text(first_pass)
    _assert(first_pass == second_pass, "clean_text must be idempotent")


def test_clean_text_empty_input():
    """clean_text on empty/whitespace-only input returns empty string."""
    from ingestion.cleaner import clean_text
    _assert(clean_text("") == "")
    _assert(clean_text("   \n\n\t\r\n  ") == "")


def test_clean_document_updates_metadata_and_text():
    """clean_document must normalize document text and add cleaning metadata."""
    from ingestion.models import make_document
    from ingestion.cleaner import clean_document

    doc = make_document(
        filename="test.txt",
        file_type=".txt",
        source="/tmp/test.txt",
        text="Line 1   \r\n\r\n\r\n\r\nLine 2   with   spaces.",
        metadata={"author": "Test Author"},
    )
    cleaned_doc = clean_document(doc)

    _assert(cleaned_doc.text == "Line 1\n\nLine 2 with spaces.")
    _assert(cleaned_doc.filename == "test.txt")
    _assert(cleaned_doc.file_type == ".txt")
    _assert(cleaned_doc.document_id == doc.document_id)
    _assert(cleaned_doc.metadata.get("author") == "Test Author")
    _assert(cleaned_doc.metadata.get("cleaned") is True)
    _assert(cleaned_doc.metadata.get("cleaned_char_count") == len("Line 1\n\nLine 2 with spaces."))


def test_clean_document_on_all_real_samples():
    """clean_document must succeed cleanly on real PDF, DOCX, TXT, and CSV samples."""
    from ingestion.cleaner import clean_document

    sample_files = [_SAMPLE_PDF, _SAMPLE_DOCX, _SAMPLE_TXT, _SAMPLE_CSV]
    for sample in sample_files:
        doc = extract_document(sample)
        cleaned_doc = clean_document(doc)

        _assert(len(cleaned_doc.text.strip()) > 0, f"Cleaned text empty for {sample}")
        _assert(cleaned_doc.metadata.get("cleaned") is True)
        _assert(cleaned_doc.filename == doc.filename)
        _assert(cleaned_doc.file_type == doc.file_type)


# ---------------------------------------------------------------------------
# Phase 3.6 — Token-aware chunking tests
# ---------------------------------------------------------------------------

def test_chunk_model_valid_construction():
    """Chunk model instantiates and exposes properties correctly."""
    from ingestion.models import Chunk, make_chunk

    chunk = make_chunk(
        document_id="doc-123",
        document_name="policy.pdf",
        chunk_index=0,
        text="Sample chunk content text.",
        source_location="/path/to/policy.pdf",
        metadata={"token_count": 5, "token_start": 0, "token_end": 5},
    )
    _assert(chunk.chunk_id == "doc-123_0")
    _assert(chunk.document_id == "doc-123")
    _assert(chunk.chunk_index == 0)
    _assert(chunk.token_count == 5)
    _assert(chunk.char_count == len("Sample chunk content text."))


def test_chunk_model_invalid_construction():
    """Chunk model raises ValueError for empty IDs or negative indices."""
    from ingestion.models import Chunk

    try:
        Chunk(
            chunk_id="",
            document_id="doc-1",
            document_name="test.txt",
            chunk_index=0,
            text="text",
            source_location="src",
        )
        raise AssertionError("Expected ValueError for empty chunk_id")
    except ValueError:
        pass

    try:
        Chunk(
            chunk_id="doc-1_0",
            document_id="",
            document_name="test.txt",
            chunk_index=0,
            text="text",
            source_location="src",
        )
        raise AssertionError("Expected ValueError for empty document_id")
    except ValueError:
        pass

    try:
        Chunk(
            chunk_id="doc-1_-1",
            document_id="doc-1",
            document_name="test.txt",
            chunk_index=-1,
            text="text",
            source_location="src",
        )
        raise AssertionError("Expected ValueError for negative chunk_index")
    except ValueError:
        pass


def test_chunk_model_to_dict():
    """Chunk.to_dict() contains all required vector store fields."""
    from ingestion.models import make_chunk

    chunk = make_chunk(
        document_id="doc-123",
        document_name="guide.txt",
        chunk_index=2,
        text="Step 3: Deploy.",
        source_location="guide.txt",
        metadata={"token_start": 100},
    )
    d = chunk.to_dict()
    required = ["chunk_id", "document_id", "document_name", "chunk_index", "text", "source_location", "metadata"]
    for k in required:
        _assert(k in d, f"Missing key in Chunk.to_dict: {k}")


def test_chunk_params_validation():
    """chunk_text raises ValueError on invalid chunk_size or chunk_overlap."""
    from ingestion.chunker import chunk_text

    # chunk_size <= 0
    try:
        chunk_text("Some text", chunk_size=0)
        raise AssertionError("Expected ValueError for chunk_size=0")
    except ValueError:
        pass

    # chunk_overlap < 0
    try:
        chunk_text("Some text", chunk_size=100, chunk_overlap=-5)
        raise AssertionError("Expected ValueError for chunk_overlap < 0")
    except ValueError:
        pass

    # chunk_overlap >= chunk_size
    try:
        chunk_text("Some text", chunk_size=100, chunk_overlap=100)
        raise AssertionError("Expected ValueError for chunk_overlap >= chunk_size")
    except ValueError:
        pass


def test_chunk_empty_document():
    """chunk_document on an empty document returns an empty list."""
    from ingestion.models import make_document
    from ingestion.chunker import chunk_document

    doc = make_document(
        filename="empty.txt",
        file_type=".txt",
        source="empty.txt",
        text="   \n\n\t  ",
    )
    chunks = chunk_document(doc)
    _assert(chunks == [], "Empty document should yield 0 chunks")


def test_chunk_document_short_text():
    """Document shorter than chunk_size yields exactly 1 chunk with full content."""
    from ingestion.models import make_document
    from ingestion.chunker import chunk_document

    short_text = "This is a short policy statement consisting of a single sentence."
    doc = make_document(
        filename="short.txt",
        file_type=".txt",
        source="short.txt",
        text=short_text,
    )
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=80)
    _assert(len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}")
    _assert(chunks[0].chunk_index == 0)
    _assert(chunks[0].chunk_id == f"{doc.document_id}_0")
    _assert(chunks[0].text == short_text)
    _assert(chunks[0].metadata["token_start"] == 0)
    _assert(chunks[0].metadata["token_count"] > 0)


def test_chunk_document_multiple_overlapping_chunks():
    """Document with 1500 tokens generates multiple sequential overlapping chunks."""
    import tiktoken
    from ingestion.models import make_document
    from ingestion.chunker import chunk_document

    # Generate a repetitive paragraph of known size
    paragraph = (
        "Enterprise API authentication requires OAuth 2.0 bearer tokens passed in headers. "
        "Each request must include a valid timestamp, signature, and client certificate. "
    )
    # Replicate to exceed 1000 tokens
    long_text = "\n\n".join([f"Section {i}: " + paragraph * 5 for i in range(15)])
    
    doc = make_document(
        filename="long_doc.txt",
        file_type=".txt",
        source="long_doc.txt",
        text=long_text,
    )

    chunk_size = 300
    chunk_overlap = 50
    chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    _assert(len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}")

    # Verify sequential indices & deterministic IDs
    for idx, c in enumerate(chunks):
        _assert(c.chunk_index == idx, f"Chunk index mismatch at {idx}")
        _assert(c.chunk_id == f"{doc.document_id}_{idx}", f"Chunk ID mismatch at {idx}")
        _assert(c.document_id == doc.document_id)
        _assert(c.document_name == doc.filename)
        _assert(c.token_count <= chunk_size, f"Chunk {idx} exceeded max token size")

    # Paragraph-aware chunker: consecutive chunks share trailing paragraphs (overlap),
    # not a strict token-window token_start arithmetic.
    for i in range(len(chunks) - 1):
        c1 = chunks[i]
        c2 = chunks[i + 1]
        _assert(c1.chunk_index < c2.chunk_index)
        _assert(c1.metadata["token_end"] > 0)
        _assert(c2.metadata["token_start"] >= 0)


def test_chunk_document_metadata_preservation():
    """chunk_document preserves parent document metadata into each chunk."""
    from ingestion.models import make_document
    from ingestion.chunker import chunk_document

    doc = make_document(
        filename="policy.pdf",
        file_type=".pdf",
        source="data/hr/leave_policy.pdf",
        text="Sample policy text for metadata preservation verification.",
        metadata={"department": "HR", "page_count": 1, "has_text": True},
    )
    chunks = chunk_document(doc, chunk_size=600, chunk_overlap=80)
    _assert(len(chunks) == 1)
    meta = chunks[0].metadata
    _assert(meta.get("department") == "HR")
    _assert(meta.get("page_count") == 1)
    _assert(meta.get("has_text") is True)
    _assert(meta.get("token_start") == 0)
    _assert("token_count" in meta)


def test_text_chunker_wrapper_class():
    """TextChunker class exposes chunk_text and chunk_document with configured defaults."""
    from ingestion.models import make_document
    from ingestion.chunker import TextChunker

    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    res_text = chunker.chunk_text("Word " * 250)
    _assert(len(res_text) > 1)

    doc = make_document(filename="test.txt", file_type=".txt", source="test.txt", text="Word " * 250)
    res_doc = chunker.chunk_document(doc)
    _assert(len(res_doc) == len(res_text))


def test_chunk_real_sample_documents():
    """Run chunk_document on all cleaned real sample documents and verify token metrics."""
    from ingestion.cleaner import clean_document
    from ingestion.chunker import chunk_document

    sample_files = [_SAMPLE_PDF, _SAMPLE_DOCX, _SAMPLE_TXT, _SAMPLE_CSV]
    print("\n--- Real Sample Chunking Metrics ---")
    for sample in sample_files:
        raw = extract_document(sample)
        cleaned = clean_document(raw)
        chunks = chunk_document(cleaned, chunk_size=600, chunk_overlap=80)

        _assert(len(chunks) >= 1, f"Failed to produce chunks for {sample}")
        total_tokens = sum(c.token_count for c in chunks)
        print(
            f"  File: {os.path.basename(sample):<25} | "
            f"Chunks: {len(chunks):<2} | "
            f"Chunk[0] tokens: {chunks[0].token_count:<4} | "
            f"Total tokens: {total_tokens:<4}"
        )
        for c in chunks:
            _assert(c.token_count <= 600, f"Token limit exceeded in {sample}: {c.token_count}")


# ---------------------------------------------------------------------------
# Phase 3.7 — Ingestion provenance validation tests
# ---------------------------------------------------------------------------

def test_validate_chunk_valid():
    """validate_chunk passes on a well-formed Chunk with matching Document."""
    from ingestion.models import make_chunk, make_document
    from ingestion.validator import validate_chunk

    doc = make_document(filename="policy.pdf", file_type=".pdf", source="data/hr/policy.pdf", text="Text")
    chunk = make_chunk(
        document_id=doc.document_id,
        document_name=doc.filename,
        chunk_index=0,
        text="Valid chunk text.",
        source_location=doc.source,
    )
    validate_chunk(chunk, document=doc)


def test_validate_chunk_missing_document_id():
    """validate_chunk raises ProvenanceError when document_id is blank."""
    from ingestion.models import Chunk
    from ingestion.validator import ProvenanceError, validate_chunk

    chunk = Chunk(
        chunk_id="chk_0",
        document_id="   ",
        document_name="test.txt",
        chunk_index=0,
        text="Text",
        source_location="test.txt",
    )
    try:
        validate_chunk(chunk)
        raise AssertionError("Expected ProvenanceError for blank document_id")
    except ProvenanceError:
        pass


def test_validate_chunk_missing_chunk_id():
    """validate_chunk raises ProvenanceError when chunk_id is blank."""
    from ingestion.models import Chunk
    from ingestion.validator import ProvenanceError, validate_chunk

    chunk = Chunk(
        chunk_id="   ",
        document_id="doc-1",
        document_name="test.txt",
        chunk_index=0,
        text="Text",
        source_location="test.txt",
    )
    try:
        validate_chunk(chunk)
        raise AssertionError("Expected ProvenanceError for blank chunk_id")
    except ProvenanceError:
        pass


def test_validate_chunk_empty_text():
    """validate_chunk raises ProvenanceError when chunk text is empty/whitespace."""
    from ingestion.models import make_chunk
    from ingestion.validator import ProvenanceError, validate_chunk

    chunk = make_chunk(
        document_id="doc-1",
        document_name="test.txt",
        chunk_index=0,
        text="   \n\t ",
        source_location="test.txt",
    )
    try:
        validate_chunk(chunk)
        raise AssertionError("Expected ProvenanceError for empty chunk text")
    except ProvenanceError:
        pass


def test_validate_chunk_missing_source_location():
    """validate_chunk raises ProvenanceError when source_location is missing."""
    from ingestion.models import make_chunk
    from ingestion.validator import ProvenanceError, validate_chunk

    chunk = make_chunk(
        document_id="doc-1",
        document_name="test.txt",
        chunk_index=0,
        text="Text",
        source_location="   ",
    )
    try:
        validate_chunk(chunk)
        raise AssertionError("Expected ProvenanceError for empty source_location")
    except ProvenanceError:
        pass


def test_validate_chunk_document_reference_mismatch():
    """validate_chunk raises ProvenanceError on document_id or source_location mismatch."""
    from ingestion.models import make_chunk, make_document
    from ingestion.validator import ProvenanceError, validate_chunk

    doc_a = make_document(filename="a.txt", file_type=".txt", source="path/a.txt", text="Text")
    doc_b = make_document(filename="b.txt", file_type=".txt", source="path/b.txt", text="Text")

    # Mismatched document_id
    chunk_mismatch_id = make_chunk(
        document_id=doc_a.document_id,
        document_name=doc_b.filename,
        chunk_index=0,
        text="Content",
        source_location=doc_b.source,
    )
    try:
        validate_chunk(chunk_mismatch_id, document=doc_b)
        raise AssertionError("Expected ProvenanceError for document_id mismatch")
    except ProvenanceError:
        pass

    # Mismatched source_location
    chunk_mismatch_src = make_chunk(
        document_id=doc_a.document_id,
        document_name=doc_a.filename,
        chunk_index=0,
        text="Content",
        source_location="different/path.txt",
    )
    try:
        validate_chunk(chunk_mismatch_src, document=doc_a)
        raise AssertionError("Expected ProvenanceError for source_location mismatch")
    except ProvenanceError:
        pass


def test_validate_chunks_duplicate_chunk_ids():
    """validate_chunks raises ProvenanceError when duplicate chunk_ids exist."""
    from ingestion.models import make_chunk
    from ingestion.validator import ProvenanceError, validate_chunks

    c1 = make_chunk(document_id="doc-1", document_name="t.txt", chunk_index=0, text="A", source_location="t.txt", chunk_id="same_id")
    c2 = make_chunk(document_id="doc-1", document_name="t.txt", chunk_index=1, text="B", source_location="t.txt", chunk_id="same_id")
    try:
        validate_chunks([c1, c2])
        raise AssertionError("Expected ProvenanceError for duplicate chunk IDs")
    except ProvenanceError:
        pass


def test_validate_chunks_non_sequential_indices():
    """validate_chunks raises ProvenanceError when chunk indices are non-sequential."""
    from ingestion.models import make_chunk
    from ingestion.validator import ProvenanceError, validate_chunks

    c1 = make_chunk(document_id="doc-1", document_name="t.txt", chunk_index=0, text="A", source_location="t.txt")
    c2 = make_chunk(document_id="doc-1", document_name="t.txt", chunk_index=5, text="B", source_location="t.txt")
    try:
        validate_chunks([c1, c2])
        raise AssertionError("Expected ProvenanceError for non-sequential indices")
    except ProvenanceError:
        pass


def test_validate_provenance_missing_format_metadata():
    """validate_provenance raises ProvenanceError when required format metadata is missing."""
    from ingestion.models import make_chunk, make_document
    from ingestion.validator import ProvenanceError, validate_provenance

    # PDF missing page_count
    bad_pdf_doc = make_document(filename="doc.pdf", file_type=".pdf", source="doc.pdf", text="Text", metadata={})
    chunk = make_chunk(
        document_id=bad_pdf_doc.document_id,
        document_name="doc.pdf",
        chunk_index=0,
        text="Text",
        source_location="doc.pdf",
        metadata={"token_start": 0, "token_end": 5},
    )
    try:
        validate_provenance([chunk], bad_pdf_doc)
        raise AssertionError("Expected ProvenanceError for missing PDF page_count")
    except ProvenanceError:
        pass


def test_validate_provenance_all_sample_documents():
    """validate_provenance passes on all real sample documents through extract -> clean -> chunk."""
    from ingestion.cleaner import clean_document
    from ingestion.chunker import chunk_document
    from ingestion.validator import validate_provenance

    sample_files = [_SAMPLE_PDF, _SAMPLE_DOCX, _SAMPLE_TXT, _SAMPLE_CSV]
    for sample in sample_files:
        doc = extract_document(sample)
        cleaned = clean_document(doc)
        chunks = chunk_document(cleaned)
        validate_provenance(chunks, cleaned)


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # Phase 3.1 — Document model
    ("Document model: valid construction",                      test_document_model_creation_valid),
    ("Document model: empty metadata defaults to dict",         test_document_model_empty_metadata_defaults),
    ("Document model: file_type lower-cased by factory",        test_document_model_file_type_lowercased),
    ("Document model: to_dict() contains all 7 keys",           test_document_model_to_dict),
    ("Document model: empty document_id raises ValueError",     test_document_model_invalid_empty_id),
    ("Document model: file_type without dot raises ValueError", test_document_model_invalid_file_type_no_dot),
    # Phase 3.1 — validate_file_path
    ("validate_file_path: missing file -> FileNotFoundError",   test_validate_missing_file),
    ("validate_file_path: directory path -> IsADirectoryError", test_validate_directory_path),
    ("validate_file_path: unsupported extension -> ValueError", test_validate_unsupported_extension),
    ("validate_file_path: all 4 supported extensions pass",     test_validate_supported_extensions_all_pass),
    ("SUPPORTED_EXTENSIONS: exactly 4 extensions",              test_supported_extensions_constant),
    # Phase 3.1 — extract_document (TXT smoke test)
    ("extract_document: returns Document for valid TXT",        test_extract_document_returns_document_instance),
    ("extract_document: missing file -> FileNotFoundError",     test_extract_document_missing_file_raises),
    ("extract_document: unsupported extension -> ValueError",   test_extract_document_unsupported_ext_raises),
    ("extract_document: unique document_id per call",           test_extract_document_unique_ids),
    # Phase 3.2 — PdfExtractor direct
    ("PdfExtractor: returns dict with text + metadata",         test_pdf_extractor_returns_dict_keys),
    ("PdfExtractor: text is non-empty for sample PDF",          test_pdf_extractor_text_non_empty),
    ("PdfExtractor: page_count captured in metadata",           test_pdf_extractor_page_count_captured),
    ("PdfExtractor: all required metadata keys present",        test_pdf_extractor_metadata_required_keys),
    ("PdfExtractor: metadata format == 'pdf'",                  test_pdf_extractor_format_is_pdf),
    ("PdfExtractor: has_text True for text-based PDF",          test_pdf_extractor_has_text_true),
    ("PdfExtractor: known content in extracted text",           test_pdf_extractor_known_content),
    ("PdfExtractor: empty_page_numbers is a list",              test_pdf_extractor_empty_page_list_type),
    ("PdfExtractor: missing file -> FileNotFoundError",         test_pdf_extractor_missing_file_raises),
    # Phase 3.2 — extract_document() PDF integration
    ("extract_document PDF: returns Document instance",         test_extract_document_pdf_returns_document),
    ("extract_document PDF: file_type == '.pdf'",               test_extract_document_pdf_file_type),
    ("extract_document PDF: filename preserved",                test_extract_document_pdf_filename_preserved),
    ("extract_document PDF: source path preserved",             test_extract_document_pdf_source_preserved),
    ("extract_document PDF: text non-empty",                    test_extract_document_pdf_text_non_empty),
    ("extract_document PDF: page_count in metadata",            test_extract_document_pdf_page_count_in_metadata),
    ("extract_document PDF: document_id present",               test_extract_document_pdf_document_id_present),
    ("extract_document PDF: created_at is ISO 8601",            test_extract_document_pdf_created_at_present),
    # Phase 3.3 — DocxExtractor direct
    ("DocxExtractor: returns dict with text + metadata",        test_docx_extractor_returns_dict_keys),
    ("DocxExtractor: text is non-empty for sample DOCX",        test_docx_extractor_text_non_empty),
    ("DocxExtractor: all required metadata keys present",       test_docx_extractor_metadata_required_keys),
    ("DocxExtractor: paragraph_count captured in metadata",     test_docx_extractor_paragraph_count_captured),
    ("DocxExtractor: table_count captured in metadata",         test_docx_extractor_table_count_captured),
    ("DocxExtractor: metadata format == 'docx'",                test_docx_extractor_format_is_docx),
    ("DocxExtractor: has_text True for sample DOCX",            test_docx_extractor_has_text_true),
    ("DocxExtractor: known content in extracted text",          test_docx_extractor_known_content),
    ("DocxExtractor: table content formatted into text",        test_docx_extractor_table_extraction),
    ("DocxExtractor: missing file -> FileNotFoundError",        test_docx_extractor_missing_file_raises),
    # Phase 3.3 — extract_document() DOCX integration
    ("extract_document DOCX: returns Document instance",        test_extract_document_docx_returns_document),
    ("extract_document DOCX: file_type == '.docx'",             test_extract_document_docx_file_type),
    ("extract_document DOCX: filename preserved",               test_extract_document_docx_filename_preserved),
    ("extract_document DOCX: source path preserved",            test_extract_document_docx_source_preserved),
    ("extract_document DOCX: text non-empty",                   test_extract_document_docx_text_non_empty),
    ("extract_document DOCX: paragraph_count in metadata",      test_extract_document_docx_paragraph_count_in_metadata),
    ("extract_document DOCX: table_count in metadata",          test_extract_document_docx_table_count_in_metadata),
    ("extract_document DOCX: document_id present",              test_extract_document_docx_document_id_present),
    ("extract_document DOCX: created_at is ISO 8601",           test_extract_document_docx_created_at_present),
    # Phase 3.4 — TxtExtractor direct
    ("TxtExtractor: returns dict with text + metadata",         test_txt_extractor_returns_dict_keys),
    ("TxtExtractor: text is non-empty for sample TXT",          test_txt_extractor_text_non_empty),
    ("TxtExtractor: all required metadata keys present",        test_txt_extractor_metadata_required_keys),
    ("TxtExtractor: line_count captured in metadata",           test_txt_extractor_line_count_captured),
    ("TxtExtractor: metadata format == 'txt'",                  test_txt_extractor_format_is_txt),
    ("TxtExtractor: has_text True for sample TXT",              test_txt_extractor_has_text_true),
    ("TxtExtractor: known content in extracted text",           test_txt_extractor_known_content),
    ("TxtExtractor: safely handles invalid UTF-8 bytes",        test_txt_extractor_decoding_safe),
    ("TxtExtractor: missing file -> FileNotFoundError",         test_txt_extractor_missing_file_raises),
    # Phase 3.4 — extract_document() TXT integration
    ("extract_document TXT: returns Document instance",         test_extract_document_txt_returns_document),
    ("extract_document TXT: file_type == '.txt'",               test_extract_document_txt_file_type),
    ("extract_document TXT: filename preserved",                test_extract_document_txt_filename_preserved),
    ("extract_document TXT: source path preserved",             test_extract_document_txt_source_preserved),
    ("extract_document TXT: text non-empty",                    test_extract_document_txt_text_non_empty),
    ("extract_document TXT: line_count in metadata",            test_extract_document_txt_line_count_in_metadata),
    ("extract_document TXT: document_id present",               test_extract_document_txt_document_id_present),
    ("extract_document TXT: created_at is ISO 8601",            test_extract_document_txt_created_at_present),
    # Phase 3.4 — CsvExtractor direct
    ("CsvExtractor: returns dict with text + metadata",         test_csv_extractor_returns_dict_keys),
    ("CsvExtractor: text is non-empty for sample CSV",          test_csv_extractor_text_non_empty),
    ("CsvExtractor: all required metadata keys present",        test_csv_extractor_metadata_required_keys),
    ("CsvExtractor: row_count captured in metadata",            test_csv_extractor_row_count_captured),
    ("CsvExtractor: column_count captured in metadata",         test_csv_extractor_column_count_captured),
    ("CsvExtractor: column_names captured in metadata",         test_csv_extractor_column_names_captured),
    ("CsvExtractor: metadata format == 'csv'",                  test_csv_extractor_format_is_csv),
    ("CsvExtractor: has_text True for sample CSV",              test_csv_extractor_has_text_true),
    ("CsvExtractor: semantic row formatting",                  test_csv_extractor_semantic_row_formatting),
    ("CsvExtractor: known content in extracted text",           test_csv_extractor_known_content),
    ("CsvExtractor: missing file -> FileNotFoundError",         test_csv_extractor_missing_file_raises),
    # Phase 3.4 — extract_document() CSV integration
    ("extract_document CSV: returns Document instance",         test_extract_document_csv_returns_document),
    ("extract_document CSV: file_type == '.csv'",               test_extract_document_csv_file_type),
    ("extract_document CSV: filename preserved",                test_extract_document_csv_filename_preserved),
    ("extract_document CSV: source path preserved",             test_extract_document_csv_source_preserved),
    ("extract_document CSV: text non-empty",                    test_extract_document_csv_text_non_empty),
    ("extract_document CSV: row_count in metadata",             test_extract_document_csv_row_count_in_metadata),
    ("extract_document CSV: column_count in metadata",          test_extract_document_csv_column_count_in_metadata),
    ("extract_document CSV: column_names in metadata",          test_extract_document_csv_column_names_in_metadata),
    ("extract_document CSV: document_id present",               test_extract_document_csv_document_id_present),
    ("extract_document CSV: created_at is ISO 8601",            test_extract_document_csv_created_at_present),
    # Phase 3.5 — Text cleaning and normalization
    ("TextCleaner: repeated whitespace",                        test_clean_text_repeated_whitespace),
    ("TextCleaner: excessive blank lines collapsed",            test_clean_text_excessive_blank_lines),
    ("TextCleaner: mixed CRLF/CR line endings normalized",      test_clean_text_mixed_line_endings),
    ("TextCleaner: preserves headings and numbering",           test_clean_text_preserves_headings_and_numbering),
    ("TextCleaner: preserves punctuation and symbols",          test_clean_text_preserves_punctuation_and_symbols),
    ("TextCleaner: preserves table rows and markup",            test_clean_text_preserves_table_rows),
    ("TextCleaner: clean_text is idempotent",                   test_clean_text_idempotence),
    ("TextCleaner: empty/blank input handled safely",           test_clean_text_empty_input),
    ("TextCleaner: clean_document updates text and metadata",   test_clean_document_updates_metadata_and_text),
    ("TextCleaner: clean_document on all 4 real sample files",  test_clean_document_on_all_real_samples),
    # Phase 3.6 — Token-aware chunking
    ("Chunk model: valid construction",                         test_chunk_model_valid_construction),
    ("Chunk model: invalid construction raises ValueError",     test_chunk_model_invalid_construction),
    ("Chunk model: to_dict contains all fields",                test_chunk_model_to_dict),
    ("Chunker: parameter validation guards",                    test_chunk_params_validation),
    ("Chunker: empty document yields 0 chunks",                 test_chunk_empty_document),
    ("Chunker: short document yields 1 chunk",                  test_chunk_document_short_text),
    ("Chunker: multiple overlapping chunks verified",           test_chunk_document_multiple_overlapping_chunks),
    ("Chunker: metadata preservation verified",                 test_chunk_document_metadata_preservation),
    ("Chunker: TextChunker wrapper class verified",             test_text_chunker_wrapper_class),
    ("Chunker: real sample document chunking metrics",          test_chunk_real_sample_documents),
    # Phase 3.7 — Ingestion provenance validation
    ("Provenance: valid chunk and document reference",          test_validate_chunk_valid),
    ("Provenance: missing document_id raises ProvenanceError",  test_validate_chunk_missing_document_id),
    ("Provenance: missing chunk_id raises ProvenanceError",     test_validate_chunk_missing_chunk_id),
    ("Provenance: empty chunk text raises ProvenanceError",     test_validate_chunk_empty_text),
    ("Provenance: missing source_location raises ProvenanceError", test_validate_chunk_missing_source_location),
    ("Provenance: document reference mismatch raises ProvenanceError", test_validate_chunk_document_reference_mismatch),
    ("Provenance: duplicate chunk IDs raise ProvenanceError",   test_validate_chunks_duplicate_chunk_ids),
    ("Provenance: non-sequential indices raise ProvenanceError", test_validate_chunks_non_sequential_indices),
    ("Provenance: missing format metadata raises ProvenanceError", test_validate_provenance_missing_format_metadata),
    ("Provenance: full pipeline validation on all sample docs", test_validate_provenance_all_sample_documents),
]

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3.1 - 3.7 - Ingestion Pipeline & Provenance Tests")
    print("=" * 60)

    for test_name, test_fn in ALL_TESTS:
        _run(test_name, test_fn)

    print()
    print(f"Results: {_PASS} passed, {_FAIL} failed out of {len(ALL_TESTS)} tests.")

    if _FAIL > 0:
        print("\nFailed tests:")
        for name, exc in _ERRORS:
            print(f"  - {name}: {exc}")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)





