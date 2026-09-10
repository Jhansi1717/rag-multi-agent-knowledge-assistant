"""M1.3 — Focused ingestion pipeline tests (all formats, chunking, metadata, dedupe)."""

import os
import uuid

import pandas as pd
import pytest
from docx import Document as DocxDocument
from fpdf import FPDF

from ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunker,
    ingest_file,
    index_file,
)
from ingestion.models import SUPPORTED_EXTENSIONS
from vector_store.store import VectorStore


@pytest.fixture
def tmp_data_dir(tmp_path):
    hr = tmp_path / "data" / "hr"
    hr.mkdir(parents=True)
    return hr


@pytest.fixture
def embedder():
    """Shared lightweight embedder for vector store tests."""
    from vector_store.embeddings import EmbeddingProvider
    return EmbeddingProvider()


def _write_pdf(path, lines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in lines:
        pdf.multi_cell(0, 8, txt=line)
    pdf.output(str(path))


def _write_docx(path, paragraphs, table_rows=None):
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r, row in enumerate(table_rows):
            for c, val in enumerate(row):
                table.rows[r].cells[c].text = str(val)
    doc.save(str(path))


def _write_txt(path, content):
    path.write_text(content, encoding="utf-8")


def _write_csv(path, rows, columns):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _assert_chunk_metadata(chunk, filename, file_type, domain):
    meta = chunk.to_dict()
    assert meta["chunk_id"]
    assert meta["document_id"]
    assert meta["chunk_index"] >= 0
    assert meta["filename"] == filename
    assert meta["file_type"] == file_type
    assert meta["domain"] == domain
    assert meta["text"].strip()
    assert chunk.metadata.get("token_count", 0) > 0


class TestFormatExtraction:
    def test_pdf_extraction_and_page_metadata(self, tmp_data_dir):
        path = tmp_data_dir / "policy.pdf"
        _write_pdf(path, ["Page one content about leave.", "Page two content about sick days."])
        doc, chunks = ingest_file(str(path), domain="HR")
        assert doc.file_type == ".pdf"
        assert doc.metadata["page_count"] >= 1
        assert "page_texts" in doc.metadata
        assert len(chunks) >= 1
        assert any(c.metadata.get("page_number") or c.metadata.get("page_start") for c in chunks)

    def test_docx_paragraphs_and_tables(self, tmp_data_dir):
        path = tmp_data_dir / "api.docx"
        _write_docx(
            path,
            ["REST API overview.", "Use DELETE to remove resources."],
            table_rows=[["Method", "Action"], ["DELETE", "Remove"]],
        )
        doc, chunks = ingest_file(str(path), domain="Software")
        assert doc.file_type == ".docx"
        assert doc.metadata["paragraph_count"] >= 2
        assert doc.metadata["table_count"] >= 1
        assert "DELETE" in doc.text
        assert len(chunks) >= 1

    def test_txt_utf8(self, tmp_data_dir):
        path = tmp_data_dir / "notes.txt"
        _write_txt(path, "Line one: café résumé\n\nLine two: naïve coöperate")
        doc, chunks = ingest_file(str(path), domain="HR")
        assert doc.file_type == ".txt"
        assert doc.metadata["line_count"] >= 2
        assert "café" in doc.text
        assert len(chunks) >= 1

    def test_csv_semantic_text(self, tmp_data_dir):
        path = tmp_data_dir / "allowance.csv"
        _write_csv(
            path,
            [["Annual", 20, 5], ["Sick", 10, 0]],
            ["Leave_Type", "Days_Allowed", "Carry_Forward_Limit"],
        )
        doc, chunks = ingest_file(str(path), domain="HR")
        assert doc.file_type == ".csv"
        assert doc.metadata["row_count"] == 2
        assert "Leave_Type: Annual" in doc.text
        assert any(c.metadata.get("row_number") or c.metadata.get("row_start") for c in chunks)


class TestChunking:
    def test_defaults_within_m13_range(self):
        assert 500 <= DEFAULT_CHUNK_SIZE <= 800
        assert 50 <= DEFAULT_CHUNK_OVERLAP <= 100

    def test_paragraph_boundary_preferred(self, tmp_data_dir):
        path = tmp_data_dir / "paragraphs.txt"
        paras = [f"Paragraph {i}. " + ("word " * 120) for i in range(8)]
        _write_txt(path, "\n\n".join(paras))
        _, chunks = ingest_file(str(path))
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "\n\n" in chunk.text or len(chunks) == 1
            assert not chunk.text.startswith(" ")

    def test_chunk_metadata_fields(self, tmp_data_dir):
        path = tmp_data_dir / "meta.pdf"
        _write_pdf(path, ["Employees receive twenty days of annual leave per year."])
        _, chunks = ingest_file(str(path), domain="HR")
        assert chunks
        _assert_chunk_metadata(chunks[0], "meta.pdf", ".pdf", "HR")


class TestVectorIndexing:
    def test_index_and_search(self, tmp_data_dir, embedder):
        path = tmp_data_dir / "leave.txt"
        _write_txt(path, "Employees receive twenty days of annual leave per year.")
        index_path = tmp_data_dir / "idx.faiss"
        meta_path = tmp_data_dir / "meta.json"
        store = VectorStore(
            index_path=str(index_path),
            meta_path=str(meta_path),
            embedder=embedder,
        )
        result = index_file(str(path), store, domain="HR", save=True)
        assert result["chunks_added"] >= 1
        store.load()
        hits = store.search("How many leave days?", top_k=1)
        assert hits
        assert hits[0]["filename"] == "leave.txt"
        assert hits[0]["domain"] == "HR"
        assert "embedding_model" in hits[0]

    def test_no_duplicate_indexing_same_source(self, tmp_data_dir, embedder):
        path = tmp_data_dir / "dup.txt"
        _write_txt(path, "Duplicate document content for indexing test.")
        index_path = tmp_data_dir / "dup.faiss"
        meta_path = tmp_data_dir / "dup_meta.json"
        store = VectorStore(
            index_path=str(index_path),
            meta_path=str(meta_path),
            embedder=embedder,
        )
        first = index_file(str(path), store, save=False)
        count_after_first = store.index.ntotal
        second = index_file(str(path), store, save=True)
        assert second["reindexed"] is True
        assert store.index.ntotal == count_after_first
        assert first["chunks_added"] == second["chunks_added"]


class TestValidation:
    def test_supported_extensions(self):
        assert SUPPORTED_EXTENSIONS == frozenset({".pdf", ".docx", ".txt", ".csv"})
