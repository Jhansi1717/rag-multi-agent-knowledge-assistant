import os
import pytest
from fastapi.testclient import TestClient
from app import app, vector_store
import app as app_module
from agents.models import AgentError, RetrievalHit, RetrievalResult, ResponseResult
from agents.orchestrator import Orchestrator

client = TestClient(app)


def test_documents_returns_grouped_indexed_metadata(monkeypatch):
    monkeypatch.setattr(vector_store, "metadata", {
        "0": {
            "document_id": "doc-1",
            "document_name": "guide.txt",
            "filename": "guide.txt",
            "file_type": ".txt",
            "domain": "Software",
            "text": "first",
            "embedding_model": "secret-model-detail",
            "source_location": "internal/path/guide.txt",
        },
        "1": {
            "document_id": "doc-1",
            "document_name": "guide.txt",
            "file_type": ".txt",
            "domain": "Software",
            "text": "second",
        },
        "2": {
            "document_id": "doc-2",
            "document_name": "policy.pdf",
            "metadata": {"file_type": ".pdf", "domain": "HR"},
            "text": "third",
        },
    })

    response = client.get("/documents")

    assert response.status_code == 200
    documents = response.json()["documents"]
    assert documents == [
        {
            "document_id": "doc-1",
            "document_name": "guide.txt",
            "file_type": ".txt",
            "domain": "Software",
            "chunk_count": 2,
            "status": "indexed",
        },
        {
            "document_id": "doc-2",
            "document_name": "policy.pdf",
            "file_type": ".pdf",
            "domain": "HR",
            "chunk_count": 1,
            "status": "indexed",
        },
    ]
    assert "source_location" not in response.text
    assert "embedding_model" not in response.text


def test_documents_prefers_original_filename_and_falls_back_for_legacy_metadata(monkeypatch):
    monkeypatch.setattr(vector_store, "metadata", {
        "0": {
            "document_id": "doc-original",
            "document_name": "cda8ab78-8948-48e5-a92c-c4b5dd8fffa8.pdf",
            "filename": "cda8ab78-8948-48e5-a92c-c4b5dd8fffa8.pdf",
            "metadata": {"original_filename": "annual_report.pdf", "file_type": ".pdf"},
        },
        "1": {
            "document_id": "doc-legacy",
            "document_name": "legacy-guide.txt",
            "filename": "legacy-guide.txt",
            "metadata": {"file_type": ".txt"},
        },
    })

    response = client.get("/documents")

    assert response.status_code == 200
    documents = {item["document_id"]: item for item in response.json()["documents"]}
    assert documents["doc-original"]["document_name"] == "annual_report.pdf"
    assert documents["doc-legacy"]["document_name"] == "legacy-guide.txt"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check passed.")


def test_api_reports_resolved_query_and_matching_retrieval(monkeypatch):
    hit = RetrievalHit(
        rank=1,
        relevance_score=0.8,
        distance_score=0.25,
        text="Git feature branch evidence.",
        document_id="doc-git",
        filename="git_workflow.txt",
        chunk_id="git-0",
        metadata={"domain": "Software Engineering"},
    )
    retrieval = RetrievalResult(
        query="Create a Git feature branch",
        query_type="procedural",
        top_k=3,
        results=[hit],
        retrieval_confidence=0.8,
        sufficient_evidence=True,
    )
    response = ResponseResult(
        answer="Grounded.",
        confidence=0.8,
        grounded=True,
        query_type="procedural",
        request_id="internal-request",
        retrieval=retrieval,
        retrieval_hits=[hit],
    )

    class FixedOrchestrator:
        def handle(self, **kwargs):
            return response

    monkeypatch.setattr(app_module, "orchestrator", FixedOrchestrator())
    result = client.post(
        "/retrieve",
        json={"query": " Create a Git feature branch ", "top_k": 3},
    )

    assert result.status_code == 200
    payload = result.json()
    assert payload["query"] == " Create a Git feature branch "
    assert payload["resolved_query"] == retrieval.query
    assert payload["retrieval"]["results"][0]["chunk_id"] == hit.chunk_id


def test_api_preserves_stage_error_payload(monkeypatch):
    response = ResponseResult(
        answer="The request could not be completed.",
        no_information_found=True,
        status="error",
        request_id="internal-request",
        error=AgentError(
            agent="retrieval",
            code="RETRIEVAL_ERROR",
            message="retriever unavailable",
        ),
    )

    class FixedOrchestrator:
        def handle(self, **kwargs):
            return response

    monkeypatch.setattr(app_module, "orchestrator", FixedOrchestrator())
    result = client.post("/chat", json={"query": "Find a procedure"})

    assert result.status_code == 500
    payload = result.json()
    assert payload["status"] == "error"
    assert payload["error"]["agent"] == "retrieval"
    assert payload["error"]["code"] == "RETRIEVAL_ERROR"
    assert payload["error"]["message"] == "retriever unavailable"

@pytest.fixture
def isolated_upload_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vector_store, "index", type(vector_store.index)(vector_store.dimension)
    )
    monkeypatch.setattr(vector_store, "metadata", {})
    monkeypatch.setattr(vector_store, "_indexed_sources", {})
    monkeypatch.setattr(vector_store, "_last_reindexed_source", None)
    monkeypatch.setattr(vector_store, "index_path", str(tmp_path / "index.faiss"))
    monkeypatch.setattr(vector_store, "meta_path", str(tmp_path / "metadata.json"))
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(upload_dir))

def test_upload_supported_file(tmp_path, isolated_upload_data):
    test_file_path = tmp_path / "test_upload.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document for ingestion pipeline validation. " * 50)
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.txt", f, "text/plain")})
        
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_upload.txt"
    assert data["file_type"] == ".txt"
    assert data["chunk_count"] > 0
    assert data["metadata"]["line_count"] == 1
    assert data["metadata"]["format"] == "txt"
    print("Supported file upload passed. Metadata:", data)


def test_same_filename_upload_replaces_chunks_and_returns_provenance(
    tmp_path, isolated_upload_data, monkeypatch
):
    monkeypatch.setattr(app_module.orchestrator.response_gen, "_client", None)

    def upload(filename, text):
        path = tmp_path / f"source-{text.split()[0]}.txt"
        path.write_text(text, encoding="utf-8")
        with path.open("rb") as source:
            return client.post(
                "/upload",
                files={"file": (filename, source, "text/plain")},
            )

    first = upload(
        "shared.txt",
        "Git feature branch workflow creates a branch from develop for review.",
    )
    second = upload(
        "other.txt",
        "Hospital patient admission requires photo identification and insurance verification.",
    )
    replacement = upload(
        "shared.txt",
        "Git feature branch replacement content documents the revised review process.",
    )

    assert first.status_code == second.status_code == replacement.status_code == 200
    first_data = first.json()
    second_data = second.json()
    replacement_data = replacement.json()
    assert replacement_data["document_id"] == first_data["document_id"]
    assert replacement_data["reindexed"] is True
    assert replacement_data["filename"] == "shared.txt"
    assert replacement_data["chunks"]
    assert replacement_data["chunks"][0]["document_id"] == first_data["document_id"]
    assert replacement_data["chunks"][0]["domain"] == "Software Engineering"
    assert vector_store.index.ntotal == len(vector_store.metadata)
    assert vector_store.index.ntotal == (
        second_data["chunk_count"] + replacement_data["chunk_count"]
    )
    assert all(
        record["document_id"] != first_data["document_id"]
        or record["chunk_id"] in {item["chunk_id"] for item in replacement_data["chunks"]}
        for record in vector_store.metadata.values()
    )
    assert sorted(vector_store.metadata) == [str(row) for row in range(vector_store.index.ntotal)]

    response = client.post(
        "/chat",
        json={
            "query": "What does Git feature branch replacement document?",
            "session_id": "upload-provenance-regression",
        },
    )
    assert response.status_code == 200
    hit = response.json()["retrieval"]["results"][0]
    assert hit["domain"] == "Software Engineering"
    assert hit["source_location"] == "upload://shared.txt"
    assert hit["metadata"]["domain"] == "Software Engineering"


def test_failed_same_filename_upload_preserves_index(tmp_path, isolated_upload_data):
    good = tmp_path / "good.txt"
    good.write_text(
        "Git feature branch workflow creates a branch from develop for review.",
        encoding="utf-8",
    )
    with good.open("rb") as source:
        uploaded = client.post(
            "/upload", files={"file": ("stable.txt", source, "text/plain")}
        )
    assert uploaded.status_code == 200
    before_metadata = {key: value.copy() for key, value in vector_store.metadata.items()}
    before_vectors = [
        vector_store.index.reconstruct(row).tolist()
        for row in range(vector_store.index.ntotal)
    ]

    with (tmp_path / "broken.txt").open("wb") as broken:
        broken.write(b" \n\t\n")
    with (tmp_path / "broken.txt").open("rb") as source:
        failed = client.post(
            "/upload", files={"file": ("stable.txt", source, "application/pdf")}
        )

    assert failed.status_code == 400
    assert vector_store.index.ntotal == len(vector_store.metadata)
    assert vector_store.metadata == before_metadata
    assert [
        vector_store.index.reconstruct(row).tolist()
        for row in range(vector_store.index.ntotal)
    ] == before_vectors


@pytest.mark.parametrize(
    ("filename", "content", "query", "expected_domain"),
    [
        (
            "git_workflow.txt",
            "Git feature branch workflow: create a branch from develop, "
            "commit the change, and open a pull request for review.",
            "How do I create a Git feature branch?",
            "Software Engineering",
        ),
        (
            "patient_admission.txt",
            "Hospital patient admission requires photo identification and "
            "insurance verification before paperwork is completed.",
            "What documents are needed for patient admission?",
            "Hospital Administration",
        ),
    ],
)
def test_uploaded_document_infers_domain_and_retrieves(
    tmp_path, isolated_upload_data, monkeypatch,
    filename, content, query, expected_domain,
):
    monkeypatch.setattr(app_module.orchestrator.response_gen, "_client", None)
    document_path = tmp_path / filename
    document_path.write_text(content, encoding="utf-8")

    with document_path.open("rb") as source:
        upload = client.post(
            "/upload",
            files={"file": (document_path.name, source, "text/plain")},
        )

    assert upload.status_code == 200
    assert upload.json()["domain"] == expected_domain
    response = client.post(
        "/chat",
        json={
            "query": query,
            "session_id": "uploaded-domain-regression",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == expected_domain
    assert data["retrieval"]["sufficient_evidence"] is True
    assert data["retrieval"]["results"][0]["filename"] == document_path.name


def test_comparative_chat_retrieves_uploaded_targets_from_both_domains(
    tmp_path, isolated_upload_data, monkeypatch
):
    monkeypatch.setattr(app_module.orchestrator.response_gen, "_client", None)
    documents = {
        "software_branch.txt": (
            "Git feature branch workflow: create a branch from develop, "
            "commit changes, and open a pull request for review."
        ),
        "patient_admission.txt": (
            "Hospital patient admission paperwork requires photo identification "
            "and insurance verification before completing the admission form."
        ),
    }
    for filename, text in documents.items():
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        with path.open("rb") as source:
            upload = client.post(
                "/upload",
                files={"file": (filename, source, "text/plain")},
            )
        assert upload.status_code == 200

    response = client.post(
        "/chat",
        json={
            "query": (
                "Compare the Git feature branch workflow with patient admission paperwork."
            ),
            "session_id": "uploaded-cross-domain-comparison",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "comparative"
    assert data["clarification_needed"] is False
    assert {
        hit["filename"] for hit in data["retrieval"]["results"]
    } == set(documents)


def test_multi_page_upload_preserves_page_range_in_api_evidence(
    tmp_path, isolated_upload_data, monkeypatch
):
    from fpdf import FPDF

    monkeypatch.setattr(app_module.orchestrator.response_gen, "_client", None)
    pdf = FPDF()
    pdf.set_font("Arial", size=12)
    for text in (
        "Northstar maintenance calibrates the inverter before sunrise.",
        "Northstar maintenance inspects the battery array after sunset.",
    ):
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    path = tmp_path / "two_page_manual.pdf"
    pdf.output(str(path))

    with path.open("rb") as source:
        upload = client.post(
            "/upload",
            files={"file": (path.name, source, "application/pdf")},
        )
    assert upload.status_code == 200
    assert upload.json()["metadata"]["page_count"] == 2
    assert "page_texts" not in upload.json()["metadata"]
    uploaded_chunk = upload.json()["chunks"][0]
    assert uploaded_chunk["filename"] == path.name
    assert uploaded_chunk["document_id"] == upload.json()["document_id"]
    assert uploaded_chunk["source_location"] == "upload://two_page_manual.pdf"
    assert uploaded_chunk["metadata"]["page_start"] == 1
    assert uploaded_chunk["metadata"]["page_end"] == 2

    response = client.post(
        "/retrieve",
        json={
            "query": "Northstar maintenance calibrates the inverter and inspects the battery array.",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    evidence = response.json()["m3"]["transparency"]["source_evidence"]
    pdf_evidence = next(item for item in evidence if item["document_name"] == path.name)
    assert pdf_evidence["page"] is None
    assert pdf_evidence["page_start"] == 1
    assert pdf_evidence["page_end"] == 2


def test_csv_upload_exposes_row_provenance(tmp_path, isolated_upload_data):
    path = tmp_path / "allowance.csv"
    path.write_text(
        "Medication,Dose\nAcetaminophen,500 mg\nIbuprofen,200 mg\n",
        encoding="utf-8",
    )

    with path.open("rb") as source:
        response = client.post(
            "/upload", files={"file": (path.name, source, "text/csv")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "Hospital Administration"
    assert data["chunks"]
    row_metadata = data["chunks"][0]["metadata"]
    assert row_metadata["row_start"] == 1
    assert row_metadata["row_end"] == 2

def test_upload_unsupported_file(tmp_path):
    test_file_path = tmp_path / "test_upload.md"
    with open(test_file_path, "w") as f:
        f.write("# Hello")
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.md", f, "text/markdown")})
        
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "HTTP_ERROR"
    assert "Unsupported file format" in data["error"]["message"]
    print("Unsupported file validation passed.")
    


def _fake_response(query: str) -> ResponseResult:
    query_type = "factual"
    if query.startswith("How do"):
        query_type = "procedural"
    elif "versus" in query:
        query_type = "comparative"
    elif query == "unclear":
        query_type = "ambiguous"

    if query_type == "ambiguous":
        return ResponseResult(
            answer="Could you provide more detail?",
            confidence=0.8,
            classification_confidence=0.8,
            confidence_level="HIGH",
            query_type=query_type,
            status="clarification_needed",
            request_id="test-request",
        )

    hits = [] if query == "unknown fact" else [
        RetrievalHit(
            rank=1,
            relevance_score=0.9,
            distance_score=0.1,
            text="Supported context.",
            document_id="doc-1",
            filename="guide.txt",
            chunk_id="chunk-1",
        )
    ]
    retrieval = RetrievalResult(
        query=query,
        query_type=query_type,
        top_k=3,
        results=hits,
        retrieval_confidence=0.9 if hits else 0.0,
        sufficient_evidence=bool(hits),
        no_relevant_information=not hits,
    )
    return ResponseResult(
        answer="The requested information is not available in the provided documents."
        if not hits
        else "Supported answer.",
        confidence=retrieval.retrieval_confidence,
        confidence_level="HIGH" if hits else "LOW",
        classification_confidence=0.9,
        grounded=bool(hits),
        no_information_found=not hits,
        query_type=query_type,
        retrieval_hits=hits,
        retrieval=retrieval,
        status="unavailable" if not hits else "answered",
        request_id="test-request",
    )


@pytest.fixture
def mocked_orchestrator(monkeypatch):
    monkeypatch.setattr(
        "app.orchestrator.handle",
        lambda query, session_id, top_k, **kwargs: _fake_response(query),
    )


@pytest.mark.parametrize(
    ("query", "query_type", "status"),
    [
        ("What is a sprint?", "factual", "answered"),
        ("How do I deploy a service?", "procedural", "answered"),
        ("Microservices versus monoliths", "comparative", "answered"),
        ("unclear", "ambiguous", "clarification_needed"),
        ("unknown fact", "factual", "unavailable"),
    ],
)
def test_retrieve_m2_pipeline_responses(
    mocked_orchestrator, query, query_type, status
):
    response = client.post(
        "/retrieve",
        json={"query": query, "top_k": 3, "session_id": "test-session"},
        headers={"X-Request-ID": "test-request"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["request_id"] == "test-request"
    assert data["query"] == query
    assert data["query_type"] == query_type
    assert data["status"] == status
    assert data["classification_confidence"] > 0
    assert "retrieval" in data
    assert "clarification_needed" in data


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "valid", "top_k": 0}, {"query": "valid", "top_k": 101}],
)
def test_retrieve_rejects_invalid_requests(mocked_orchestrator, payload):
    response = client.post("/retrieve", json=payload)
    assert response.status_code == 400
    
if __name__ == "__main__":
    test_health()
    test_upload_supported_file()
    test_upload_unsupported_file()
    print("All tests passed successfully!")
