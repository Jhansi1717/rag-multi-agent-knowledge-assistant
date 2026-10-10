import faiss
import pytest
from fastapi.testclient import TestClient
import app as app_module
from app import app

client = TestClient(app)

@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    vector_store = app_module.vector_store
    vector_store.index = faiss.IndexFlatL2(vector_store.dimension)
    vector_store.metadata = {}
    vector_store._indexed_sources = {}
    vector_store._last_reindexed_source = None
    vector_store.index_path = str(tmp_path / "index.faiss")
    vector_store.meta_path = str(tmp_path / "metadata.json")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(upload_dir))


def test_semantic_retrieval(tmp_path, isolated_data):
    
    # 1. Upload a document
    test_file_path = tmp_path / "retrieval_test_doc.txt"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems. ")
        f.write("Specific applications of AI include expert systems, natural language processing, speech recognition and machine vision. ")
        f.write("Machine learning is a subset of artificial intelligence.")
        
    print("\n--- Uploading Document ---")
    with open(test_file_path, "rb") as f:
        upload_resp = client.post("/upload", files={"file": ("retrieval_test_doc.txt", f, "text/plain")})
        
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    print(f"Uploaded successfully. Document ID: {upload_data['document_id']}")
    print(f"Generated {upload_data['chunk_count']} chunks.")
    
    # 2. Query the semantic retrieval endpoint
    print("\n--- Performing Semantic Retrieval ---")
    query_payload = {
        "query": "What is machine learning related to?",
        "top_k": 2
    }
    
    retrieve_resp = client.post("/retrieve", json=query_payload)
    assert retrieve_resp.status_code == 200
    
    retrieve_data = retrieve_resp.json()
    print(f"Query: {retrieve_data['query']}")
    
    results = retrieve_data['results']
    print(f"Returned {len(results)} results:")
    
    for idx, res in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"  Text: {res['text'][:100]}...")
        print(f"  Document: {res['document_name']}")
        print(f"  Score: {res['similarity_score']:.4f}")
        
    assert len(results) > 0, "No results returned"
    
    # The first result should ideally contain 'Machine learning'
    assert 'Machine learning' in results[0]['text'], "Top result is incorrect"
    
    print("\nSemantic Retrieval works successfully via API.")

if __name__ == "__main__":
    test_semantic_retrieval()
