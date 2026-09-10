import os
import uuid
from vector_store.store import VectorStore

def run_tests():
    os.makedirs("data", exist_ok=True)
    index_path = "data/test_index.faiss"
    meta_path = "data/test_metadata.json"
    
    # Clean up previous tests if they exist
    if os.path.exists(index_path): os.remove(index_path)
    if os.path.exists(meta_path): os.remove(meta_path)
    
    print("--- Initializing VectorStore ---")
    store = VectorStore(index_path=index_path, meta_path=meta_path)
    
    # 1. Add chunks
    print("\n--- Generating Embeddings and Indexing Chunks ---")
    from ingestion.models import make_chunk
    
    chunks = [
        make_chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc1",
            document_name="leave_policy.pdf",
            chunk_index=0,
            text="Employees get 20 days of annual leave.",
            source_location="data/hr/leave_policy.pdf"
        ),
        make_chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc1",
            document_name="leave_policy.pdf",
            chunk_index=1,
            text="Sick leave is 10 days per year.",
            source_location="data/hr/leave_policy.pdf"
        ),
        make_chunk(
            chunk_id=str(uuid.uuid4()),
            document_id="doc2",
            document_name="api_docs.txt",
            chunk_index=0,
            text="Use REST with standard HTTP methods GET, POST.",
            source_location="data/technical/api_docs.txt"
        )
    ]
    store.add_chunks(chunks)
    
    # Validations
    print(f"Number of embeddings in FAISS index: {store.index.ntotal}")
    print(f"Number of metadata entries: {len(store.metadata)}")
    
    assert store.index.ntotal == 3, "Embedding count mismatch"
    assert len(store.metadata) == 3, "Metadata count mismatch"
    print("Count validations passed: number of embeddings equals number of indexed chunks.")
    
    # 2. Save
    print("\n--- Persisting Index and Metadata locally ---")
    store.save()
    assert os.path.exists(index_path), "Index file not saved"
    assert os.path.exists(meta_path), "Metadata file not saved"
    print("FAISS index successfully created and saved.")
    print("Metadata successfully persisted.")
    
    # 3. Reload
    print("\n--- Reloading Index from Disk ---")
    store2 = VectorStore(index_path=index_path, meta_path=meta_path)
    store2.load()
    
    print(f"Reloaded embeddings: {store2.index.ntotal}")
    print(f"Reloaded metadata: {len(store2.metadata)}")
    
    assert store2.index.ntotal == 3, "Reload embedding count mismatch"
    assert len(store2.metadata) == 3, "Reload metadata count mismatch"
    print("Index reload works correctly.")
    
    # 4. Search
    print("\n--- Searching the Index ---")
    query = "How many sick days do I get?"
    print(f"Query: '{query}'")
    
    results = store2.search(query, top_k=2)
    print(f"Returned {len(results)} results.")
    
    for idx, res in enumerate(results):
        print(f"\nResult {idx+1}:")
        print(f"  Text: {res['text']}")
        print(f"  Doc: {res['document_name']}")
        print(f"  Score (L2 distance): {res['similarity_score']:.4f}")
        
    assert len(results) == 2, "Search should return top 2 results"
    assert "Sick leave" in results[0]['text'], "Top result should be about sick leave"
    print("\nSample search returns successfully ranked chunks.")
    print("\nAll Acceptance Criteria Met.")

if __name__ == "__main__":
    run_tests()
