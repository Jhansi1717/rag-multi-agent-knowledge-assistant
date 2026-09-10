import os
import glob
from vector_store.store import VectorStore
from ingestion import extract_document, clean_document, TextChunker, validate_provenance

def main():
    print("--- Re-indexing Sample Knowledge Base (Phase 4) ---")
    vector_store = VectorStore()
    chunker = TextChunker()
    
    # Start clean
    if os.path.exists("data/index.faiss"): os.remove("data/index.faiss")
    if os.path.exists("data/metadata.json"): os.remove("data/metadata.json")
    
    files = []
    files.extend(glob.glob("data/hr/*.*"))
    files.extend(glob.glob("data/technical/*.*"))
    files = [f for f in files if os.path.isfile(f) and not f.endswith('.gitkeep')]
    
    print(f"Found {len(files)} files to index.")
    
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            # Phase 3 Pipeline
            doc = extract_document(file_path)
            cleaned_doc = clean_document(doc)
            chunks = chunker.chunk_document(cleaned_doc)
            validate_provenance(chunks, cleaned_doc)
            
            # Phase 4 Vector Store Integration
            vector_store.add_chunks(chunks)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    vector_store.save()
    print(f"Index saved. Total chunks: {vector_store.index.ntotal}")

if __name__ == '__main__':
    main()
