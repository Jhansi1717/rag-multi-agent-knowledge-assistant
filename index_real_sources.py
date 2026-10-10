"""Index REAL_SOURCE_DOCUMENT corpus for M1.4 evaluation."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.pipeline import index_file
from vector_store.store import VectorStore


def index_real_sources():
    """Ingest all real source documents into a dedicated evaluation index."""

    real_sources_dir = Path("data/real_sources")
    eval_index_dir = Path("data/real_sources/evaluation_index")

    # Create evaluation index directory
    eval_index_dir.mkdir(parents=True, exist_ok=True)

    # Initialize vector store for evaluation
    vector_store = VectorStore(
        index_path=str(eval_index_dir / "index.faiss"),
        meta_path=str(eval_index_dir / "metadata.json")
    )

    # Load provenance manifest
    with open(real_sources_dir / "provenance_manifest.json", 'r') as f:
        manifest = json.load(f)

    print(f"Indexing {len(manifest['documents'])} real source documents...")

    for doc in manifest['documents']:
        file_path = real_sources_dir / doc['path']
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        print(f"Ingesting: {doc['title']}")

        try:
            # Index the file using the production pipeline
            result = index_file(
                str(file_path),
                vector_store=vector_store,
                domain=doc['domain'],
                source_location=str(file_path)
            )
            print(f"  - Chunks: {result['chunk_count']}")
            print(f"  - Added: {result['chunks_added']}")
        except Exception as e:
            print(f"  - Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save the final index
    vector_store.save()

    print(f"\nEvaluation index created at: {eval_index_dir}")
    print(f"Total chunks indexed: {len(vector_store.metadata)}")

    # Create index summary
    summary = {
        "index_created_at": manifest['generated_at'],
        "provenance_type": "REAL_SOURCE_DOCUMENT",
        "documents_indexed": len(manifest['documents']),
        "total_chunks": len(vector_store.metadata),
        "total_vectors": vector_store.index.ntotal,
        "domains": list(set(doc['domain'] for doc in manifest['documents'])),
        "index_path": str(eval_index_dir / "index.faiss"),
        "meta_path": str(eval_index_dir / "metadata.json")
    }

    with open(eval_index_dir / "index_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Index summary saved to: {eval_index_dir / 'index_summary.json'}")


if __name__ == "__main__":
    index_real_sources()
