"""Index M1 evaluation corpus through the ingestion pipeline."""

import glob
import json
import os
import sys

from ingestion.pipeline import index_file, ingest_file
from vector_store.store import VectorStore

DOMAIN_MAP = {
    "software_engineering": "Software Engineering",
    "hospital_administration": "Hospital Administration",
}


def discover_evaluation_files() -> list[tuple[str, str]]:
    pairs = []
    for folder, domain in DOMAIN_MAP.items():
        pattern = os.path.join("data", folder, "*.*")
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path) and not path.endswith(".gitkeep"):
                pairs.append((path, domain))
    return pairs


def main() -> int:
    files = discover_evaluation_files()
    if not files:
        print("FAIL: No evaluation files found. Run generate_evaluation_corpus.py first.")
        return 1

    index_path = "data/evaluation/index.faiss"
    meta_path = "data/evaluation/metadata.json"
    os.makedirs("data/evaluation", exist_ok=True)

    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(meta_path):
        os.remove(meta_path)

    store = VectorStore(index_path=index_path, meta_path=meta_path)

    report = {"files": [], "totals": {"files": 0, "chunks": 0, "vectors": 0}}
    failures = []

    print("=== M1 Evaluation Corpus Ingestion ===")
    for path, domain in files:
        print(f"\n--- {path} ({domain}) ---")
        try:
            doc, chunks = ingest_file(path, domain=domain)
            added = store.index_document(doc, chunks)
            print(f"  Chunks: {len(chunks)} | Indexed: {added}")
            report["files"].append({
                "path": path,
                "domain": domain,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "chunks": len(chunks),
                "indexed": added,
                "status": "PASS",
            })
            report["totals"]["files"] += 1
            report["totals"]["chunks"] += len(chunks)
        except Exception as exc:
            print(f"  FAIL: {exc}")
            failures.append({"path": path, "error": str(exc)})
            report["files"].append({
                "path": path,
                "domain": domain,
                "status": "FAIL",
                "error": str(exc),
            })

    store.save()
    report["totals"]["vectors"] = store.index.ntotal

    report_path = "data/evaluation/ingestion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Files ingested: {report['totals']['files']}/{len(files)}")
    print(f"Total chunks:   {report['totals']['chunks']}")
    print(f"Vectors indexed: {report['totals']['vectors']}")
    print(f"Embeddings OK:  {store.index.ntotal == report['totals']['chunks']}")
    print(f"Report saved:   {report_path}")

    if failures or store.index.ntotal != report["totals"]["chunks"]:
        print("RESULT: FAIL")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
