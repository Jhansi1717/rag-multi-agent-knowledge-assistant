"""M1.4 Retrieval Evaluation using REAL_SOURCE_DOCUMENT corpus."""

import json
from pathlib import Path
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore
from agents.retrieval_agent import RetrievalAgent


def evaluate_m14_real_sources():
    """Run M1.4 retrieval evaluation against real-source corpus."""

    # Load evaluation index
    eval_index_dir = Path("data/real_sources/evaluation_index")
    vector_store = VectorStore(
        index_path=str(eval_index_dir / "index.faiss"),
        meta_path=str(eval_index_dir / "metadata.json")
    )
    vector_store.load()

    # Load provenance manifest
    with open("data/real_sources/provenance_manifest.json", 'r') as f:
        manifest = json.load(f)

    # Load queries
    with open("data/real_sources/m14_real_queries.json", 'r') as f:
        queries = json.load(f)['queries']

    # Initialize retriever
    retriever = SemanticRetriever(vector_store)
    agent = RetrievalAgent(retriever, top_k=5)

    print("=" * 80)
    print("M1.4 RETRIEVAL EVALUATION - REAL_SOURCE_DOCUMENT CORPUS")
    print("=" * 80)
    print(f"\nProvenance Type: {manifest['provenance_type']}")
    print(f"Documents Indexed: {len(manifest['documents'])}")
    print(f"Total Chunks: {len(vector_store.metadata)}")
    print(f"Total Vectors: {vector_store.index.ntotal}")
    print(f"Domains: {', '.join(set(doc['domain'] for doc in manifest['documents']))}")
    print()

    # Track results
    results = {
        'factual': {'total': 0, 'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0},
        'procedural': {'total': 0, 'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0},
        'comparative': {'total': 0, 'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0},
        'unavailable': {'total': 0, 'false_positives': 0},
        'low-relevance': {'total': 0, 'rejected': 0},
        'overall': {'total': 0, 'hit_at_1': 0, 'hit_at_3': 0, 'hit_at_5': 0}
    }

    detailed_results = []

    for query_item in queries:
        query_id = query_item['id']
        query = query_item['query']
        category = query_item['category']
        expected_doc = query_item.get('expected_document', 'NONE')
        expected_domain = query_item.get('expected_domain')

        print(f"\n[{query_id}] {query}")
        print(f"Category: {category}")
        print(f"Expected: {expected_doc}")

        # Retrieve
        raw_hits = retriever.retrieve(query, top_k=5)
        d = raw_hits[0].get('distance_score', float('inf')) if raw_hits else float('inf')
        top_score = 1.0 / (1.0 + d)

        q_type = category if category in ('factual', 'procedural', 'comparative') else 'factual'
        agent_result = agent.retrieve(query, query_type=q_type)

        if category == 'unavailable':
            results['unavailable']['total'] += 1
            results['overall']['total'] += 1
            is_rejected = agent_result.no_relevant_information or not agent_result.results
            if not is_rejected:
                results['unavailable']['false_positives'] += 1
                print(f"Result: FALSE POSITIVE (score: {top_score:.3f})")
            else:
                print(f"Result: CORRECTLY REJECTED (score: {top_score:.3f})")
            detailed_results.append({
                'id': query_id,
                'query': query,
                'category': category,
                'expected': expected_doc,
                'result': 'false_positive' if not is_rejected else 'correctly_rejected',
                'top_score': top_score
            })
            continue

        if category == 'low-relevance':
            results['low-relevance']['total'] += 1
            results['overall']['total'] += 1
            is_rejected = agent_result.no_relevant_information or not agent_result.results
            if is_rejected:
                results['low-relevance']['rejected'] += 1
                print(f"Result: CORRECTLY REJECTED (score: {top_score:.3f})")
            else:
                print(f"Result: FALSE POSITIVE (score: {top_score:.3f})")
            detailed_results.append({
                'id': query_id,
                'query': query,
                'category': category,
                'expected': expected_doc,
                'result': 'correctly_rejected' if is_rejected else 'false_positive',
                'top_score': top_score
            })
            continue

        # Regular retrieval queries
        results[category]['total'] += 1
        results['overall']['total'] += 1

        # Check Hit@1, Hit@3, Hit@5 using raw FAISS candidates
        filenames = [hit.get('filename') or hit.get('document_name') or '' for hit in raw_hits]
        top_1 = expected_doc in filenames[:1]
        top_3 = expected_doc in filenames[:3]
        top_5 = expected_doc in filenames[:5]

        if top_1:
            results[category]['hit_at_1'] += 1
            results['overall']['hit_at_1'] += 1
        if top_3:
            results[category]['hit_at_3'] += 1
            results['overall']['hit_at_3'] += 1
        if top_5:
            results[category]['hit_at_5'] += 1
            results['overall']['hit_at_5'] += 1

        if not agent_result.results:
            print("Result: NO HITS")
            print(f"Hit@1: {'PASS' if top_1 else 'FAIL'}")
            print(f"Hit@3: {'PASS' if top_3 else 'FAIL'}")
            print(f"Hit@5: {'PASS' if top_5 else 'FAIL'}")
            detailed_results.append({
                'id': query_id,
                'query': query,
                'category': category,
                'expected': expected_doc,
                'result': 'no_hits',
                'top_filename': None,
                'top_score': top_score,
                'hit_at_1': top_1,
                'hit_at_3': top_3,
                'hit_at_5': top_5
            })
            continue

        print(f"Top-1: {filenames[0] if filenames else 'NONE'} (score: {top_score:.3f})")
        print(f"Hit@1: {'PASS' if top_1 else 'FAIL'}")
        print(f"Hit@3: {'PASS' if top_3 else 'FAIL'}")
        print(f"Hit@5: {'PASS' if top_5 else 'FAIL'}")

        detailed_results.append({
            'id': query_id,
            'query': query,
            'category': category,
            'expected': expected_doc,
            'result': 'hit' if top_1 else 'miss',
            'top_filename': filenames[0] if filenames else None,
            'top_score': top_score,
            'hit_at_1': top_1,
            'hit_at_3': top_3,
            'hit_at_5': top_5
        })

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for category in ['factual', 'procedural', 'comparative']:
        if results[category]['total'] > 0:
            print(f"\n{category.upper()}:")
            print(f"  Total: {results[category]['total']}")
            print(f"  Hit@1: {results[category]['hit_at_1']}/{results[category]['total']} ({100*results[category]['hit_at_1']/results[category]['total']:.1f}%)")
            print(f"  Hit@3: {results[category]['hit_at_3']}/{results[category]['total']} ({100*results[category]['hit_at_3']/results[category]['total']:.1f}%)")
            print(f"  Hit@5: {results[category]['hit_at_5']}/{results[category]['total']} ({100*results[category]['hit_at_5']/results[category]['total']:.1f}%)")

    print(f"\nUNAVAILABLE:")
    print(f"  Total: {results['unavailable']['total']}")
    print(f"  False Positives: {results['unavailable']['false_positives']}")

    print(f"\nLOW-RELEVANCE:")
    print(f"  Total: {results['low-relevance']['total']}")
    print(f"  Rejected: {results['low-relevance']['rejected']}")

    print(f"\nOVERALL (in-KB queries only):")
    in_kb_total = results['overall']['total'] - results['unavailable']['total'] - results['low-relevance']['total']
    if in_kb_total > 0:
        print(f"  Total: {in_kb_total}")
        print(f"  Hit@1: {results['overall']['hit_at_1']}/{in_kb_total} ({100*results['overall']['hit_at_1']/in_kb_total:.1f}%)")
        print(f"  Hit@3: {results['overall']['hit_at_3']}/{in_kb_total} ({100*results['overall']['hit_at_3']/in_kb_total:.1f}%)")
        print(f"  Hit@5: {results['overall']['hit_at_5']}/{in_kb_total} ({100*results['overall']['hit_at_5']/in_kb_total:.1f}%)")

    # Save results
    evaluation_report = {
        'provenance_type': manifest['provenance_type'],
        'corpus_info': {
            'documents': len(manifest['documents']),
            'chunks': len(vector_store.metadata),
            'vectors': vector_store.index.ntotal,
            'domains': list(set(doc['domain'] for doc in manifest['documents']))
        },
        'results_by_category': results,
        'detailed_results': detailed_results
    }

    with open("data/real_sources/m14_evaluation_results.json", 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"\nDetailed results saved to: data/real_sources/m14_evaluation_results.json")


if __name__ == "__main__":
    evaluate_m14_real_sources()
