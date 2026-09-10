import os
import json
from retrieval.retriever import SemanticRetriever
from vector_store.store import VectorStore

def main():
    print("--- Running Retrieval Evaluation Harness ---")
    vector_store = VectorStore()
    vector_store.load()
    retriever = SemanticRetriever(vector_store)

    queries = [
        # Domain A: HR
        {
            "query_id": "q1",
            "domain": "HR",
            "query_type": "Factual",
            "query_text": "How many annual leave days are provided?",
            "expected_source": "leave_policy.pdf"
        },
        {
            "query_id": "q2",
            "domain": "HR",
            "query_type": "Procedural",
            "query_text": "How do employees apply for leave?",
            "expected_source": "leave_policy.pdf"
        },
        {
            "query_id": "q3",
            "domain": "HR",
            "query_type": "Comparative",
            "query_text": "What is the difference between Annual and Sick leave carry-forward limit?",
            "expected_source": "leave_allowance.csv"
        },
        {
            "query_id": "q4",
            "domain": "HR",
            "query_type": "Unavailable-information",
            "query_text": "What was the company's revenue in 2025?",
            "expected_source": "NONE"
        },
        # Domain B: Tech
        {
            "query_id": "q5",
            "domain": "Software",
            "query_type": "Factual",
            "query_text": "What HTTP method is used to remove a resource?",
            "expected_source": "api_documentation.docx"
        },
        {
            "query_id": "q6",
            "domain": "Software",
            "query_type": "Procedural",
            "query_text": "What are the steps to deploy a backend service?",
            "expected_source": "deployment_guide.txt"
        },
        {
            "query_id": "q7",
            "domain": "Software",
            "query_type": "Comparative",
            "query_text": "How does REST differ from SOAP?",
            "expected_source": "api_documentation.docx"
        },
        {
            "query_id": "q8",
            "domain": "Software",
            "query_type": "Unavailable-information",
            "query_text": "Who is the CEO of the software division?",
            "expected_source": "NONE"
        }
    ]

    results_log = []
    
    total_queries = 0
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    
    failures = []

    for q in queries:
        top_5 = retriever.retrieve(q["query_text"], top_k=5)
        
        actual_top1 = top_5[0]['document_name'] if len(top_5) > 0 else None
        actual_top3 = [res['document_name'] for res in top_5[:3]]
        actual_top5 = [res['document_name'] for res in top_5]
        
        expected = q["expected_source"]
        
        # Calculate Rank and Hits
        relevant_rank = None
        if expected != "NONE":
            for rank, res in enumerate(top_5):
                if res['document_name'] == expected:
                    relevant_rank = rank + 1
                    break
                    
        h1 = 1 if relevant_rank == 1 else 0
        h3 = 1 if relevant_rank and relevant_rank <= 3 else 0
        h5 = 1 if relevant_rank and relevant_rank <= 5 else 0
        
        # For unavailable queries, we don't count towards Hit@k calculation because there's no correct doc to hit
        if expected != "NONE":
            total_queries += 1
            hit_1 += h1
            hit_3 += h3
            hit_5 += h5
            
            if h1 == 0:
                failures.append({
                    "query": q["query_text"],
                    "expected": expected,
                    "actual_top1": actual_top1,
                    "type": q["query_type"]
                })

        record = {
            "query_id": q["query_id"],
            "domain": q["domain"],
            "query_type": q["query_type"],
            "query_text": q["query_text"],
            "expected_source": expected,
            "actual_top1_source": actual_top1,
            "actual_top3_sources": actual_top3,
            "actual_top5_sources": actual_top5,
            "relevant_rank": relevant_rank,
            "score": top_5[0]['similarity_score'] if len(top_5) > 0 else None,
            "Hit@1": h1,
            "Hit@3": h3,
            "Hit@5": h5
        }
        results_log.append(record)

    # Save JSON
    with open("data/evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump(results_log, f, indent=2)

    # Generate Markdown
    md_content = "# Retrieval Validation Report\n\n"
    md_content += "## Aggregate Metrics\n"
    md_content += f"- **Total Eval Queries (Excl. Unavailable)**: {total_queries}\n"
    md_content += f"- **Hit@1**: {hit_1 / total_queries:.2%}\n"
    md_content += f"- **Hit@3**: {hit_3 / total_queries:.2%}\n"
    md_content += f"- **Hit@5**: {hit_5 / total_queries:.2%}\n\n"
    
    md_content += "## Detailed Results\n"
    md_content += "| ID | Domain | Type | Query | Expected | Top-1 | Rank | Hit@1 | Hit@3 | Hit@5 |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|---|\n"
    
    for r in results_log:
        rank_str = str(r['relevant_rank']) if r['relevant_rank'] else "N/A"
        md_content += f"| {r['query_id']} | {r['domain']} | {r['query_type']} | {r['query_text']} | {r['expected_source']} | {r['actual_top1_source']} | {rank_str} | {r['Hit@1']} | {r['Hit@3']} | {r['Hit@5']} |\n"
        
    md_content += "\n## Analysis\n"
    md_content += "### Failure Cases & Low-Relevance Results\n"
    md_content += "Reviewing the results above allows us to spot any discrepancies between Expected and Top-1.\n"
    
    if len(failures) > 0:
        for fail in failures:
            md_content += f"- **Query**: {fail['query']} (Type: {fail['type']})\n  - Expected: {fail['expected']}\n  - Actual Top-1: {fail['actual_top1']}\n"
    else:
        md_content += "- No Hit@1 failures observed for relevant queries.\n"
        
    md_content += "\n### Limitations and Likely Causes\n"
    md_content += "- **Unavailable Information**: These queries returned documents because FAISS always returns the nearest neighbor, even if the absolute distance is high. This highlights a limitation: the system needs a similarity threshold or an LLM to decide if the retrieved context actually contains the answer.\n"
    md_content += "- **Comparative Queries**: Semantic embeddings sometimes struggle with relational comparisons if the exact wording differs heavily from the chunk.\n"
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/validation.md", "w", encoding='utf-8') as f:
        f.write(md_content)

    print("Evaluation completed. Reports generated at data/evaluation_results.json and docs/validation.md")

if __name__ == '__main__':
    main()
