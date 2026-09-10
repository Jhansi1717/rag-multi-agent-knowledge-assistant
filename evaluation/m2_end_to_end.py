"""Run the final M2 end-to-end evaluation without changing the corpus.

When OPENAI_API_KEY is unavailable, a deterministic context-echo client is used
so the orchestration and contract metrics can still be measured. Those outputs
are explicitly marked as mocked and are not used to claim LLM factual quality.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
QUERIES_PATH = ROOT / "data" / "evaluation" / "queries.json"
INDEX_PATH = ROOT / "data" / "evaluation" / "index.faiss"
META_PATH = ROOT / "data" / "evaluation" / "metadata.json"
JSON_OUTPUT_PATH = ROOT / "evaluation" / "m2_end_to_end_results.json"
MARKDOWN_OUTPUT_PATH = ROOT / "evaluation" / "m2_end_to_end_results.md"
sys.path.insert(0, str(ROOT))

from agents.models import RetrievalHit
from agents.orchestrator import Orchestrator
from agents.response_generation import ResponseGenerationAgent
from agents.query_understanding import QueryUnderstandingAgent
from vector_store.store import VectorStore


class ContextEchoClient:
    """Offline test double matching the OpenAI chat-completions call shape."""

    class _Completions:
        def create(self, *, messages: list[dict[str, str]], **_: Any) -> Any:
            context = messages[1]["content"].split("\n\nQuestion:", 1)[0]
            text = context.split("\n", 3)[-1].strip()

            class Message:
                content = f"According to the supplied context: {text}"

            class Choice:
                message = Message()

            class Result:
                choices = [Choice()]

            return Result()

    chat = type("Chat", (), {"completions": _Completions()})()


def load_queries() -> list[dict[str, Any]]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    queries = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(queries, list):
        raise ValueError("Evaluation corpus must contain a query list")
    # The source corpus has no ambiguous examples; these additions test routing
    # while leaving the source dataset untouched.
    return [
        *queries,
        {
            "query_id": "se_ambiguous_01",
            "domain": "Software Engineering",
            "category": "ambiguous",
            "query": "That one?",
            "expected_document": "NONE",
            "expected_keywords": [],
        },
        {
            "query_id": "ha_ambiguous_01",
            "domain": "Hospital Administration",
            "category": "ambiguous",
            "query": "What about it?",
            "expected_document": "NONE",
            "expected_keywords": [],
        },
    ]


def expected_class(category: str) -> str:
    return "factual" if category == "unavailable-information" else category


def hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "rank": hit.rank,
        "relevance_score": hit.relevance_score,
        "distance_score": hit.distance_score,
        "filename": hit.filename,
        "document_id": hit.document_id,
        "chunk_id": hit.chunk_id,
        "text": hit.text,
        "metadata": hit.metadata,
    }


def make_orchestrator(vector_store: VectorStore) -> tuple[Orchestrator, str]:
    if os.getenv("OPENAI_API_KEY"):
        return Orchestrator(vector_store), "openai"
    return (
        Orchestrator(
            vector_store,
            response_gen=ResponseGenerationAgent(llm_client=ContextEchoClient()),
        ),
        "mock_context_echo",
    )


def run() -> dict[str, Any]:
    vector_store = VectorStore(
        index_path=str(INDEX_PATH),
        meta_path=str(META_PATH),
    )
    vector_store.load()
    orchestrator, generation_mode = make_orchestrator(vector_store)
    understanding = QueryUnderstandingAgent()
    records: list[dict[str, Any]] = []

    for item in load_queries():
        query = str(item["query"])
        expected = str(item["category"])
        parsed = understanding.analyze(query)
        response = orchestrator.handle(
            query,
            session_id="m2-evaluation",
            top_k=3,
            request_id=f"m2-eval-{item['query_id']}-{uuid.uuid4()}",
        )
        retrieval = response.retrieval
        available_expected = expected != "unavailable-information"
        classification_expected = expected_class(expected)
        records.append(
            {
                "query_id": item["query_id"],
                "domain": item["domain"],
                "query": query,
                "expected_type": classification_expected,
                "expected_category": expected,
                "expected_document": item.get("expected_document"),
                "query_understanding": {
                    "predicted_type": parsed.query_type,
                    "classification_confidence": parsed.classification_confidence,
                    "routing": parsed.routing,
                },
                "retrieval": {
                    "top_k": retrieval.top_k if retrieval is not None else None,
                    "ranked_evidence": (
                        [hit_payload(hit) for hit in retrieval.results]
                        if retrieval is not None
                        else []
                    ),
                    "filtered_evidence": (
                        [hit_payload(hit) for hit in retrieval.results]
                        if retrieval is not None
                        else []
                    ),
                    "filtered_count": (
                        retrieval.filtered_count if retrieval is not None else None
                    ),
                    "retrieval_confidence": (
                        retrieval.retrieval_confidence if retrieval is not None else 0.0
                    ),
                    "sufficient_evidence": (
                        retrieval.sufficient_evidence if retrieval is not None else False
                    ),
                    "no_relevant_information": (
                        retrieval.no_relevant_information
                        if retrieval is not None
                        else True
                    ),
                    "expected_information_available": available_expected,
                },
                "response": {
                    "answer": response.answer,
                    "grounded": response.grounded,
                    "citations": [
                        {
                            "chunk_id": citation.chunk_id,
                            "filename": citation.filename,
                            "document_id": citation.document_id,
                        }
                        for citation in response.citations
                    ],
                    "confidence": response.confidence,
                    "confidence_level": response.confidence_level,
                    "no_information_found": response.no_information_found,
                },
                "orchestration": {
                    "completed_successfully": response.error is None,
                    "request_id": response.request_id,
                    "agent_sequence": (
                        ["QueryUnderstandingAgent", "RetrievalAgent", "ResponseGenerationAgent"]
                        if parsed.routing == "RETRIEVAL"
                        else ["QueryUnderstandingAgent"]
                    ),
                    "status": response.status,
                },
            }
        )

    total = len(records)
    classification_correct = sum(
        r["query_understanding"]["predicted_type"] == r["expected_type"]
        for r in records
    )
    available_records = [
        r for r in records if r["expected_category"] != "unavailable-information"
    ]
    ambiguous_records = [r for r in records if r["expected_category"] == "ambiguous"]
    citation_eligible = [
        r
        for r in records
        if r["expected_category"] not in {"ambiguous", "unavailable-information"}
    ]
    summary = {
        "total_queries": total,
        "classification_accuracy": classification_correct / total if total else 0.0,
        "retrieval_success": sum(
            r["retrieval"]["sufficient_evidence"] for r in available_records
        ) / len(available_records) if available_records else 0.0,
        "grounded_response_rate": sum(
            r["response"]["grounded"] for r in available_records
        ) / len(available_records) if available_records else 0.0,
        "citation_coverage": sum(
            bool(r["response"]["citations"]) for r in citation_eligible
        ) / len(citation_eligible) if citation_eligible else 0.0,
        "ambiguous_detection_rate": sum(
            r["query_understanding"]["predicted_type"] == "ambiguous"
            and r["query_understanding"]["routing"] == "CLARIFICATION"
            for r in ambiguous_records
        ) / len(ambiguous_records) if ambiguous_records else 0.0,
        "no_evidence_handling_rate": sum(
            r["response"]["no_information_found"]
            for r in records
            if r["expected_category"] == "unavailable-information"
        ) / max(
            1,
            sum(r["expected_category"] == "unavailable-information" for r in records),
        ),
        "end_to_end_success_rate": sum(
            r["orchestration"]["completed_successfully"] for r in records
        ) / total if total else 0.0,
    }
    output = {
        "methodology": {
            "source_corpus": str(QUERIES_PATH.relative_to(ROOT)),
            "corpus_queries": 19,
            "added_ambiguous_queries": 2,
            "domains": ["Software Engineering", "Hospital Administration"],
            "top_k": 3,
            "unavailable_labels_are_factual_for_classification": True,
            "generation_mode": generation_mode,
            "llm_quality_claim": "No automated factual-accuracy claim; manual review is required.",
            "manual_review_sample": [
                {
                    "query_id": r["query_id"],
                    "correctness": None,
                    "relevance": None,
                    "groundedness": None,
                    "readability": None,
                    "review_status": "pending_human_review",
                }
                for r in records[:4]
            ],
        },
        "summary": summary,
        "queries": records,
    }
    JSON_OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_OUTPUT_PATH.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return output


def render_markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    methodology = output["methodology"]
    lines = [
        "# M2 End-to-End Evaluation",
        "",
        "## Methodology",
        "",
        f"- Source corpus: `{methodology['source_corpus']}` ({methodology['corpus_queries']} queries).",
        "- Domains: Software Engineering and Hospital Administration.",
        "- Added two explicit ambiguous cases because the source corpus contains none.",
        "- `unavailable-information` is treated as factual for classification; evidence availability is measured separately.",
        f"- Generation mode: `{methodology['generation_mode']}`.",
        "- No automated factual-accuracy claim is made for LLM output.",
        "- Manual review fields are included for a small sample and remain pending human review.",
        "",
        "## Actual metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        formatted = f"{value:.1%}" if key != "total_queries" else str(value)
        lines.append(f"| {key.replace('_', ' ').title()} | {formatted} |")
    lines.extend(
        [
            "",
            "## Query-level results",
            "",
            "| ID | Domain | Expected | Predicted | Routing | Evidence | Grounded | Citations | Status |",
            "|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for record in output["queries"]:
        lines.append(
            f"| {record['query_id']} | {record['domain']} | {record['expected_category']} | "
            f"{record['query_understanding']['predicted_type']} | "
            f"{record['query_understanding']['routing']} | "
            f"{'yes' if record['retrieval']['sufficient_evidence'] else 'no'} | "
            f"{'yes' if record['response']['grounded'] else 'no'} | "
            f"{'yes' if record['response']['citations'] else 'no'} | "
            f"{record['orchestration']['status']} |"
        )
    lines.extend(["", "## Manual review sample", ""])
    for item in methodology["manual_review_sample"]:
        lines.append(f"- `{item['query_id']}`: pending human review.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
