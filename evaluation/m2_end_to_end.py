"""Run the M2 end-to-end evaluation with only the external LLM mocked."""

from __future__ import annotations

import json
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


class MockOpenAIClient:
    """Mock only the OpenAI chat-completions boundary used by the real agent."""

    class _Completions:
        def create(self, *, messages: list[dict[str, str]], **_: Any) -> Any:
            question = messages[1]["content"].rsplit("\n\nQuestion:", 1)[-1].strip()
            context = messages[1]["content"].split("\n\nQuestion:", 1)[0]

            class Message:
                content = (
                    f"Based only on the retrieved context, the answer to "
                    f"{question!r} is supported by these passages:\n{context}"
                )

            class Choice:
                message = Message()

            class Result:
                choices = [Choice()]

            return Result()

    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": self._Completions()})()


def load_queries() -> list[dict[str, Any]]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    source = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(source, list):
        raise ValueError("Evaluation corpus must contain a query list")

    selected: list[dict[str, Any]] = []
    categories = {
        "factual": lambda item: (
            item.get("category") == "factual"
            and item.get("expected_document") != "NONE"
        ),
        "procedural": lambda item: item.get("category") == "procedural",
        "comparative": lambda item: item.get("category") == "comparative",
        "ambiguous": lambda item: item.get("category") == "ambiguous",
        "unavailable-information": lambda item: (
            item.get("category") == "factual"
            and item.get("expected_document") == "NONE"
        ),
    }
    for category, predicate in categories.items():
        matches = [item for item in source if predicate(item)]
        chosen = [
            item
            for domain in ("Software Engineering", "Hospital Administration")
            for item in matches
            if item.get("domain") == domain
        ]
        chosen = chosen[:2] + [
            item
            for item in matches
            if item.get("domain") == "Hospital Administration"
        ][:2]
        if len(chosen) != 4:
            raise ValueError(
                f"Evaluation corpus needs four {category} queries across both domains"
            )
        for item in chosen:
            normalized = dict(item)
            normalized["category"] = category
            selected.append(normalized)

    domains = {"Software Engineering", "Hospital Administration"}
    if {item.get("domain") for item in selected} != domains:
        raise ValueError("Selected evaluation queries must cover both domains")
    return selected


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


def make_orchestrator(vector_store: VectorStore) -> Orchestrator:
    return Orchestrator(
        vector_store,
        response_gen=ResponseGenerationAgent(llm_client=MockOpenAIClient()),
    )


def run() -> dict[str, Any]:
    vector_store = VectorStore(
        index_path=str(INDEX_PATH),
        meta_path=str(META_PATH),
    )
    vector_store.load()
    orchestrator = make_orchestrator(vector_store)
    understanding = QueryUnderstandingAgent()
    records: list[dict[str, Any]] = []

    for item in load_queries():
        query = str(item["query"])
        expected_category = str(item["category"])
        parsed = understanding.analyze(query)
        request_id = f"m2-eval-{item['query_id']}-{uuid.uuid4()}"
        response = orchestrator.handle(
            query,
            session_id="m2-evaluation",
            top_k=3,
            request_id=request_id,
        )
        retrieval = response.retrieval
        expected_available = expected_category not in {
            "ambiguous",
            "unavailable-information",
        }
        expected_document = item.get("expected_document")
        expected_evidence = bool(
            retrieval
            and expected_document
            and expected_document != "NONE"
            and any(
                hit.filename == expected_document
                for hit in retrieval.results
            )
        )
        records.append(
            {
                "query_id": item["query_id"],
                "domain": item["domain"],
                "query": query,
                "expected_category": expected_category,
                "expected_type": expected_class(expected_category),
                "expected_document": expected_document,
                "query_understanding": {
                    "predicted_type": parsed.query_type,
                    "classification_confidence": parsed.classification_confidence,
                    "routing": parsed.routing,
                },
                "retrieval": {
                    "top_k": retrieval.top_k if retrieval else None,
                    "ranked_evidence": (
                        [hit_payload(hit) for hit in retrieval.results]
                        if retrieval
                        else []
                    ),
                    "retrieval_confidence": (
                        retrieval.retrieval_confidence if retrieval else 0.0
                    ),
                    "sufficient_evidence": (
                        retrieval.sufficient_evidence if retrieval else False
                    ),
                    "expected_evidence_found": expected_evidence,
                },
                "response": {
                    "answer": response.answer,
                    "grounded": response.grounded,
                    "citations": [
                        {
                            "chunk_id": citation.chunk_id,
                            "filename": citation.filename,
                            "document_id": citation.document_id,
                            "excerpt": citation.excerpt,
                        }
                        for citation in response.citations
                    ],
                    "confidence": response.confidence,
                    "confidence_level": response.confidence_level,
                    "no_information_found": response.no_information_found,
                },
                "final_status": response.status,
                "request_id": response.request_id,
            }
        )

    total = len(records)
    classified = [
        record["query_understanding"]["predicted_type"]
        == record["expected_type"]
        for record in records
    ]
    available = [r for r in records if r["expected_category"] in {
        "factual", "procedural", "comparative"
    } and r["expected_document"] != "NONE"]
    ambiguous = [r for r in records if r["expected_category"] == "ambiguous"]
    unavailable = [
        r for r in records if r["expected_category"] == "unavailable-information"
    ]
    grounded = [r for r in available if r["retrieval"]["expected_evidence_found"]]
    cited = [r for r in grounded if r["response"]["citations"]]
    no_evidence = [
        r for r in unavailable if r["response"]["no_information_found"]
    ]
    successful = [
        r for r in records
        if (
            (r["expected_category"] == "ambiguous"
             and r["final_status"] == "clarification_needed")
            or (
                r["expected_category"] == "unavailable-information"
                and r["response"]["no_information_found"]
            )
            or (
                r in available
                and r["retrieval"]["expected_evidence_found"]
                and r["response"]["grounded"]
                and bool(r["response"]["citations"])
                and r["final_status"] == "answered"
            )
        )
    ]
    summary = {
        "total_queries": total,
        "classification_accuracy": sum(classified) / total if total else 0.0,
        "retrieval_evidence_success": (
            sum(r["retrieval"]["expected_evidence_found"] for r in available)
            / len(available) if available else 0.0
        ),
        "grounded_response_rate": (
            sum(r["response"]["grounded"] for r in grounded) / len(grounded)
            if grounded else 0.0
        ),
        "citation_coverage": len(cited) / len(grounded) if grounded else 0.0,
        "no_evidence_handling_rate": (
            len(no_evidence) / len(unavailable) if unavailable else 0.0
        ),
        "end_to_end_success_rate": len(successful) / total if total else 0.0,
        "ambiguous_detection_rate": (
            sum(
                r["query_understanding"]["predicted_type"] == "ambiguous"
                and r["query_understanding"]["routing"] == "CLARIFICATION"
                for r in ambiguous
            ) / len(ambiguous) if ambiguous else 0.0
        ),
    }
    output = {
        "methodology": {
            "source_corpus": str(QUERIES_PATH.relative_to(ROOT)),
            "query_count": total,
            "queries_per_category": 4,
            "domains": sorted({r["domain"] for r in records}),
            "top_k": 3,
            "generation_path": (
                "Real QueryUnderstandingAgent -> RetrievalAgent -> "
                "ResponseGenerationAgent -> Orchestrator; only the external "
                "OpenAI chat-completions API is mocked."
            ),
            "llm_quality_claim": (
                "No automated factual-accuracy claim is made for mocked LLM output."
            ),
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
        f"- Source corpus: `{methodology['source_corpus']}`.",
        f"- Query count: {methodology['query_count']} (four per category).",
        "- Domains: Software Engineering and Hospital Administration.",
        f"- Generation path: {methodology['generation_path']}",
        f"- {methodology['llm_quality_claim']}",
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
            "| ID | Domain | Expected | Predicted | Confidence | Routing | Evidence | Grounded | Citations | No information | Status |",
            "|---|---|---|---|---:|---|---|---|---|---|---|",
        ]
    )
    for record in output["queries"]:
        understanding = record["query_understanding"]
        retrieval = record["retrieval"]
        response = record["response"]
        lines.append(
            f"| {record['query_id']} | {record['domain']} | "
            f"{record['expected_category']} | {understanding['predicted_type']} | "
            f"{understanding['classification_confidence']:.3f} | "
            f"{understanding['routing']} | "
            f"{'yes' if retrieval['expected_evidence_found'] else 'no'} | "
            f"{'yes' if response['grounded'] else 'no'} | "
            f"{'yes' if response['citations'] else 'no'} | "
            f"{'yes' if response['no_information_found'] else 'no'} | "
            f"{record['final_status']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run()
