"""Evaluate M2.1 query classification against the evaluation query corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
QUERIES_PATH = ROOT / "data" / "evaluation" / "queries.json"
JSON_OUTPUT_PATH = ROOT / "evaluation" / "m2_query_classification_results.json"
MARKDOWN_OUTPUT_PATH = ROOT / "evaluation" / "m2_query_classification_results.md"
QUERY_TYPES = ("factual", "procedural", "comparative", "ambiguous")

sys.path.insert(0, str(ROOT))

from agents.query_understanding import QueryUnderstandingAgent


def load_queries() -> list[dict[str, Any]]:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        queries = payload.get("queries", [])
    else:
        queries = payload
    if not isinstance(queries, list):
        raise ValueError("Evaluation queries must be a list or an object containing 'queries'.")
    return queries


def expected_type(query: dict[str, Any]) -> str:
    label = str(query.get("query_type") or query.get("category") or "").lower()
    if label not in QUERY_TYPES:
        raise ValueError(f"Unsupported expected query type: {label!r}")
    return label


def render_markdown(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# M2.1 Query Classification Results",
        "",
        "The official M2.1 corpus uses only factual, procedural, comparative, "
        "and ambiguous classification labels. Evidence availability is determined "
        "by retrieval, not classification.",
        "",
        "## Summary",
        "",
        f"- Total queries: {summary['total_queries']}",
        f"- Correct: {summary['correct']}",
        f"- Incorrect: {summary['incorrect']}",
        f"- Accuracy: {summary['accuracy']:.1%}",
        "",
        "## Per-class accuracy",
        "",
        "| Class | Total | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for query_type in QUERY_TYPES:
        metrics = summary["per_class_accuracy"][query_type]
        accuracy = "n/a" if metrics["total"] == 0 else f"{metrics['accuracy']:.1%}"
        lines.append(
            f"| {query_type} | {metrics['total']} | {metrics['correct']} | {accuracy} |"
        )

    lines.extend(
        [
            "",
            "## Query results",
            "",
            "| Query | Expected | Predicted | Confidence | Routing | Correct |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for result in results:
        query = result["query"].replace("|", "\\|")
        lines.append(
            f"| {query} | {result['expected_type']} | {result['predicted_type']} | "
            f"{result['confidence']:.2f} | {result['routing']} | "
            f"{'yes' if result['correct'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    agent = QueryUnderstandingAgent()
    results: list[dict[str, Any]] = []

    for item in load_queries():
        query = str(item["query"])
        parsed = agent.analyze(query)
        expected = expected_type(item)
        results.append(
            {
                "query": query,
                "expected_type": expected,
                "predicted_type": parsed.query_type,
                "confidence": parsed.classification_confidence,
                "routing": parsed.routing,
                "correct": parsed.query_type == expected,
            }
        )

    correct = sum(result["correct"] for result in results)
    per_class: dict[str, dict[str, Any]] = {}
    for query_type in QUERY_TYPES:
        class_results = [r for r in results if r["expected_type"] == query_type]
        class_correct = sum(r["correct"] for r in class_results)
        per_class[query_type] = {
            "total": len(class_results),
            "correct": class_correct,
            "accuracy": (
                class_correct / len(class_results) if class_results else None
            ),
        }

    summary = {
        "total_queries": len(results),
        "correct": correct,
        "incorrect": len(results) - correct,
        "accuracy": correct / len(results) if results else 0.0,
        "per_class_accuracy": per_class,
    }
    output = {"summary": summary, "queries": results}
    JSON_OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT_PATH.write_text(
        render_markdown(summary, results),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Results written to: {JSON_OUTPUT_PATH}")
    print(f"Report written to: {MARKDOWN_OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    run()
