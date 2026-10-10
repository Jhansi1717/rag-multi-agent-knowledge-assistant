"""Provenance-gated M1.4 retrieval acceptance evaluation.

This module never creates a corpus or uses a mock retriever. It refuses to
score production acceptance unless expected sources are verified originals and
are present in the evaluated FAISS index. The default report is still emitted
when that gate is blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "evaluation" / "m14_provenance_manifest.json"
DEFAULT_OUTPUT_JSON = ROOT / "evaluation" / "m14_provenance_results.json"
DEFAULT_OUTPUT_MD = ROOT / "evaluation" / "m14_provenance_results.md"
INDEX_PATH = ROOT / "data" / "evaluation" / "index.faiss"
META_PATH = ROOT / "data" / "evaluation" / "metadata.json"
ALLOWED_PROVENANCE = {
    "REAL_SOURCE_DOCUMENT",
    "GENERATED_SAMPLE",
    "TEST_FIXTURE",
    "MOCK_RETRIEVER",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else root / path


def _verified_real_source(record: dict[str, Any], root: Path) -> tuple[bool, Optional[str], str]:
    if record.get("provenance_class") != "REAL_SOURCE_DOCUMENT":
        return False, None, "Source is not classified as REAL_SOURCE_DOCUMENT."
    if record.get("provenance_status") != "VERIFIED":
        return False, None, "Source provenance status is not VERIFIED."
    if record.get("approved_for_production") is not True:
        return False, None, "Source is not explicitly approved for production acceptance."
    if not record.get("origin_uri"):
        return False, None, "Authoritative origin_uri is missing."
    expected_hash = str(record.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        return False, None, "A verified SHA-256 source fingerprint is missing."
    path = _source_path(root, str(record.get("path", "")))
    if not path.is_file():
        return False, None, f"Source file is missing: {record.get('path', '')}."
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        return False, actual_hash, "Source SHA-256 does not match the provenance record."
    return True, actual_hash, "Verified original source document."


def _source_registry(manifest: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    registry = {}
    for record in manifest.get("sources", []):
        path = _source_path(root, str(record.get("path", "")))
        filename = path.name
        if filename in registry:
            raise ValueError(f"Source manifest has duplicate basename: {filename}")
        item = dict(record)
        item["sha256_actual"] = _sha256(path) if path.is_file() else None
        item["exists"] = path.is_file()
        item["counts_toward_acceptance"], actual_hash, reason = _verified_real_source(item, root)
        item["verified_sha256"] = actual_hash
        item["acceptance_reason"] = reason
        if item.get("provenance_class") not in ALLOWED_PROVENANCE:
            item["provenance_class"] = "UNVERIFIED_SOURCE"
            item["counts_toward_acceptance"] = False
            item["acceptance_reason"] = "Unrecognized provenance class."
        registry[filename] = item
    return registry


def _query_source(spec: dict[str, Any], registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = str(spec.get("expected_document", "NONE"))
    if expected.upper() == "NONE":
        return {
            "provenance_class": "NO_EXPECTED_SOURCE",
            "path": None,
            "sha256": None,
            "counts_toward_acceptance": False,
            "reason": "The query is expected to have no supporting source in the declared corpus scope.",
        }
    item = registry.get(Path(expected).name)
    if item is None:
        return {
            "provenance_class": "UNVERIFIED_SOURCE",
            "path": None,
            "sha256": None,
            "counts_toward_acceptance": False,
            "reason": f"Expected source {expected!r} is absent from the provenance registry.",
        }
    return {
        "provenance_class": item["provenance_class"],
        "path": item.get("path"),
        "sha256": item.get("sha256_actual"),
        "origin": item.get("origin_uri") or item.get("origin"),
        "counts_toward_acceptance": item["counts_toward_acceptance"],
        "reason": item["acceptance_reason"],
    }


def _combined_queries(payload: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    queries = [dict(item) for item in payload.get("queries", [])]
    overrides = manifest.get("comparative_targets", {})
    for item in queries:
        item["query_provenance_class"] = manifest["query_dataset"].get(
            "provenance_class", "UNVERIFIED_SOURCE"
        )
        item["query_provenance_status"] = manifest["query_dataset"].get(
            "provenance_status", "UNVERIFIED"
        )
        item["evaluation_kind"] = item.get("category", "")
        if (
            str(item.get("expected_document", "NONE")).upper() == "NONE"
            and item.get("category") != "ambiguous"
        ):
            item["evaluation_kind"] = "unavailable"
        if item.get("query_id") in overrides:
            item["comparison_targets"] = overrides[item["query_id"]]
    for item in manifest.get("additional_test_only_queries", []):
        additional = dict(item)
        additional["query_provenance_class"] = additional.get(
            "provenance_class", "UNVERIFIED_SOURCE"
        )
        additional["query_provenance_status"] = additional.get(
            "provenance_status", "UNVERIFIED"
        )
        additional["evaluation_kind"] = additional.get("category", "")
        queries.append(additional)
    return queries


def _coverage_is_complete(domain: Optional[str], manifest: dict[str, Any], registry: dict[str, dict[str, Any]]) -> bool:
    if not domain or domain not in manifest.get("complete_coverage_domains", []):
        return False
    domain_sources = [
        item for item in registry.values()
        if item.get("domain") == domain
    ]
    return bool(domain_sources) and all(
        item.get("counts_toward_acceptance")
        and item.get("provenance_class") == "REAL_SOURCE_DOCUMENT"
        for item in domain_sources
    )


def _eligible(spec: dict[str, Any], expected_source: dict[str, Any], manifest: dict[str, Any], registry: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    category = str(spec.get("evaluation_kind", spec.get("category", ""))).lower()
    if category == "ambiguous":
        return False, "Ambiguous query is not a retrieval acceptance case."
    if spec.get("query_provenance_class") == "MOCK_RETRIEVER":
        return False, "Mock-retriever cases are test-only."
    if spec.get("query_provenance_class") not in {"TEST_FIXTURE", "REAL_SOURCE_DOCUMENT"}:
        return False, "Query provenance is unverified."
    expected = str(spec.get("expected_document", "NONE"))
    if expected.upper() != "NONE":
        if expected_source["provenance_class"] != "REAL_SOURCE_DOCUMENT":
            return False, "Expected answer source is not a verified real source document."
        if not expected_source["counts_toward_acceptance"]:
            return False, expected_source["reason"]
        if category == "comparative":
            targets = spec.get("comparison_targets", [])
            if len(targets) < 2:
                return False, "Comparative query is missing explicit expectations for both targets."
            if any(
                not _query_source(
                    {"expected_document": target.get("expected_document", "NONE")},
                    registry,
                )["counts_toward_acceptance"]
                for target in targets
            ):
                return False, "At least one comparison target lacks a verified real source."
        return True, "Expected answer source is a verified real source document."
    if category in {"unavailable", "unavailable-information", "low-relevance"}:
        if not _coverage_is_complete(spec.get("domain"), manifest, registry):
            return False, "Cannot prove absence or low relevance without a verified complete corpus scope for this domain."
        return True, "Query is testable against a verified complete corpus scope."
    return False, "Query category has no declared retrieval acceptance policy."


def _location_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "page", "page_number", "page_start", "page_end", "section",
        "row_number", "row_start", "row_end",
    )
    return {key: metadata[key] for key in keys if metadata.get(key) is not None}


def _serialize_hit(hit: Any, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = dict(getattr(hit, "metadata", {}) or {})
    filename = str(getattr(hit, "filename", "") or "")
    source = _query_source({"expected_document": filename}, registry)
    return {
        "rank": int(hit.rank),
        "document": filename,
        "document_id": str(getattr(hit, "document_id", "") or ""),
        "chunk_id": str(getattr(hit, "chunk_id", "") or ""),
        "domain": metadata.get("domain", ""),
        "source_location": metadata.get("source_location", ""),
        "provenance_class": source["provenance_class"],
        "counts_toward_acceptance": source["counts_toward_acceptance"],
        "distance_score": float(hit.distance_score),
        "relevance_score": float(hit.relevance_score),
        "location_metadata": _location_metadata(metadata),
        "metadata": metadata,
        "text_preview": str(getattr(hit, "text", "") or "")[:240].replace("\n", " "),
        "_full_text": str(getattr(hit, "text", "") or ""),
    }


def _contains_term(text: str, term: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def _evaluate_live(store: Any, specs: list[dict[str, Any]], registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from agents.query_understanding import QueryUnderstandingAgent
    from agents.retrieval_agent import RetrievalAgent
    from retrieval.retriever import SemanticRetriever

    understanding = QueryUnderstandingAgent()
    agent = RetrievalAgent(SemanticRetriever(store), top_k=5)
    rows = []
    for spec in specs:
        if not spec["acceptance_eligible"]:
            continue
        parsed = understanding.analyze(spec["query"])
        kind = parsed.query_type
        if spec["evaluation_kind"] in {"unavailable", "low-relevance"}:
            kind = "factual"
        result = agent.retrieve(
            parsed.normalized_query,
            query_type=kind,
            domain=spec.get("domain"),
            top_k=5,
        )
        hits = [_serialize_hit(hit, registry) for hit in result.results[:5]]
        targets = []
        for target in spec.get("comparison_targets", []):
            target_doc = Path(target["expected_document"]).name
            support = [
                hit for hit in hits
                if hit["document"] == target_doc
                and all(_contains_term(hit["_full_text"], term) for term in target.get("terms", []))
            ]
            targets.append({
                "target": target["target"],
                "expected_document": target_doc,
                "terms": target.get("terms", []),
                "supported": bool(support),
                "supporting_chunk_ids": [hit["chunk_id"] for hit in support],
            })

        expected = str(spec.get("expected_document", "NONE"))
        if expected.upper() == "NONE":
            hit_values = {"hit@1": None, "hit@3": None, "hit@5": None}
        elif spec["category"] == "comparative":
            hit_values = {
                f"hit@{k}": int(all(
                    any(
                        hit["document"] == Path(target["expected_document"]).name
                        and all(_contains_term(hit["_full_text"], term) for term in target.get("terms", []))
                        for hit in hits[:k]
                    )
                    for target in spec.get("comparison_targets", [])
                ))
                for k in (1, 3, 5)
            }
        else:
            ranks = [hit["rank"] for hit in hits if hit["document"] == Path(expected).name]
            rank = min(ranks) if ranks else None
            hit_values = {f"hit@{k}": int(rank is not None and rank <= k) for k in (1, 3, 5)}
        unavailable = spec["evaluation_kind"] == "unavailable"
        low_relevance = spec["evaluation_kind"] == "low-relevance"
        rows.append({
            **spec,
            "expected_source_provenance": spec["expected_source_provenance"],
            "results": [
                {key: value for key, value in hit.items() if key != "_full_text"}
                for hit in hits
            ],
            "retrieval_confidence": float(result.retrieval_confidence),
            "sufficient_evidence": bool(result.sufficient_evidence),
            "no_relevant_information": bool(result.no_relevant_information),
            "comparison_target_evidence": targets,
            "hit_metrics": hit_values,
            "unavailable_false_positive": unavailable and bool(hits),
            "low_relevance_rejected": low_relevance and not hits and result.no_relevant_information,
        })
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Optional[float]]:
    return {
        metric: round(
            100.0 * sum(row["hit_metrics"][metric] for row in rows) / len(rows), 2
        ) if rows else None
        for metric in ("hit@1", "hit@3", "hit@5")
    }


def build_report(root: Path = ROOT, manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    query_path = _source_path(root, manifest["query_dataset"]["path"])
    query_payload = _read_json(query_path)
    registry = _source_registry(manifest, root)
    query_specs = _combined_queries(query_payload, manifest)
    rows = []
    for spec in query_specs:
        expected_source = _query_source(spec, registry)
        eligible, reason = _eligible(spec, expected_source, manifest, registry)
        rows.append({
            **spec,
            "query_provenance": {
                "provenance_class": spec.get("query_provenance_class", "UNVERIFIED_SOURCE"),
                "provenance_status": spec.get("query_provenance_status", spec.get("provenance_status", "UNVERIFIED")),
                "query_set": manifest["query_dataset"]["path"] if spec.get("query_id") not in {item.get("query_id") for item in manifest.get("additional_test_only_queries", [])} else "evaluation/m14_provenance_manifest.json",
            },
            "expected_source_provenance": expected_source,
            "acceptance_eligible": eligible,
            "eligibility_reason": reason,
            "results": [],
            "hit_metrics": {"hit@1": None, "hit@3": None, "hit@5": None},
            "unavailable_false_positive": None,
            "low_relevance_rejected": None,
        })

    positive_rows = [
        row for row in rows
        if row["acceptance_eligible"]
        and str(row.get("expected_document", "NONE")).upper() != "NONE"
    ]
    unavailable_rows = [
        row for row in rows
        if row["acceptance_eligible"]
        and row.get("evaluation_kind") == "unavailable"
    ]
    low_relevance_rows = [
        row for row in rows
        if row["acceptance_eligible"] and row.get("evaluation_kind") == "low-relevance"
    ]

    blockers = []
    for row in rows:
        if str(row.get("expected_document", "NONE")).upper() != "NONE" and not row["acceptance_eligible"]:
            expected = row["expected_source_provenance"]
            blockers.append({
                "query_id": row["query_id"],
                "expected_document": row["expected_document"],
                "provenance_class": expected["provenance_class"],
                "reason": row["eligibility_reason"],
                "source_path": expected.get("path"),
            })
    if not positive_rows:
        blockers.append({
            "requirement": "verified_real_source_corpus",
            "reason": "No query has a verified REAL_SOURCE_DOCUMENT as its expected answer source.",
            "missing_source_paths": sorted({
                item["path"] for item in registry.values()
                if item.get("provenance_class") == "GENERATED_SAMPLE"
                and item.get("path")
                and any(Path(spec.get("expected_document", "")).name == Path(item["path"]).name for spec in query_specs)
            }),
        })
    if any(row.get("evaluation_kind") in {"unavailable", "low-relevance"} for row in rows) and not (unavailable_rows or low_relevance_rows):
        blockers.append({
            "requirement": "complete_verified_negative_test_scope",
            "reason": "Unavailable and low-relevance rejection require complete, verified source coverage for the tested domain; no such domain scope is registered.",
            "complete_coverage_domains": manifest.get("complete_coverage_domains", []),
        })

    report = {
        "evaluation": "M1.4 provenance-aware retrieval validation",
        "status": "BLOCKED" if not positive_rows or blockers else "NOT_RUN",
        "acceptance_status": "BLOCKED" if not positive_rows or blockers else "NOT_RUN",
        "retrieval_algorithm_modified": False,
        "index_path": str(INDEX_PATH.relative_to(ROOT)),
        "manifest_path": str(manifest_path.relative_to(root)),
        "query_dataset": manifest["query_dataset"],
        "provenance_policy": {
            "accepted_class": "REAL_SOURCE_DOCUMENT",
            "generated_samples_count_for_acceptance": False,
            "test_fixtures_count_for_acceptance": False,
            "mock_retrievers_count_for_acceptance": False,
            "unregistered_sources_are": "UNVERIFIED_SOURCE",
        },
        "counts": {
            "query_count": len(rows),
            "valid_real_source_query_count": len(positive_rows),
            "synthetic_test_only_query_count": sum(not row["acceptance_eligible"] for row in rows),
            "generated_sample_expected_source_query_count": sum(
                row["expected_source_provenance"]["provenance_class"] == "GENERATED_SAMPLE"
                for row in rows
            ),
            "unavailable_query_count": sum(row.get("evaluation_kind") == "unavailable" for row in rows),
            "low_relevance_query_count": sum(row.get("category") == "low-relevance" for row in rows),
            "comparative_query_count": sum(row.get("category") == "comparative" for row in rows),
            "category_counts": {
                category: sum(row.get("evaluation_kind") == category for row in rows)
                for category in ("factual", "procedural", "comparative", "unavailable", "low-relevance")
            },
            "mock_retriever_test_suites": sum(item["provenance_class"] == "MOCK_RETRIEVER" for item in manifest.get("test_only_suites", [])),
        },
        "production_hit_at_k_percent": _metrics([]),
        "unavailable_false_positives": {
            "count": None,
            "eligible_queries": len(unavailable_rows),
            "status": "NOT PROVEN" if not unavailable_rows else "NOT RUN",
        },
        "low_relevance_rejection": {
            "rejected": None,
            "eligible_queries": len(low_relevance_rows),
            "status": "NOT PROVEN" if not low_relevance_rows else "NOT RUN",
        },
        "corpus_sources": list(registry.values()),
        "test_only_suites": manifest.get("test_only_suites", []),
        "missing_evidence": blockers,
        "queries": rows,
        "retrieval_diagnostics": {
            "status": "NOT RUN",
            "reason": "No verified real-source acceptance query is eligible. Synthetic/test-only corpus results are intentionally not substituted.",
        },
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["production_hit_at_k_percent"]
    lines = [
        "# M1.4 Retrieval Validation",
        "",
        f"**Status: {report['status']}**",
        "",
        "Only verified `REAL_SOURCE_DOCUMENT` records can contribute to production acceptance. Generated samples, test fixtures, and mock retrievers are excluded.",
        "",
        "## Acceptance",
        "",
        f"- Valid real-source queries: {counts['valid_real_source_query_count']}",
        f"- Synthetic/test-only queries: {counts['synthetic_test_only_query_count']}",
        f"- Generated-sample expected-source queries: {counts['generated_sample_expected_source_query_count']}",
        f"- Unavailable queries: {counts['unavailable_query_count']}",
        f"- Low-relevance queries: {counts['low_relevance_query_count']}",
        f"- Comparative queries: {counts['comparative_query_count']}",
        f"- Mock-retriever test suites classified separately: {counts['mock_retriever_test_suites']}",
        "",
        "| Query category | Cases |",
        "|---|---:|",
        *[
            f"| {category} | {count} |"
            for category, count in counts["category_counts"].items()
        ],
        "",
        "| Metric (valid real-source queries only) | Result |",
        "|---|---:|",
        f"| Hit@1 | {metrics['hit@1']} |",
        f"| Hit@3 | {metrics['hit@3']} |",
        f"| Hit@5 | {metrics['hit@5']} |",
        "",
        "`null` means not measured; it is not a zero score.",
        "",
        "## Negative Cases",
        "",
        f"- Unavailable false positives: {report['unavailable_false_positives']['count']} ({report['unavailable_false_positives']['status']}; eligible queries: {report['unavailable_false_positives']['eligible_queries']})",
        f"- Low-relevance rejection: {report['low_relevance_rejection']['rejected']} ({report['low_relevance_rejection']['status']}; eligible queries: {report['low_relevance_rejection']['eligible_queries']})",
        "",
        "## Missing Evidence",
        "",
    ]
    if not report["missing_evidence"]:
        lines.append("None.")
    else:
        for item in report["missing_evidence"]:
            if item.get("source_path"):
                lines.append(f"- `{item.get('query_id')}` expects `{item['expected_document']}` at `{item['source_path']}` ({item['provenance_class']}): {item['reason']}")
            else:
                lines.append(f"- {item.get('requirement', item.get('query_id', 'requirement'))}: {item['reason']}")
                for path in item.get("missing_source_paths", []):
                    lines.append(f"  - Generated source requiring an original: `{path}`")
    lines.extend(["", "## Query Provenance", "", "| ID | Category | Query provenance | Expected source provenance | Production eligible |", "|---|---|---|---|---|"])
    for row in report["queries"]:
        lines.append(
            f"| {row.get('query_id')} | {row.get('category')} | "
            f"{row['query_provenance']['provenance_class']} | "
            f"{row['expected_source_provenance']['provenance_class']} | "
            f"{row['acceptance_eligible']} |"
        )
    lines.extend([
        "",
        "## Test-Only Suites",
        "",
        "| Suite | Provenance | Counts toward acceptance |",
        "|---|---|---|"])
    for item in report["test_only_suites"]:
        lines.append(f"| `{item['path']}` | {item['provenance_class']} | {item['counts_toward_acceptance']} |")
    lines.extend([
        "",
        "## Decision",
        "",
        "M1.4 is BLOCKED until original source documents with verifiable origin records, source hashes, and complete domain scope are registered and evaluated. Do not interpret generated-sample results or mock-retriever tests as production evidence.",
        "",
    ])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_json: Path = DEFAULT_OUTPUT_JSON, output_md: Path = DEFAULT_OUTPUT_MD) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_render_markdown(report), encoding="utf-8")


def run(output_json: Path = DEFAULT_OUTPUT_JSON, output_md: Path = DEFAULT_OUTPUT_MD) -> dict[str, Any]:
    report = build_report()
    eligible = report["counts"]["valid_real_source_query_count"]
    if eligible:
        blockers = []
        if not INDEX_PATH.is_file() or not META_PATH.is_file():
            blockers.append("The evaluated FAISS index or metadata file is missing.")
        if blockers:
            report["status"] = report["acceptance_status"] = "BLOCKED"
            report["missing_evidence"].extend({"requirement": "evaluated_index", "reason": reason} for reason in blockers)
        else:
            try:
                from agents.retrieval_agent import RetrievalAgent
                from vector_store.store import VectorStore

                store = VectorStore(index_path=str(INDEX_PATH), meta_path=str(META_PATH))
                store.load()
                registered_paths = {
                    os.path.normcase(os.path.normpath(str(_source_path(ROOT, item["path"]))))
                    for item in report["corpus_sources"]
                    if item["counts_toward_acceptance"]
                }
                indexed_paths = {
                    os.path.normcase(os.path.normpath(str(_source_path(ROOT, str(item.get("source_location", ""))))))
                    for item in store.metadata.values()
                }
                if not registered_paths.issubset(indexed_paths):
                    raise ValueError("Verified real source documents are missing from the evaluated index.")
                non_real_indexed_sources = sorted(
                    {
                        str(item.get("source_location", ""))
                        for item in store.metadata.values()
                        if os.path.normcase(os.path.normpath(
                            str(_source_path(ROOT, str(item.get("source_location", ""))))
                        )) not in registered_paths
                    }
                )
                if non_real_indexed_sources:
                    raise ValueError(
                        "Index contains sources not approved as REAL_SOURCE_DOCUMENT: "
                        + ", ".join(non_real_indexed_sources)
                    )
                specs = [row for row in report["queries"] if row["acceptance_eligible"]]
                live_rows = _evaluate_live(store, specs, _source_registry(_read_json(MANIFEST_PATH), ROOT))
                report["queries"] = [
                    next((live for live in live_rows if live["query_id"] == row["query_id"]), row)
                    for row in report["queries"]
                ]
                positive_live = [
                    row for row in live_rows
                    if str(row.get("expected_document", "NONE")).upper() != "NONE"
                ]
                unavailable_live = [row for row in live_rows if row.get("evaluation_kind") == "unavailable"]
                low_live = [row for row in live_rows if row.get("evaluation_kind") == "low-relevance"]
                report["production_hit_at_k_percent"] = _metrics(positive_live)
                report["unavailable_false_positives"] = {
                    "count": sum(bool(row["unavailable_false_positive"]) for row in unavailable_live),
                    "eligible_queries": len(unavailable_live),
                    "status": "MEASURED",
                }
                report["low_relevance_rejection"] = {
                    "rejected": sum(bool(row["low_relevance_rejected"]) for row in low_live),
                    "eligible_queries": len(low_live),
                    "status": "MEASURED",
                }
                report["retrieval_diagnostics"] = {"status": "MEASURED", "reason": None}
                report["status"] = report["acceptance_status"] = (
                    "BLOCKED" if report["missing_evidence"] else "PASS"
                )
            except Exception as exc:
                report["status"] = report["acceptance_status"] = "BLOCKED"
                report["missing_evidence"].append({
                    "requirement": "live_provenance_aware_retrieval",
                    "reason": f"Live evaluation could not be proven: {type(exc).__name__}: {exc}",
                })
                report["retrieval_diagnostics"] = {"status": "BLOCKED", "reason": str(exc)}
    write_report(report, output_json, output_md)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()
    report = run(args.output_json, args.output_md)
    print(json.dumps({
        "status": report["status"],
        "valid_real_source_query_count": report["counts"]["valid_real_source_query_count"],
        "synthetic_test_only_query_count": report["counts"]["synthetic_test_only_query_count"],
        "production_hit_at_k_percent": report["production_hit_at_k_percent"],
        "report_json": str(args.output_json),
        "report_markdown": str(args.output_md),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
