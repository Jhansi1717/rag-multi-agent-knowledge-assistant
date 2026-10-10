"""Provenance-gate tests for the M1.4 evaluation framework."""

from types import SimpleNamespace

from evaluation.m14_provenance import _serialize_hit, build_report


def test_current_benchmark_is_blocked_and_excludes_generated_sources():
    report = build_report()

    assert report["status"] == "BLOCKED"
    assert report["retrieval_algorithm_modified"] is False
    assert report["counts"]["valid_real_source_query_count"] == 0
    assert report["counts"]["synthetic_test_only_query_count"] == 28
    assert report["counts"]["generated_sample_expected_source_query_count"] == 15
    assert report["counts"]["unavailable_query_count"] == 4
    assert report["counts"]["low_relevance_query_count"] == 1
    assert report["production_hit_at_k_percent"] == {
        "hit@1": None,
        "hit@3": None,
        "hit@5": None,
    }


def test_every_query_reports_query_and_expected_source_provenance():
    report = build_report()

    assert len(report["queries"]) == report["counts"]["query_count"]
    for query in report["queries"]:
        assert query["query_provenance"]["provenance_class"] == "TEST_FIXTURE"
        assert query["expected_source_provenance"]["provenance_class"] in {
            "GENERATED_SAMPLE",
            "NO_EXPECTED_SOURCE",
        }
        assert query["acceptance_eligible"] is False


def test_comparative_cases_declare_and_require_both_targets():
    report = build_report()
    comparisons = [row for row in report["queries"] if row["category"] == "comparative"]

    assert len(comparisons) == 4
    assert all(len(row["comparison_targets"]) == 2 for row in comparisons)
    assert all(
        all(target["expected_document"] for target in row["comparison_targets"])
        for row in comparisons
    )


def test_negative_results_are_not_claimed_without_complete_real_corpus():
    report = build_report()

    assert report["unavailable_false_positives"] == {
        "count": None,
        "eligible_queries": 0,
        "status": "NOT PROVEN",
    }
    assert report["low_relevance_rejection"] == {
        "rejected": None,
        "eligible_queries": 0,
        "status": "NOT PROVEN",
    }
    assert report["counts"]["mock_retriever_test_suites"] == 1
    assert all(
        suite["counts_toward_acceptance"] is False
        for suite in report["test_only_suites"]
    )


def test_result_serialization_preserves_required_provenance_fields():
    hit = SimpleNamespace(
        rank=1,
        filename="patient_admission_policy.pdf",
        document_id="sample-document",
        chunk_id="sample-chunk-0",
        distance_score=0.25,
        relevance_score=0.8,
        text="Patient admission evidence on page 2, section registration, row 4.",
        metadata={
            "domain": "Hospital Administration",
            "source_location": "data/hospital_administration/patient_admission_policy.pdf",
            "page_number": 2,
            "section": "Registration",
            "row_number": 4,
        },
    )
    report = build_report()
    registry = {
        item["path"].split("/")[-1]: item
        for item in report["corpus_sources"]
    }

    result = _serialize_hit(hit, registry)

    assert result["document"] == hit.filename
    assert result["chunk_id"] == hit.chunk_id
    assert result["domain"] == "Hospital Administration"
    assert result["distance_score"] == 0.25
    assert result["relevance_score"] == 0.8
    assert result["location_metadata"] == {
        "page_number": 2,
        "section": "Registration",
        "row_number": 4,
    }
    assert result["provenance_class"] == "GENERATED_SAMPLE"
    assert result["counts_toward_acceptance"] is False
