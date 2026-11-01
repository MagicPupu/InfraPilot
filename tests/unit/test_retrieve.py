from infrapilot.rag.retrieve import or_tsquery, rrf_merge


def test_rrf_prefers_documents_found_by_both_retrievers():
    # "b" is second in both lists; "a" and "c" each appear once, at rank 1.
    merged = rrf_merge([["a", "b", "x"], ["c", "b", "y"]])
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "x", "y"}


def test_rrf_scores_match_the_formula():
    # one list: ranks 1, 2 -> 1/61 > 1/62, order is preserved
    assert rrf_merge([["a", "b"]]) == ["a", "b"]


def test_rrf_is_deterministic_on_ties():
    assert rrf_merge([["a"], ["b"]]) == rrf_merge([["a"], ["b"]]) == ["a", "b"]


def test_rrf_handles_empty_lists():
    assert rrf_merge([[], []]) == []
    assert rrf_merge([["a"], []]) == ["a"]


def test_or_tsquery_is_safe_and_deduplicated():
    assert or_tsquery("What is the limit? limit!") == "what | is | the | limit"
    assert or_tsquery("'; DROP TABLE chunks; --") == "drop | table | chunks"
    assert or_tsquery("???") == ""
