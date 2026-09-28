import pytest
from rag_retrieval import RankedResults, Result


def test_result_creation_and_to_dict():
    res = Result(doc_id="doc_1", text="Machine learning is fascinating.", score=0.95, rank=1)
    assert res.doc_id == "doc_1"
    assert res.text == "Machine learning is fascinating."
    assert res.score == 0.95
    assert res.rank == 1

    d = res.to_dict()
    assert d["doc_id"] == "doc_1"
    assert d["score"] == 0.95
    assert d["rank"] == 1


def test_result_validation():
    # Should work with only score
    res1 = Result(doc_id=1, text="Test", score=0.8)
    assert res1.score == 0.8

    # Should work with only rank
    res2 = Result(doc_id=2, text="Test", rank=1)
    assert res2.rank == 1

    # Should raise ValueError when neither score nor rank is given
    with pytest.raises(ValueError):
        Result(doc_id=3, text="Test")


def test_ranked_results_container_methods():
    results = [
        Result(doc_id=1, text="Doc A", score=0.9, rank=1),
        Result(doc_id=2, text="Doc B", score=0.7, rank=2),
        Result(doc_id=3, text="Doc C", score=0.5, rank=3),
    ]
    ranked = RankedResults(results=results, query="What is A?", has_scores=True)

    # __len__
    assert len(ranked) == 3
    assert ranked.results_count() == 3

    # __iter__
    docs = [r.text for r in ranked]
    assert docs == ["Doc A", "Doc B", "Doc C"]

    # __getitem__ single index
    assert ranked[0].doc_id == 1
    assert ranked[1].doc_id == 2

    # __getitem__ slice returns RankedResults
    sliced = ranked[0:2]
    assert isinstance(sliced, RankedResults)
    assert len(sliced) == 2
    assert sliced.query == "What is A?"
    assert sliced[0].doc_id == 1
    assert sliced[1].doc_id == 2


def test_ranked_results_top_k():
    results = [
        Result(doc_id=1, text="Doc A", score=0.9, rank=1),
        Result(doc_id=2, text="Doc B", score=0.7, rank=2),
        Result(doc_id=3, text="Doc C", score=0.5, rank=3),
    ]
    ranked = RankedResults(results=results, query="test", has_scores=True)
    top_2 = ranked.top_k(2)
    assert len(top_2) == 2
    assert top_2[0].doc_id == 1
    assert top_2[1].doc_id == 2


def test_ranked_results_get_score_by_docid():
    results = [
        Result(doc_id="a", text="Doc A", score=0.9, rank=1),
        Result(doc_id="b", text="Doc B", score=0.7, rank=2),
    ]
    ranked = RankedResults(results=results, query="test", has_scores=True)
    assert ranked.get_score_by_docid("a") == 0.9
    assert ranked.get_score_by_docid("b") == 0.7
    assert ranked.get_score_by_docid("c") is None


def test_ranked_results_filtering():
    results = [
        Result(doc_id=1, text="Doc A", score=0.95, rank=1),
        Result(doc_id=2, text="Doc B", score=0.75, rank=2),
        Result(doc_id=3, text="Doc C", score=0.45, rank=3),
        Result(doc_id=4, text="Doc D", score=0.20, rank=4),
    ]
    ranked = RankedResults(results=results, query="filter test", has_scores=True)

    # Filter by score >= 0.70
    filtered_score = ranked.filter_by_score(0.70)
    assert len(filtered_score) == 2
    assert filtered_score.get_doc_ids() == [1, 2]
    assert filtered_score.query == "filter test"

    # Filter by rank <= 2
    filtered_rank = ranked.filter_by_rank(2)
    assert len(filtered_rank) == 2
    assert filtered_rank.get_doc_ids() == [1, 2]


def test_ranked_results_helpers_and_export():
    results = [
        Result(doc_id=10, text="Doc 10", score=0.88, rank=1),
        Result(doc_id=20, text="Doc 20", score=0.66, rank=2),
    ]
    ranked = RankedResults(results=results, query="export test", has_scores=True)

    assert ranked.get_texts() == ["Doc 10", "Doc 20"]
    assert ranked.get_doc_ids() == [10, 20]

    dict_export = ranked.to_dict()
    assert dict_export["query"] == "export test"
    assert dict_export["has_scores"] is True
    assert len(dict_export["results"]) == 2
    assert dict_export["results"][0]["doc_id"] == 10

    list_export = ranked.to_list()
    assert len(list_export) == 2
    assert list_export[0]["doc_id"] == 10
    assert list_export[1]["score"] == 0.66


if __name__ == "__main__":
    pytest.main([__file__])
