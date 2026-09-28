import pytest
from unittest.mock import MagicMock
from rag_retrieval.infer.reranker_models.cross_encoder_ranker import CorssEncoderRanker, CrossEncoderRanker
from rag_retrieval.infer.reranker_models.result import RankedResults


def test_cross_encoder_aliases():
    assert CrossEncoderRanker is CorssEncoderRanker


def test_cross_encoder_rerank_filtering_and_top_k():
    # Test rerank logic by mocking compute_score
    ranker = object.__new__(CrossEncoderRanker)
    ranker.verbose = 0
    ranker.compute_score = MagicMock(return_value=[0.1, 0.9, 0.5, 0.8])

    docs = ["doc0", "doc1", "doc2", "doc3"]
    query = "test query"

    # 1. Standard rerank
    res = ranker.rerank(query, docs, long_doc_process_strategy="max_length_truncation")
    assert isinstance(res, RankedResults)
    assert len(res) == 4
    assert res.get_doc_ids() == [1, 3, 2, 0]

    # 2. top_k=2
    res_top_k = ranker.rerank(query, docs, long_doc_process_strategy="max_length_truncation", top_k=2)
    assert len(res_top_k) == 2
    assert res_top_k.get_doc_ids() == [1, 3]

    # 3. score_threshold=0.6
    res_threshold = ranker.rerank(query, docs, long_doc_process_strategy="max_length_truncation", score_threshold=0.6)
    assert len(res_threshold) == 2
    assert res_threshold.get_doc_ids() == [1, 3]

    # 4. Combined top_k and score_threshold
    res_combined = ranker.rerank(
        query, docs,
        long_doc_process_strategy="max_length_truncation",
        score_threshold=0.4,
        top_k=2
    )
    assert len(res_combined) == 2
    assert res_combined.get_doc_ids() == [1, 3]

    # 5. Empty docs or query
    empty_res = ranker.rerank("", docs)
    assert isinstance(empty_res, RankedResults)
    assert len(empty_res) == 0


if __name__ == "__main__":
    pytest.main([__file__])
