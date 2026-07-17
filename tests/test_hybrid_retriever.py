from src.search.hybrid_retriever import reciprocal_rank_fusion

def test_reciprocal_rank_fusion():
    list1 = [
        {"url": "https://example.com/1", "title": "Doc 1"},
        {"url": "https://example.com/2", "title": "Doc 2"},
    ]
    list2 = [
        {"url": "https://example.com/2", "title": "Doc 2"},
        {"url": "https://example.com/3", "title": "Doc 3"},
    ]
    
    # Doc 2 is in both, so it should get a higher score.
    # List 1 ranks: Doc 1 (1), Doc 2 (2)
    # List 2 ranks: Doc 2 (1), Doc 3 (2)
    
    # Fusion should prioritize Doc 2
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)
    
    assert len(fused) == 3
    assert fused[0]["url"] == "https://example.com/2"
    # Doc 1 and Doc 3 will have the same score since they are both rank 1 or rank 2 
    # (Doc 1 is rank 1 in list1, Doc 3 is rank 2 in list2)
    # Wait, Doc 1 is rank 1 -> score = 1/61
    # Doc 3 is rank 2 -> score = 1/62
    # So Doc 1 should be second.
    assert fused[1]["url"] == "https://example.com/1"
    assert fused[2]["url"] == "https://example.com/3"

def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []

def test_rrf_missing_url():
    list1 = [
        {"title": "Doc 1 without URL"}, # Should be skipped if no metadata
        {"metadata": {"source": "https://example.com/fallback"}, "title": "Fallback"}
    ]
    fused = reciprocal_rank_fusion([list1])
    assert len(fused) == 1
    # RRF adapter uses doc.get("url", doc.get("metadata", {}).get("source", ""))
    assert fused[0].get("metadata", {}).get("source") == "https://example.com/fallback"
