import pytest
from unittest.mock import MagicMock, patch
from src.search.hybrid_retriever import HybridRetriever

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.NUM_SEARCH_RESULTS = 2
    config.CHROMA_NUM_RESULTS = 2
    config.FUSION_TOP_N = 8
    config.CHROMA_PATH = "dummy/path"
    config.OPENAPI_API_KEY = "test_key"
    config.FAST_MODEL = "gpt-4o-mini"
    config.STRONG_MODEL = "gpt-4o"
    config.USE_HYDE = False
    return config

@patch('src.search.hybrid_retriever.Database')
@patch('src.search.hybrid_retriever.brave_search_engine')
def test_hybrid_retriever_both_sources(mock_ddg, mock_db_class, mock_config):
    mock_db = MagicMock()
    mock_db_class.return_value = mock_db
    
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "http://chroma.edu", "title": "Chroma Title"}
    mock_doc.page_content = "Chroma content"
    mock_db.search.return_value = [(mock_doc, 0.5)]
    
    # Mock DDG search
    mock_ddg.return_value = [
        {"url": "http://ddg.edu", "title": "DDG Title", "description": "DDG content", "metadata": {"content": "DDG content"}}
    ]
    
    retriever = HybridRetriever(mock_config)
    results = retriever.retrieve("test query")
    
    assert len(results) == 2
    urls = [r["url"] for r in results]
    assert "http://ddg.edu" in urls
    assert "http://chroma.edu" in urls

@patch('src.search.hybrid_retriever.Database')
@patch('src.search.hybrid_retriever.brave_search_engine')
def test_hybrid_retriever_ddg_fail(mock_ddg, mock_db_class, mock_config):
    mock_db = MagicMock()
    mock_db_class.return_value = mock_db
    
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "http://chroma.edu", "title": "Chroma Title"}
    mock_doc.page_content = "Chroma content"
    mock_db.search.return_value = [(mock_doc, 0.8)]
    
    mock_ddg.side_effect = Exception("DDG blocked")
    
    retriever = HybridRetriever(mock_config)
    results = retriever.retrieve("test query")
    
    assert len(results) == 1
    assert results[0]["url"] == "http://chroma.edu"

@patch('src.search.hybrid_retriever.Database')
@patch('src.search.hybrid_retriever.brave_search_engine')
def test_hybrid_retriever_chroma_fail(mock_ddg, mock_db_class, mock_config):
    # Pass db as None to simulate Chroma unavailable
    mock_db_class.side_effect = Exception("No DB")
    
    mock_ddg.return_value = [
        {"url": "http://ddg.edu", "title": "DDG Title", "description": "DDG content"}
    ]
    
    retriever = HybridRetriever(mock_config)
    results = retriever.retrieve("test query")
    
    assert len(results) == 1
    assert results[0]["url"] == "http://ddg.edu"
    assert retriever._chroma_available is False
