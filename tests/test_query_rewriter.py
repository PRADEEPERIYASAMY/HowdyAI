import pytest
from unittest.mock import MagicMock
from src.search.query_rewriter import QueryRewriter

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.REWRITE_TEMPLATE_PATH = "dummy.txt"
    return config

def test_query_rewriter(mock_config, monkeypatch):
    mock_model_class = MagicMock()
    mock_model_instance = MagicMock()
    
    # Mock LLM response
    class MockResponse:
        def __init__(self, content):
            self.content = content
            
    mock_model_instance.invoke.return_value = MockResponse("Texas A&M University admissions requirements")
    mock_model_class.return_value = mock_model_instance
    
    monkeypatch.setattr("src.search.query_rewriter.OpenAILanguageModel", mock_model_class)
    
    rewriter = QueryRewriter(mock_config)
    rewritten = rewriter.rewrite("what are the requirements to get in?", history="")
    
    assert rewritten == "Texas A&M University admissions requirements"
    mock_model_instance.invoke.assert_called_once()
