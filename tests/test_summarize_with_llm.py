import pytest
from unittest.mock import MagicMock, patch
from src.search.summarize_with_llm import summarize_search_results_with_llm

@patch('src.search.summarize_with_llm.OpenAILanguageModel')
def test_summarize_search_results(mock_llm_class):
    mock_config = MagicMock()
    mock_config.SUMMARY_TEMPLATE_PATH = "dummy.txt"
    
    mock_model_instance = MagicMock()
    
    class MockResponse:
        def __init__(self, content):
            self.content = content
            
    mock_model_instance.invoke.return_value = MockResponse("Summarized content")
    mock_llm_class.return_value = mock_model_instance
    
    results = [
        {"url": "http://1", "title": "T1", "description": "D1", "metadata": {"content": "Long content 1"}},
        {"url": "http://2", "title": "T2", "description": "D2", "metadata": {"content": "Long content 2"}}
    ]
    
    filtered_results, context_str = summarize_search_results_with_llm(mock_config, "test query", results)
    
    assert len(filtered_results) == 2
    assert filtered_results[0]["llm_summary"] == "Summarized content"
    assert "Summarized content" in context_str

@patch('src.search.summarize_with_llm.OpenAILanguageModel')
def test_summarize_search_results_error(mock_llm_class):
    mock_config = MagicMock()
    mock_config.SUMMARY_TEMPLATE_PATH = "dummy.txt"
    
    mock_model_instance = MagicMock()
    mock_model_instance.invoke.side_effect = Exception("API error")
    mock_llm_class.return_value = mock_model_instance
    
    results = [
        {"url": "http://1", "title": "T1", "description": "D1", "metadata": {"content": "Long content 1"}}
    ]
    
    filtered_results, context_str = summarize_search_results_with_llm(mock_config, "test query", results)
    
    assert len(filtered_results) == 1
    assert filtered_results[0]["llm_summary"] == ""
