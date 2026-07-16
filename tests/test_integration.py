import pytest
from unittest.mock import MagicMock, patch
from main import run_pipeline
from config import AppConfig
from src.cache import ResponseCache
from src.memory import ConversationMemory
from src.guardrail_rewriter import GuardrailRewriter
from src.search.hybrid_retriever import HybridRetriever
from src.language_models.openai_language_model import OpenAILanguageModel

class MockResponse:
    def __init__(self, content):
        self.content = content

@pytest.fixture
def mocked_pipeline_components(monkeypatch, tmp_path):
    config = AppConfig()
    
    # Use temporary cache
    cache = ResponseCache(db_path=str(tmp_path / "test_cache.db"))
    memory = ConversationMemory()
    
    # Mock GuardrailRewriter
    guardrail_rewriter = MagicMock(spec=GuardrailRewriter)
    guardrail_rewriter.check_and_rewrite.return_value = (True, "rewritten query", "")
    
    # Mock Retriever
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve.return_value = [
        {"url": "https://tamu.edu/1", "title": "Doc 1", "llm_summary": "Summary 1"},
        {"url": "https://tamu.edu/2", "title": "Doc 2", "llm_summary": "Summary 2"}
    ]
    
    # Mock LLM generation
    def mock_invoke(prompt):
        if "grading an AI's response" in prompt:
            return MockResponse("NO")
        return MockResponse("This is a mocked answer with [1] and [2].")
        
    generator = MagicMock(spec=OpenAILanguageModel)
    generator.invoke.side_effect = mock_invoke
    
    
    return config, cache, memory, guardrail_rewriter, retriever, generator

@patch('langchain_openai.ChatOpenAI')
def test_full_pipeline_integration(mock_chatopenai, mocked_pipeline_components):
    mock_judge = MagicMock()
    mock_judge.invoke.return_value = MockResponse("FINAL: FAITHFUL")
    mock_chatopenai.return_value = mock_judge

    config, cache, memory, guardrail_rewriter, retriever, generator = mocked_pipeline_components
    
    query = "What is the admissions deadline?"
    
    result = run_pipeline(
        query=query,
        config=config,
        cache=cache,
        memory=memory,
        guardrail_rewriter=guardrail_rewriter,
        retriever=retriever,
        generator=generator,
        use_cache=True
    )
    
    assert result["cache_hit"] is False
    assert result["guardrail_hit"] is False
    assert result["rewritten_query"] == "rewritten query"
    assert "mocked answer" in result["answer"]
    assert len(result["sources"]) == 2
    
    # Run again to test cache hit
    result_cached = run_pipeline(
        query=query,
        config=config,
        cache=cache,
        memory=memory,
        guardrail_rewriter=guardrail_rewriter,
        retriever=retriever,
        generator=generator,
        use_cache=True
    )
    
    assert result_cached["cache_hit"] is True
    assert result_cached["answer"] == result["answer"]
