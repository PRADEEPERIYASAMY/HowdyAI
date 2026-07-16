import pytest
from unittest.mock import MagicMock, patch
from src.embeddings import Embeddings

@patch('src.embeddings.OpenAIEmbeddings')
def test_embeddings(mock_openai_embeddings):
    mock_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_instance
    mock_instance.embed_documents.return_value = [[1.0, 0.0], [0.0, 1.0]]
    mock_instance.embed_query.return_value = [0.5, 0.5]
    
    emb = Embeddings()
    
    docs = ["Doc 1", "Doc 2"]
    res_docs = emb.embed_documents(docs)
    assert len(res_docs) == 2
    mock_instance.embed_documents.assert_called_once_with(docs)
    
    res_query = emb.embed_query("Query")
    assert len(res_query) == 2
    mock_instance.embed_query.assert_called_once_with("Query")
