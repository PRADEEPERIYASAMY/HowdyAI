import pytest
from unittest.mock import MagicMock, patch
from src.database import Database
from config import AppConfig

@patch('src.database.Chroma')
@patch('src.database.Embeddings')
def test_database_init(mock_embeddings, mock_chroma):
    mock_db_instance = MagicMock()
    mock_chroma.return_value = mock_db_instance
    
    db = Database("dummy/path")
    assert db.db == mock_db_instance

@patch('src.database.Chroma')
@patch('src.database.Embeddings')
def test_database_search(mock_embeddings, mock_chroma):
    mock_db_instance = MagicMock()
    mock_chroma.return_value = mock_db_instance
    mock_doc = MagicMock()
    mock_doc.page_content = "content"
    mock_db_instance.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.9)]
    
    db = Database("dummy/path")
    res = db.search("query")
    assert res == [(mock_doc, 0.9)]
