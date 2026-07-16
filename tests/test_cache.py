import os
import pytest
from src.cache import ResponseCache

@pytest.fixture
def cache_db(tmp_path):
    db_file = tmp_path / "test_cache.db"
    cache = ResponseCache(db_path=str(db_file))
    yield cache
    # teardown
    cache.clear()

def test_cache_hashing(cache_db):
    q1 = "What is the mascot?"
    q2 = "what is the mascot?"
    q3 = "  What is the mascot?  "
    
    h1 = cache_db._hash(q1)
    h2 = cache_db._hash(q2)
    h3 = cache_db._hash(q3)
    
    assert h1 == h2 == h3

def test_cache_get_set(cache_db):
    query = "How many credits is CSCE 110?"
    response = {"answer": "CSCE 110 is 4 credits.", "sources": []}
    
    assert cache_db.get(query) is None
    
    cache_db.set(query, response)
    
    cached_response = cache_db.get(query)
    assert cached_response is not None
    assert cached_response["answer"] == "CSCE 110 is 4 credits."
    
def test_cache_clear(cache_db):
    query = "Test query"
    cache_db.set(query, {"answer": "Test answer"})
    assert cache_db.stats()["entries"] == 1
    
    cache_db.clear()
    assert cache_db.stats()["entries"] == 0
