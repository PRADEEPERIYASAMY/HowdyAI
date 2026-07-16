import pytest
import numpy as np
from unittest.mock import MagicMock
from create_database import split_into_sentences, semantic_chunk

def test_split_into_sentences():
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes it is."
    sentences = split_into_sentences(text)
    assert len(sentences) >= 3
    assert "This is sentence one." in text
    
def test_semantic_chunking(monkeypatch):
    text = "This is a very long Sentence A. This is a very long Sentence B. This is a very long Sentence C. This is a very long Sentence D."
    
    # Mock the embedding function to return specific vectors
    # Let's say A and B are similar, C and D are similar, but B and C are very different.
    def mock_embed(sentences, client):
        vecs = []
        for s in sentences:
            if "A" in s or "B" in s:
                vecs.append(np.array([1.0, 0.0, 0.0]))
            else:
                vecs.append(np.array([0.0, 1.0, 0.0]))
        return vecs
        
    monkeypatch.setattr("create_database.embed_sentences", mock_embed)
    
    # Mock OpenAI client
    mock_client = MagicMock()
    
    # Adjust thresholds for test
    monkeypatch.setattr("create_database.MIN_CHUNK_CHARS", 10)
    
    chunks = semantic_chunk(text, mock_client)
    
    # Expect 2 chunks because B and C have 0.0 cosine similarity (which is < 0.80 threshold)
    assert len(chunks) == 2
    assert "Sentence A" in chunks[0]
    assert "Sentence B" in chunks[0]
    # OVERLAP_SENTENCES = 1, so Sentence B should also carry over into the next chunk
    assert "Sentence B" in chunks[1]
    assert "Sentence C" in chunks[1]
