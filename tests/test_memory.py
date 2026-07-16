from src.memory import ConversationMemory

def test_memory_add_and_retrieve():
    mem = ConversationMemory(max_turns=3)
    
    assert mem.as_string() == "(no prior conversation)"
    
    mem.add_turn("Q1", "A1")
    assert "Q1" in mem.as_string()
    assert "A1" in mem.as_string()
    
    mem.add_turn("Q2", "A2")
    mem.add_turn("Q3", "A3")
    
    # Exceed max_turns
    mem.add_turn("Q4", "A4")
    
    # Since max_turns = 3, Q1 should be evicted
    history = mem.as_string()
    assert "Q1" not in history
    assert "Q2" in history
    assert "Q4" in history
    
def test_memory_get_all():
    mem = ConversationMemory(max_turns=2)
    mem.add_turn("Q1", "A1")
    mem.add_turn("Q2", "A2")
    
    turns = mem.as_messages()
    assert len(turns) == 4
    assert turns[0]["content"] == "Q1"
    assert turns[1]["content"] == "A1"
