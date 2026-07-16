import pytest
from unittest.mock import MagicMock, patch
from src.language_models.openai_language_model import OpenAILanguageModel

@patch('src.language_models.openai_language_model.ChatOpenAI')
def test_openai_language_model(mock_chat_openai, tmp_path):
    template_file = tmp_path / "template.txt"
    template_file.write_text("Hello {name}!")
    
    mock_instance = MagicMock()
    mock_chat_openai.return_value = mock_instance
    mock_instance.invoke.return_value = "Response"
    
    model = OpenAILanguageModel(template_path=str(template_file))
    
    prompt = model.generate_prompt(name="World")
    assert "Hello World!" in str(prompt)
    
    response = model.invoke(prompt)
    assert response == "Response"
    mock_instance.invoke.assert_called_once_with(prompt)

def test_openai_language_model_missing_template(tmp_path):
    with pytest.raises(FileNotFoundError):
        OpenAILanguageModel(template_path=str(tmp_path / "missing.txt"))
