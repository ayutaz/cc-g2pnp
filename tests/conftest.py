import pytest
from transformers import AutoTokenizer

CALM2_MODEL_NAME = "cyberagent/calm2-7b"


@pytest.fixture(scope="session")
def calm2_tokenizer():
    """Load CALM2 tokenizer once per test session."""
    return AutoTokenizer.from_pretrained(CALM2_MODEL_NAME, trust_remote_code=True)
