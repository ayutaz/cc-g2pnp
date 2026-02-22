import pytest
from transformers import AutoTokenizer

from cc_g2pnp.data.tokenizer import G2PnPTokenizer

CALM2_MODEL_NAME = "cyberagent/calm2-7b"


@pytest.fixture(scope="session")
def calm2_tokenizer():
    """Load CALM2 tokenizer once per test session."""
    return AutoTokenizer.from_pretrained(CALM2_MODEL_NAME, trust_remote_code=True)


@pytest.fixture(scope="session")
def g2pnp_tokenizer():
    """Load G2PnPTokenizer once per test session."""
    return G2PnPTokenizer()
