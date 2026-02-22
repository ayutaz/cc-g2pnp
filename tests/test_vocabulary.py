"""Tests for the PnP CTC vocabulary."""

from cc_g2pnp.data.vocabulary import PHONEMES, PROSODY_SYMBOLS, PnPVocabulary


def test_vocab_size():
    """Vocabulary size should be exactly 140 (blank + 134 mora + 3 prosody + unk + pad)."""
    vocab = PnPVocabulary()
    assert vocab.vocab_size == 140


def test_blank_is_zero():
    """CTC blank token must be ID 0."""
    vocab = PnPVocabulary()
    assert vocab.blank_id == 0
    assert vocab.id_to_token[0] == "<blank>"


def test_all_phonemes_present():
    """Every phoneme in the inventory should have a valid ID."""
    vocab = PnPVocabulary()
    for phoneme in PHONEMES:
        assert phoneme in vocab.token_to_id, f"Missing phoneme: {phoneme}"


def test_prosody_symbols_present():
    """Prosody symbols *, /, # should have valid IDs."""
    vocab = PnPVocabulary()
    for sym in PROSODY_SYMBOLS:
        assert sym in vocab.token_to_id, f"Missing prosody symbol: {sym}"


def test_special_tokens():
    """<blank>, <unk>, <pad> should all have valid IDs."""
    vocab = PnPVocabulary()
    assert "<blank>" in vocab.token_to_id
    assert "<unk>" in vocab.token_to_id
    assert "<pad>" in vocab.token_to_id


def test_encode_decode_roundtrip():
    """Encoding then decoding should recover the original tokens."""
    vocab = PnPVocabulary()
    tokens = ["キョ", "*", "ー", "ワ", "/", "イ", "ー"]
    ids = vocab.encode(tokens)
    decoded = vocab.decode(ids)
    assert decoded == tokens


def test_encode_unknown_token():
    """Unknown tokens should map to <unk> ID."""
    vocab = PnPVocabulary()
    ids = vocab.encode(["NOTREAL"])
    assert ids == [vocab.unk_id]


def test_no_duplicate_ids():
    """Each token should map to a unique ID."""
    vocab = PnPVocabulary()
    ids = list(vocab.token_to_id.values())
    assert len(ids) == len(set(ids))


def test_loanword_tokens_encode_decode():
    """Loanword tokens added for foreign words should encode/decode correctly."""
    vocab = PnPVocabulary()
    loanword_tokens = ["シェ", "ジェ", "チェ", "ディャ", "ディョ", "フュ"]
    ids = vocab.encode(loanword_tokens)
    # None should map to <unk>
    for tok, tok_id in zip(loanword_tokens, ids):
        assert tok_id != vocab.unk_id, f"{tok} mapped to <unk>"
    # Roundtrip
    decoded = vocab.decode(ids)
    assert decoded == loanword_tokens
