"""CC-G2PnP model: Conformer encoder with CTC decoder for grapheme-to-phoneme."""

from __future__ import annotations

from cc_g2pnp.model.attention import (
    ChunkAwareAttention,
    create_chunk_mask,
    create_mla_mask,
)
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.conformer_block import ConformerBlock
from cc_g2pnp.model.convolution import ConformerConvModule
from cc_g2pnp.model.ctc_decoder import CTCHead, greedy_decode
from cc_g2pnp.model.embedding import TokenEmbedding
from cc_g2pnp.model.encoder import ConformerEncoder
from cc_g2pnp.model.feed_forward import FeedForwardModule
from cc_g2pnp.model.positional_encoding import RelativePositionalEncoding

__all__ = [
    "CC_G2PnP",
    "CC_G2PnPConfig",
    "CTCHead",
    "ChunkAwareAttention",
    "ConformerBlock",
    "ConformerConvModule",
    "ConformerEncoder",
    "FeedForwardModule",
    "RelativePositionalEncoding",
    "TokenEmbedding",
    "create_chunk_mask",
    "create_mla_mask",
    "greedy_decode",
]
