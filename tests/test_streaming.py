"""StreamingInference 単体テスト。"""

from __future__ import annotations

import torch

from cc_g2pnp.inference.streaming import StreamingInference, StreamingState
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.encoder import EncoderStreamingState


def _make_engine() -> tuple[StreamingInference, CC_G2PnPConfig]:
    """テスト用の小さいモデルと StreamingInference を生成する。"""
    config = CC_G2PnPConfig(
        d_model=32,
        num_heads=2,
        num_layers=2,
        d_ff=64,
        conv_kernel_size=3,
        intermediate_ctc_layers=(0,),
    )
    model = CC_G2PnP(config)
    model.eval()
    engine = StreamingInference(model)
    return engine, config


def test_reset_returns_streaming_state() -> None:
    """reset() が StreamingState を返し、初期化済みフィールドを持つことを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    assert isinstance(state, StreamingState)
    assert isinstance(state.encoder_state, EncoderStreamingState)
    # frame_buffer は [B, 0, D] で初期化される
    assert state.frame_buffer.shape == (1, 0, config.d_model)
    assert state.encoder_state.processed_frames == 0


def test_process_tokens_output_shape() -> None:
    """process_tokens() の出力が正しい形状 (batch_size 個のリスト) を持つことを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    bpe_ids = torch.randint(0, config.bpe_vocab_size, (1, 2))
    labels, new_state = engine.process_tokens(bpe_ids, state)

    # labels は List[List[int]] で、長さは batch_size
    assert isinstance(labels, list)
    assert len(labels) == 1
    assert isinstance(labels[0], list)
    # new_state も StreamingState であること
    assert isinstance(new_state, StreamingState)
    # frame_buffer の batch 次元が保持されていること
    assert new_state.frame_buffer.size(0) == 1


def test_process_tokens_returns_new_state() -> None:
    """process_tokens() 呼び出し後に状態が更新されることを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    # 1 トークン -> upsample_factor=8 フレーム -> frames_needed=chunk+mla=5+1=6 -> 1チャンク処理
    token = torch.randint(0, config.bpe_vocab_size, (1, 1))
    _, new_state = engine.process_tokens(token, state)

    # 初期の state と new_state は別オブジェクトでなければならない
    assert new_state is not state
    # 少なくとも 1 チャンク分処理されているか、バッファに残りがあること
    total = new_state.encoder_state.processed_frames + new_state.frame_buffer.size(1)
    assert total > 0


def test_flush_returns_remaining() -> None:
    """flush() がバッファに残ったフレームを処理し、空バッファを返すことを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    # トークンを供給してバッファに残りを作る
    token = torch.randint(0, config.bpe_vocab_size, (1, 1))
    _, state_after = engine.process_tokens(token, state)

    labels, final_state = engine.flush(state_after)

    # 戻り値の型チェック
    assert isinstance(labels, list)
    assert len(labels) == 1
    assert isinstance(labels[0], list)
    # flush 後はバッファが空になること
    assert final_state.frame_buffer.size(1) == 0


def test_multiple_chunks() -> None:
    """複数チャンクの逐次処理が正常に動作することを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    # 複数回のトークン供給を繰り返す
    all_labels: list[int] = []
    for _ in range(3):
        tokens = torch.randint(0, config.bpe_vocab_size, (1, 2))
        labels, state = engine.process_tokens(tokens, state)
        all_labels.extend(labels[0])

    # 残りをフラッシュ
    flush_labels, final_state = engine.flush(state)
    all_labels.extend(flush_labels[0])

    # 処理全体を通じてエラーなく完了し、リストが返ること
    assert isinstance(all_labels, list)
    # フラッシュ後はバッファが空
    assert final_state.frame_buffer.size(1) == 0
    # 複数チャンク分の処理がなされていること
    assert final_state.encoder_state.processed_frames > 0


def test_single_token_input() -> None:
    """単一トークン入力のエッジケース: 有効な結果が返ることを確認。"""
    engine, config = _make_engine()
    state = engine.reset(batch_size=1)

    # 1 トークンだけ処理する (upsample_factor=8 -> 8 フレーム)
    single_token = torch.randint(0, config.bpe_vocab_size, (1, 1))
    labels, new_state = engine.process_tokens(single_token, state)

    assert isinstance(labels, list)
    assert len(labels) == 1
    assert isinstance(labels[0], list)
    # 全ラベルが非負整数であること
    for lbl in labels[0]:
        assert isinstance(lbl, int)
        assert lbl >= 0

    # バッファに残ったフレームは frames_needed 未満であること
    frames_needed = config.chunk_size + config.mla_size
    assert new_state.frame_buffer.size(1) < frames_needed


def test_batch_size_greater_than_one() -> None:
    """batch_size > 1 の場合に各バッチの結果が正しく返ることを確認。"""
    engine, config = _make_engine()
    batch_size = 3
    state = engine.reset(batch_size=batch_size)

    # 初期バッファの batch 次元チェック
    assert state.frame_buffer.size(0) == batch_size

    bpe_ids = torch.randint(0, config.bpe_vocab_size, (batch_size, 2))
    labels, new_state = engine.process_tokens(bpe_ids, state)

    # labels は batch_size 個のリストを含むこと
    assert len(labels) == batch_size
    for item in labels:
        assert isinstance(item, list)

    # バッファの batch 次元が保持されていること
    assert new_state.frame_buffer.size(0) == batch_size

    # flush でも batch 次元が保持されること
    flush_labels, final_state = engine.flush(new_state)
    assert len(flush_labels) == batch_size
    assert final_state.frame_buffer.size(0) == batch_size
    assert final_state.frame_buffer.size(1) == 0
