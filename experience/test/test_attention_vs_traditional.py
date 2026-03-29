"""Test that symbolic st_attention aligns with traditional transformer attention behavior.

Compares the symbolic pipeline (slice_attention + merge) against the behavioral
properties of traditional scaled dot-product attention:
  output[b, i] = Aggregate(input[b, j] for j where mask[b, i, j])
"""
import os
import tempfile
import torch

from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_causal_attention_mask import (
    get_causal_attention_mask,
)
from experience.fs_util.text_merger import TextMerger


def read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def get_frames(tensor, flat_index):
    """Read and unpack frames at a flat index."""
    content = read_storage(tensor, flat_index)
    if content is None:
        return None
    return TextMerger.unpack(content)


def frame_texts(frames):
    """Extract just the text content from frames."""
    return [f[2] for f in frames]


if __name__ == "__main__":
    print("Running st_attention vs Traditional Transformer tests...\n")

    passed = 0
    failed = 0

    def run_test(name: str, condition: bool, expected=None, actual=None):
        global passed, failed
        if condition:
            print(f"  \u2713 {name}")
            passed += 1
        else:
            print(f"  \u2717 {name}")
            failed += 1
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # ================================================================
    # Test 1: Shape preservation — output.shape == input.shape
    # Traditional attention: output = softmax(QK^T/sqrt(d)) V has same shape as Q
    # Symbolic attention: (batch, seq) -> slice -> (batch, seq, seq) -> merge -> (batch, seq)
    # ================================================================
    print("Test 1: Shape preservation (residual-compatible)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c", "d"]], tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        output = st_attention(inp, mask)
        run_test("output.shape == input.shape", list(output.shape) == list(inp.shape))
        run_test("shape is (1, 4)", list(output.shape) == [1, 4])

    # ================================================================
    # Test 2: Causal (autoregressive) mask — token i sees exactly [0..i]
    # Like GPT: each token only attends to previous tokens + self
    # ================================================================
    print("Test 2: Causal mask — autoregressive attention")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["The", "cat", "sat", "down"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        output = st_attention(inp, mask)
        # Token 0 sees [The]
        f0 = get_frames(output, 0)
        run_test("token 0 sees 1 token", len(f0) == 1)
        run_test("token 0 sees [The]", frame_texts(f0) == ["The"])
        # Token 1 sees [The, cat]
        f1 = get_frames(output, 1)
        run_test("token 1 sees 2 tokens", len(f1) == 2)
        run_test("token 1 sees [The, cat]", frame_texts(f1) == ["The", "cat"])
        # Token 3 sees [The, cat, sat, down]
        f3 = get_frames(output, 3)
        run_test("token 3 sees all 4", len(f3) == 4)
        run_test("token 3 sees full sequence", frame_texts(f3) == tokens)

    # ================================================================
    # Test 3: Full (bidirectional) attention — every token sees all tokens
    # Like BERT: all-to-all attention
    # ================================================================
    print("Test 3: Full attention — bidirectional (BERT-style)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["I", "love", "NLP"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        output = st_attention(inp, mask)
        for i in range(3):
            fi = get_frames(output, i)
            run_test(f"token {i} sees all 3", len(fi) == 3)
            run_test(f"token {i} content = {tokens}", frame_texts(fi) == tokens)

    # ================================================================
    # Test 4: Padding mask — padded tokens produce zero, non-padded unaffected
    # Traditional: padding positions are masked out in attention
    # ================================================================
    print("Test 4: Padding mask integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["hello", "world", "<pad>"]], tmpdir)
        token_mask = torch.tensor([[True, True, False]])
        mask = get_causal_attention_mask(token_mask)
        output = st_attention(inp, mask)
        # Non-padded tokens work normally
        f0 = get_frames(output, 0)
        run_test("token 0 sees [hello]", frame_texts(f0) == ["hello"])
        f1 = get_frames(output, 1)
        run_test("token 1 sees [hello, world]", frame_texts(f1) == ["hello", "world"])
        # Padded token produces zero output
        run_test("padded token coeff = 0", output.data[0, 2].item() == 0.0)

    # ================================================================
    # Test 5: Sliding window attention — each token sees local window
    # Like Longformer local attention or sliding-window attention
    # ================================================================
    print("Test 5: Sliding window attention (window=2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["a", "b", "c", "d", "e"]
        inp = make_tensor([tokens], tmpdir)
        seq_len = 5
        # Window size 2: token i sees [max(0,i-1), i]
        mask = torch.zeros(1, seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            for j in range(max(0, i - 1), i + 1):
                mask[0, i, j] = True
        output = st_attention(inp, mask)
        # Token 0: sees [a] (no left neighbor)
        run_test("t0 sees [a]", frame_texts(get_frames(output, 0)) == ["a"])
        # Token 2: sees [b, c]
        run_test("t2 sees [b, c]", frame_texts(get_frames(output, 2)) == ["b", "c"])
        # Token 4: sees [d, e]
        run_test("t4 sees [d, e]", frame_texts(get_frames(output, 4)) == ["d", "e"])

    # ================================================================
    # Test 6: Diagonal (self-only) mask — each token attends only to itself
    # Like an identity attention pattern
    # ================================================================
    print("Test 6: Self-only attention (diagonal mask)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["x", "y", "z"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        output = st_attention(inp, mask)
        for i, t in enumerate(tokens):
            fi = get_frames(output, i)
            run_test(f"token {i} sees only [{t}]", len(fi) == 1 and fi[0][2] == t)

    # ================================================================
    # Test 7: CLS-token pattern — first token attends to all, rest attend to self
    # Like BERT [CLS] token for classification
    # ================================================================
    print("Test 7: CLS-token attention pattern")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["[CLS]", "hello", "world"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        mask[0, 0, :] = True  # CLS attends to all
        output = st_attention(inp, mask)
        f_cls = get_frames(output, 0)
        run_test("[CLS] sees all 3 tokens", len(f_cls) == 3)
        run_test("[CLS] content", frame_texts(f_cls) == tokens)
        f1 = get_frames(output, 1)
        run_test("token 1 sees only self", len(f1) == 1 and f1[0][2] == "hello")
        f2 = get_frames(output, 2)
        run_test("token 2 sees only self", len(f2) == 1 and f2[0][2] == "world")

    # ================================================================
    # Test 8: Cross-attention pattern — query tokens attend to separate key tokens
    # Like encoder-decoder: decoder attends to encoder positions
    # Simulated: tokens 2,3 attend to tokens 0,1 (not themselves)
    # ================================================================
    print("Test 8: Cross-attention pattern")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Positions 0,1 = encoder; positions 2,3 = decoder
        tokens = ["enc_a", "enc_b", "dec_x", "dec_y"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        # Decoder tokens attend to encoder tokens only
        mask[0, 2, 0] = True; mask[0, 2, 1] = True  # dec_x -> enc_a, enc_b
        mask[0, 3, 0] = True; mask[0, 3, 1] = True  # dec_y -> enc_a, enc_b
        output = st_attention(inp, mask)
        # Encoder tokens get no attention
        run_test("enc_a coeff = 0", output.data[0, 0].item() == 0.0)
        run_test("enc_b coeff = 0", output.data[0, 1].item() == 0.0)
        # Decoder tokens see encoder tokens
        f2 = get_frames(output, 2)
        run_test("dec_x sees [enc_a, enc_b]", frame_texts(f2) == ["enc_a", "enc_b"])
        f3 = get_frames(output, 3)
        run_test("dec_y sees [enc_a, enc_b]", frame_texts(f3) == ["enc_a", "enc_b"])

    # ================================================================
    # Test 9: No attention (all masked) — entire output is zero
    # Traditional: if no attention weights, output is zero
    # ================================================================
    print("Test 9: No attention — all masked")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"]], tmpdir)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        output = st_attention(inp, mask)
        run_test("all coefficients zero", (output.data == 0).all().item())
        for i in range(3):
            content = read_storage(output, i)
            run_test(f"token {i} no content", content is None)

    # ================================================================
    # Test 10: Batch independence — different masks yield independent results
    # Traditional: attention is computed per-sample in batch
    # ================================================================
    print("Test 10: Batch independence")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        # Batch 0: full attention; Batch 1: no attention
        mask[0] = True
        output = st_attention(inp, mask)
        # Batch 0 has content
        f_b0_t0 = get_frames(output, 0)
        run_test("batch 0 token 0 has frames", f_b0_t0 is not None and len(f_b0_t0) == 2)
        # Batch 1 all zero
        run_test("batch 1 all zero", (output.data[1] == 0).all().item())

    # ================================================================
    # Test 11: Frame ordering — attended tokens in positional order
    # Traditional: attention aggregates tokens in sequence order (via QK^T @ V)
    # ================================================================
    print("Test 11: Frame ordering matches positional order")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["first", "second", "third", "fourth"]
        inp = make_tensor([tokens], tmpdir)
        # Token 3 attends to [0, 2, 3] (skip 1)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        mask[0, 3, 0] = True
        mask[0, 3, 2] = True
        mask[0, 3, 3] = True
        output = st_attention(inp, mask)
        f3 = get_frames(output, 3)
        run_test("frame order = positional", frame_texts(f3) == ["first", "third", "fourth"])
        # Frame indices match original positions
        run_test("frame indices [0, 2, 3]", [f[0] for f in f3] == [0, 2, 3])

    # ================================================================
    # Test 12: Coefficient = number of attended tokens
    # Traditional: attention weights sum to 1 (softmax). Symbolic: coeff = count of attended
    # ================================================================
    print("Test 12: Coefficient = attended token count")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c", "d"]], tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        output = st_attention(inp, mask)
        for i in range(4):
            expected_coeff = float(i + 1)
            actual_coeff = output.data[0, i].item()
            run_test(
                f"token {i} coeff = {expected_coeff}",
                abs(actual_coeff - expected_coeff) < 1e-5,
                expected_coeff, actual_coeff,
            )

    # ================================================================
    # Test 13: Single token — degenerate, output = wrapped input
    # ================================================================
    print("Test 13: Single token degenerate case")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only"]], tmpdir)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        output = st_attention(inp, mask)
        run_test("shape (1, 1)", list(output.shape) == [1, 1])
        f0 = get_frames(output, 0)
        run_test("1 frame", len(f0) == 1)
        run_test("content = 'only'", f0[0][2] == "only")

    # ================================================================
    # Test 14: Content preservation — merge(slice(input)) keeps original text
    # Traditional: V values are preserved through attention (just reweighted)
    # ================================================================
    print("Test 14: Content preservation through attention pipeline")
    with tempfile.TemporaryDirectory() as tmpdir:
        original_texts = ["def foo():", "  return 42", "# comment"]
        inp = make_tensor([original_texts], tmpdir)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        output = st_attention(inp, mask)
        # Every output position contains all 3 original texts verbatim
        for i in range(3):
            fi = get_frames(output, i)
            texts = frame_texts(fi)
            for j, original in enumerate(original_texts):
                run_test(
                    f"out[{i}] preserves '{original[:15]}...'",
                    texts[j] == original,
                    original, texts[j] if j < len(texts) else "MISSING",
                )

    # ================================================================
    # Test 15: Monotonic information growth (causal)
    # Later positions in causal mask carry strictly more frames
    # ================================================================
    print("Test 15: Monotonic frame count growth (causal)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c", "d", "e"]], tmpdir)
        mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))
        output = st_attention(inp, mask)
        prev_count = 0
        all_monotonic = True
        for i in range(5):
            fi = get_frames(output, i)
            if len(fi) <= prev_count:
                all_monotonic = False
            prev_count = len(fi)
        run_test("frame counts strictly increasing", all_monotonic)

    # ================================================================
    # Test 16: Attention sparsity — sparse mask = fewer frames
    # Traditional: sparse attention (Longformer, BigBird) reduces computation
    # ================================================================
    print("Test 16: Sparse vs dense frame count")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c", "d"]], tmpdir)
        # Dense: full attention
        dense_mask = torch.ones(1, 4, 4, dtype=torch.bool)
        dense_out = st_attention(inp, dense_mask)
        # Sparse: diagonal only
        sparse_mask = torch.eye(4, dtype=torch.bool).unsqueeze(0)
        sparse_out = st_attention(inp, sparse_mask)
        for i in range(4):
            dense_frames = get_frames(dense_out, i)
            sparse_frames = get_frames(sparse_out, i)
            run_test(
                f"token {i}: sparse ({len(sparse_frames)}) < dense ({len(dense_frames)})",
                len(sparse_frames) < len(dense_frames),
            )

    # ================================================================
    # Test 17: Symmetric mask — symmetric attention produces same frames count
    # If mask[i, j] == mask[j, i], token i and j see same number of tokens
    # ================================================================
    print("Test 17: Symmetric mask symmetry")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["a", "b", "c"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)  # Symmetric full mask
        output = st_attention(inp, mask)
        f0 = get_frames(output, 0)
        f1 = get_frames(output, 1)
        f2 = get_frames(output, 2)
        run_test("all positions same frame count", len(f0) == len(f1) == len(f2))
        # All see the same content (full attention)
        run_test("all see same texts", frame_texts(f0) == frame_texts(f1) == frame_texts(f2))

    # ================================================================
    # Test 18: Multi-head simulation — independent masks per batch = independent heads
    # Traditional: multi-head attention runs h independent attention operations
    # ================================================================
    print("Test 18: Multi-head simulation via batch dimension")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Simulate 2 heads: head 0 = causal, head 1 = reverse causal
        inp = make_tensor([["a", "b", "c"], ["a", "b", "c"]], tmpdir)
        mask = torch.zeros(2, 3, 3, dtype=torch.bool)
        # Head 0: causal (lower triangular)
        mask[0] = torch.tril(torch.ones(3, 3, dtype=torch.bool))
        # Head 1: reverse causal (upper triangular)
        mask[1] = torch.triu(torch.ones(3, 3, dtype=torch.bool))
        output = st_attention(inp, mask)
        # Head 0, token 0: sees [a]
        h0_t0 = get_frames(output, 0)
        run_test("head0 t0 sees [a]", frame_texts(h0_t0) == ["a"])
        # Head 0, token 2: sees [a, b, c]
        h0_t2 = get_frames(output, 2)
        run_test("head0 t2 sees [a,b,c]", frame_texts(h0_t2) == ["a", "b", "c"])
        # Head 1, token 0: sees [a, b, c]
        h1_t0 = get_frames(output, 3)
        run_test("head1 t0 sees [a,b,c]", frame_texts(h1_t0) == ["a", "b", "c"])
        # Head 1, token 2: sees [c]
        h1_t2 = get_frames(output, 5)
        run_test("head1 t2 sees [c]", frame_texts(h1_t2) == ["c"])

    # ================================================================
    # Test 19: Prefix sharing (causal) — tokens sharing prefix have shared leading frames
    # Traditional: in causal attention, token i's attention is a prefix of token i+1's
    # ================================================================
    print("Test 19: Prefix sharing in causal attention")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["w", "x", "y", "z"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        output = st_attention(inp, mask)
        # Token i's frames should be a prefix of token i+1's frames
        for i in range(3):
            fi = get_frames(output, i)
            fi_next = get_frames(output, i + 1)
            prefix_match = frame_texts(fi) == frame_texts(fi_next)[:len(fi)]
            run_test(f"t{i} frames are prefix of t{i+1}", prefix_match)

    # ================================================================
    # Test 20: get_causal_attention_mask integration — full pipeline
    # ================================================================
    print("Test 20: get_causal_attention_mask integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["hello", "beautiful", "world", "<pad>", "<pad>"]
        inp = make_tensor([tokens], tmpdir)
        token_mask = torch.tensor([[True, True, True, False, False]])
        mask = get_causal_attention_mask(token_mask)
        output = st_attention(inp, mask)
        run_test("shape (1, 5)", list(output.shape) == [1, 5])
        # Active tokens
        f0 = get_frames(output, 0)
        run_test("t0 = [hello]", frame_texts(f0) == ["hello"])
        f2 = get_frames(output, 2)
        run_test("t2 = [hello, beautiful, world]",
                 frame_texts(f2) == ["hello", "beautiful", "world"])
        # Padded tokens are zero
        run_test("pad t3 = 0", output.data[0, 3].item() == 0.0)
        run_test("pad t4 = 0", output.data[0, 4].item() == 0.0)

    # ================================================================
    # Test 21: Larger batch with mixed padding
    # Traditional: batched attention handles variable-length sequences via padding
    # ================================================================
    print("Test 21: Batched mixed-length sequences")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Batch 0: 3 real tokens + 1 pad; Batch 1: 2 real + 2 pad
        inp = make_tensor([
            ["a", "b", "c", "<p>"],
            ["x", "y", "<p>", "<p>"],
        ], tmpdir)
        token_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])
        mask = get_causal_attention_mask(token_mask)
        output = st_attention(inp, mask)
        run_test("shape (2, 4)", list(output.shape) == [2, 4])
        # Batch 0
        f_b0_t2 = get_frames(output, 2)
        run_test("b0 t2 = [a, b, c]", frame_texts(f_b0_t2) == ["a", "b", "c"])
        run_test("b0 t3 (pad) = 0", output.data[0, 3].item() == 0.0)
        # Batch 1
        f_b1_t1 = get_frames(output, 5)
        run_test("b1 t1 = [x, y]", frame_texts(f_b1_t1) == ["x", "y"])
        run_test("b1 t2 (pad) = 0", output.data[1, 2].item() == 0.0)
        run_test("b1 t3 (pad) = 0", output.data[1, 3].item() == 0.0)

    # ================================================================
    # Test 22: Attention with realistic NLP content
    # Verify the pipeline works with multi-word, multi-line text elements
    # ================================================================
    print("Test 22: Realistic NLP text content")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = [
            "The quick brown fox",
            "jumps over\nthe lazy dog",
            "and runs away",
        ]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        output = st_attention(inp, mask)
        f2 = get_frames(output, 2)
        run_test("3 frames for last token", len(f2) == 3)
        run_test("multiline preserved", f2[1][2] == "jumps over\nthe lazy dog")
        run_test("all texts exact", frame_texts(f2) == tokens)

    # ================================================================
    # Test 23: Skip-attention pattern — attend to every other token
    # Like strided/dilated attention in some efficient transformers
    # ================================================================
    print("Test 23: Strided attention (every other token)")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["a", "b", "c", "d", "e", "f"]
        inp = make_tensor([tokens], tmpdir)
        seq_len = 6
        mask = torch.zeros(1, seq_len, seq_len, dtype=torch.bool)
        # Each token attends to every other token (stride 2, starting from self's parity)
        for i in range(seq_len):
            for j in range(i % 2, seq_len, 2):
                mask[0, i, j] = True
        output = st_attention(inp, mask)
        # Token 0 (even): sees [a, c, e]
        f0 = get_frames(output, 0)
        run_test("t0 sees [a,c,e]", frame_texts(f0) == ["a", "c", "e"])
        # Token 1 (odd): sees [b, d, f]
        f1 = get_frames(output, 1)
        run_test("t1 sees [b,d,f]", frame_texts(f1) == ["b", "d", "f"])

    # ================================================================
    # Summary
    # ================================================================
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    print(f"{'='*60}")
