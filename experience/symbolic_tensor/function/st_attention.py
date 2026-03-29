import os
import tempfile
import torch
from typing import Callable, Dict, Optional, Type

from experience.symbolic_tensor.function.slice_attention import slice_attention
from experience.symbolic_tensor.function.merge import merge
from experience.fs_util.text_merger import TextMerger


def st_attention(
    input: torch.Tensor,
    attention_mask: torch.Tensor,
    return_view: bool = False,
    grad_input_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
    text_merger: Optional[Type] = None,
) -> torch.Tensor:
    """Symbolic attention: slice_attention + merge last dim.

    Pipeline:
      input (batch, seq_len) + attention_mask (batch, seq_len, seq_len)
        -> slice_attention -> (batch, seq_len, seq_len)
        -> merge(axis=-1)  -> (batch, seq_len)

    Each output token is the merged representation of all tokens it attends to.

    Args:
        input: Symbolic tensor of shape (batch, seq_len).
        attention_mask: Bool tensor of shape (batch, seq_len, seq_len).
        return_view: If True, use symlinks (views) instead of copies in slice_attention.
        grad_input_prompt: Custom prompt callable for slice_attention backward.
        task_prompt: High-level task description.
        llm_method: LLM backend to use.
        llm_env: Environment variable dict for LLM client.
        text_merger: Class with pack/unpack for merge. Defaults to TextMerger.

    Returns:
        Symbolic tensor of shape (batch, seq_len).
    """
    # slice_attention: (batch, seq_len) -> (batch, seq_len, seq_len)
    sliced = slice_attention(
        input, attention_mask,
        return_view, grad_input_prompt,
        task_prompt, llm_method, llm_env,
    )
    # merge last dim: (batch, seq_len, seq_len) -> (batch, seq_len)
    return merge(sliced, -1, text_merger)


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.fs_util.text_merger import kFrameMarker

    print("Running st_attention tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

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

    # Test 1: Basic causal 1x3
    print("Test 1: Causal 1x3 forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))

        output = st_attention(inp, mask)
        run_test("Output shape (1, 3)", list(output.shape) == [1, 3])
        run_test("Output has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("Output has st_tensor_uid", hasattr(output, "st_tensor_uid"))
        # output[0, 0] = merge of [a]
        content_0 = read_storage(output, 0)
        run_test("output[0,0] not None", content_0 is not None)
        frames_0 = TextMerger.unpack(content_0)
        run_test("output[0,0] 1 frame", len(frames_0) == 1)
        run_test("output[0,0] frame = 'a'", frames_0[0][2] == "a")
        # output[0, 1] = merge of [a, b]
        content_1 = read_storage(output, 1)
        frames_1 = TextMerger.unpack(content_1)
        run_test("output[0,1] 2 frames", len(frames_1) == 2)
        run_test("output[0,1] frame[0] = 'a'", frames_1[0][2] == "a")
        run_test("output[0,1] frame[1] = 'b'", frames_1[1][2] == "b")
        # output[0, 2] = merge of [a, b, c]
        content_2 = read_storage(output, 2)
        frames_2 = TextMerger.unpack(content_2)
        run_test("output[0,2] 3 frames", len(frames_2) == 3)
        run_test("output[0,2] frame[0] = 'a'", frames_2[0][2] == "a")
        run_test("output[0,2] frame[2] = 'c'", frames_2[2][2] == "c")

    # Test 2: Partial mask
    print("Test 2: Partial mask 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 2, 0] = True
        mask[0, 2, 2] = True

        output = st_attention(inp, mask)
        run_test("Output shape (1, 3)", list(output.shape) == [1, 3])
        frames_0 = TextMerger.unpack(read_storage(output, 0))
        run_test("output[0,0] 1 frame = 'x'", len(frames_0) == 1 and frames_0[0][2] == "x")
        run_test("output[0,1] coeff zero", output.data[0, 1].item() == 0.0)
        frames_2 = TextMerger.unpack(read_storage(output, 2))
        run_test("output[0,2] 2 frames", len(frames_2) == 2)
        run_test("output[0,2] frame[0] = 'x'", frames_2[0][2] == "x")
        run_test("output[0,2] frame[1] = 'z'", frames_2[1][2] == "z")

    # Test 3: Multi-batch 2x2
    print("Test 3: Multi-batch 2x2")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["p", "q"], ["r", "s"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True
        mask[0, 1, 1] = True
        mask[1, 0, 0] = True

        output = st_attention(inp, mask)
        run_test("Output shape (2, 2)", list(output.shape) == [2, 2])
        frames_b0t1 = TextMerger.unpack(read_storage(output, 1))
        run_test("b0[1] 2 frames", len(frames_b0t1) == 2)
        run_test("b0[1] frame[0] = 'p'", frames_b0t1[0][2] == "p")
        run_test("b0[1] frame[1] = 'q'", frames_b0t1[1][2] == "q")
        frames_b1t0 = TextMerger.unpack(read_storage(output, 2))
        run_test("b1[0] 1 frame = 'r'", len(frames_b1t0) == 1 and frames_b1t0[0][2] == "r")

    # Test 4: Empty mask
    print("Test 4: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        output = st_attention(inp, mask)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        run_test("all zero", (output == 0).all().item())

    # Test 5: Full attention (bidirectional)
    print("Test 5: Full attention")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"]], tmpdir)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)

        output = st_attention(inp, mask)
        for i in range(3):
            fi = TextMerger.unpack(read_storage(output, i))
            run_test(f"token {i} sees all 3", len(fi) == 3)

    # Test 6: Custom text_merger
    print("Test 6: Custom text_merger")

    class SimpleMerger:
        @staticmethod
        def pack(frames):
            return " | ".join(f"{idx}:{text}" for idx, _, text in frames)

        @staticmethod
        def unpack(merged):
            parts = merged.split(" | ")
            return [(int(p.split(":", 1)[0]), 1.0, p.split(":", 1)[1]) for p in parts]

    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        output = st_attention(inp, mask, text_merger=SimpleMerger)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        content_1 = read_storage(output, 1)
        run_test("custom format", " | " in content_1)

    # Test 7: return_view=True
    print("Test 7: return_view=True")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha", "beta"]], tmpdir)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = st_attention(inp, mask, return_view=True)
        run_test("Output shape (1, 2)", list(output.shape) == [1, 2])
        frames_1 = TextMerger.unpack(read_storage(output, 1))
        run_test("output[0,1] 2 frames", len(frames_1) == 2)
        run_test("frame[0] = 'alpha'", frames_1[0][2] == "alpha")
        run_test("frame[1] = 'beta'", frames_1[1][2] == "beta")

    # Test 8: Single token
    print("Test 8: Single token")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only"]], tmpdir)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        output = st_attention(inp, mask)
        run_test("shape (1, 1)", list(output.shape) == [1, 1])
        f0 = TextMerger.unpack(read_storage(output, 0))
        run_test("1 frame = 'only'", len(f0) == 1 and f0[0][2] == "only")

    # Test 9: Coefficient = attended count
    print("Test 9: Coefficient = attended count")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c", "d"]], tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))
        output = st_attention(inp, mask)
        for i in range(4):
            expected = float(i + 1)
            actual = output.data[0, i].item()
            run_test(f"token {i} coeff = {expected}", abs(actual - expected) < 1e-5,
                     expected, actual)

    # Test 10: Default parameters
    print("Test 10: Default parameters work")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        # Call with only required args
        output = st_attention(inp, mask)
        run_test("works with defaults", list(output.shape) == [1, 2])

    print("\nAll tests completed.")
