import torch

from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor


def slice_attention_forward(
    input: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter input elements into a 3D output according to attention mask.

    For each active token (b, i), copies the attended input files at positions j
    (where attention_mask[b, i, j] is True) into output[b, i, j].

    Args:
        input: Symbolic tensor of shape (batch, seq_len).
        attention_mask: Bool tensor of shape (batch, seq_len, seq_len).

    Returns:
        Symbolic tensor of shape (batch, seq_len, seq_len).
    """
    batch, seq_len = input.shape

    # 3D output: (batch, seq_len, seq_len)
    final_output = make_none_tensor([batch, seq_len, seq_len], input.st_relative_to)

    # Which tokens are active (attend to at least one position)
    token_mask = attention_mask.any(dim=-1)  # (batch, seq_len)
    valid_token_points = list(torch.nonzero(token_mask, as_tuple=True))

    if valid_token_points[0].numel() == 0:
        return final_output

    for batch_i, token_j in zip(
        valid_token_points[0].tolist(), valid_token_points[1].tolist()
    ):
        # Attended prefix positions for this token
        prefix_indices = torch.nonzero(
            attention_mask[batch_i, token_j, :], as_tuple=True
        )[0]

        # 1D output view: final_output[batch_i, token_j, prefix_indices]
        token_output_view = slice_view(final_output, [batch_i, token_j, prefix_indices])

        # 1D input view: input[batch_i, prefix_indices]
        token_input_view = slice_view(input, [batch_i, prefix_indices])

        # Copy input storage files into output positions
        assign_tensor(token_output_view, token_input_view)

    # Mark all attended positions with 1.0
    final_output[attention_mask] = 1.0

    return final_output


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running slice_attention_forward tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_out(tensor, flat_index):
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

    # Test 1: Causal mask on 1x3 — lower triangular scatter
    print("Test 1: Causal 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"]], tmpdir)  # (1, 3)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))

        result = slice_attention_forward(inp, mask)
        run_test("shape (1, 3, 3)", list(result.shape) == [1, 3, 3])
        # Row 0: attends to [0] → output[0,0,0]='a'
        run_test("[0,0,0]='a'", read_out(result, 0) == "a")
        # Row 1: attends to [0,1] → output[0,1,0]='a', output[0,1,1]='b'
        run_test("[0,1,0]='a'", read_out(result, 3) == "a")
        run_test("[0,1,1]='b'", read_out(result, 4) == "b")
        # Row 2: attends to [0,1,2]
        run_test("[0,2,0]='a'", read_out(result, 6) == "a")
        run_test("[0,2,1]='b'", read_out(result, 7) == "b")
        run_test("[0,2,2]='c'", read_out(result, 8) == "c")
        # Non-attended positions are zero
        run_test("[0,0,1] is zero", result[0, 0, 1].item() == 0.0)
        run_test("[0,0,2] is zero", result[0, 0, 2].item() == 0.0)

    # Test 2: Partial mask — only specific positions attended
    print("Test 2: Partial mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        mask[0, 2, 0] = True
        mask[0, 2, 2] = True

        result = slice_attention_forward(inp, mask)
        run_test("[0,0,0]='x'", read_out(result, 0) == "x")
        run_test("[0,2,0]='x'", read_out(result, 6) == "x")
        run_test("[0,2,2]='z'", read_out(result, 8) == "z")
        # Row 1 inactive
        run_test("row 1 all zero", (result[0, 1] == 0).all().item())

    # Test 3: Multi-batch 2x2
    print("Test 3: Multi-batch 2x2")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["p", "q"], ["r", "s"]], tmpdir)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True  # batch 0, token 1 attends to token 0
        mask[0, 1, 1] = True  # batch 0, token 1 attends to self
        mask[1, 0, 0] = True  # batch 1, token 0 attends to self

        result = slice_attention_forward(inp, mask)
        run_test("shape (2, 2, 2)", list(result.shape) == [2, 2, 2])
        # batch 0, token 1: [p, q]
        run_test("b0[1,0]='p'", read_out(result, 2) == "p")
        run_test("b0[1,1]='q'", read_out(result, 3) == "q")
        # batch 1, token 0: [r]
        run_test("b1[0,0]='r'", read_out(result, 4) == "r")
        # batch 0, token 0 inactive
        run_test("b0 row 0 zero", (result[0, 0] == 0).all().item())

    # Test 4: Empty mask — no active tokens
    print("Test 4: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        result = slice_attention_forward(inp, mask)
        run_test("all zero", (result == 0).all().item())

    # Test 5: Single token
    print("Test 5: Single token")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only"]], tmpdir)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        result = slice_attention_forward(inp, mask)
        run_test("shape (1, 1, 1)", list(result.shape) == [1, 1, 1])
        run_test("[0,0,0]='only'", read_out(result, 0) == "only")

    # Test 6: Integration with causal mask generator
    print("Test 6: With causal mask + padding")
    with tempfile.TemporaryDirectory() as tmpdir:
        from experience.symbolic_tensor.function.get_causal_attention_mask import (
            get_causal_attention_mask,
        )

        inp = make_tensor([["hello", "world", "!"]], tmpdir)
        token_mask = torch.tensor([[True, True, False]])
        causal = get_causal_attention_mask(token_mask)

        result = slice_attention_forward(inp, causal)
        run_test("[0,0,0]='hello'", read_out(result, 0) == "hello")
        run_test("[0,1,0]='hello'", read_out(result, 3) == "hello")
        run_test("[0,1,1]='world'", read_out(result, 4) == "world")
        # Padded token row is all zero
        run_test("row 2 all zero", (result[0, 2] == 0).all().item())

    # Test 7: Values marked correctly
    print("Test 7: Tensor values")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.tensor([[[True, False], [True, True]]])

        result = slice_attention_forward(inp, mask)
        run_test("attended positions are 1.0", result[mask].eq(1.0).all().item())
        run_test("non-attended positions are 0.0", result[~mask].eq(0.0).all().item())

    print("\nAll tests completed.")
