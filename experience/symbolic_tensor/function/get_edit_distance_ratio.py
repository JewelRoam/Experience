import os
import subprocess
import tempfile

import Levenshtein
import torch
from torch.autograd import Function

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


def _read_storage(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the text content at a given flat storage index."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    with open(path) as f:
        return f.read()


def _get_diff(actual_content: str, expected_content: str) -> str:
    """Run diff -u on two content strings via temp files. Returns unified diff output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".expected", delete=False) as f_exp, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".actual", delete=False) as f_act:
        f_exp.write(expected_content)
        f_exp.flush()
        f_act.write(actual_content)
        f_act.flush()

        result = subprocess.run(
            ["diff", "-u", f_exp.name, f_act.name],
            capture_output=True,
            text=True,
        )
        diff_output = result.stdout

        os.unlink(f_exp.name)
        os.unlink(f_act.name)

    return diff_output


def get_edit_distance_ratio_impl(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """
    Forward: for each flat element, compute edit distance ratio.
    edit_distance_ratio = edit_distance(actual, expected) / (max(len(actual), len(expected)) + epsilon)
    Returns a bfloat16 tensor with the same shape as actual.
    """
    epsilon = 1e-8
    numel = actual.numel()
    ratios = []

    for i in range(numel):
        actual_text = _read_storage(actual, i)
        expected_text = _read_storage(expected, i)

        dist = Levenshtein.distance(actual_text, expected_text)
        max_len = max(len(actual_text), len(expected_text))
        ratio = dist / (max_len + epsilon)

        ratios.append(ratio)

    return torch.tensor(ratios, dtype=torch.bfloat16).reshape(actual.shape)


def get_edit_distance_ratio_backward_impl(
    grad_output: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """
    Backward: compute diff text for each element, store as symbolic gradient.
    The grad coefficient channel carries grad_output values.
    Returns a symbolic tensor of diff texts.
    """
    numel = actual.numel()
    diff_texts = []

    for i in range(numel):
        actual_text = _read_storage(actual, i)
        expected_text = _read_storage(expected, i)
        diff_output = _get_diff(actual_text, expected_text)
        diff_texts.append(diff_output)

    # Unflatten diff_texts into nested list matching actual.shape
    def _unflatten(flat, shape):
        if len(shape) == 0:
            return flat[0]
        if len(shape) == 1:
            return flat[:shape[0]]
        chunk = 1
        for s in shape[1:]:
            chunk *= s
        return [_unflatten(flat[i * chunk:(i + 1) * chunk], shape[1:]) for i in range(shape[0])]

    nested = _unflatten(diff_texts, list(actual.shape))
    actual_grad = make_tensor(nested, actual.st_relative_to)
    # Set coefficient channel from grad_output
    actual_grad.data = grad_output.broadcast_to(actual.shape).clone().to(actual_grad.dtype)
    return actual_grad


from experience.symbolic_tensor.function import symbolic_grad_registry


class GetEditDistanceRatio(Function):
    @staticmethod
    def forward(ctx, actual, expected):
        ctx.save_for_backward(actual, expected)
        # save_for_backward strips custom attributes; preserve them manually
        ctx.st_attrs = {}
        for name, tensor in [("actual", actual), ("expected", expected)]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        return get_edit_distance_ratio_impl(actual, expected)

    @staticmethod
    def backward(ctx, grad_output):
        actual, expected = ctx.saved_tensors
        # Restore custom st_* attributes stripped by save_for_backward
        for name, tensor in [("actual", actual), ("expected", expected)]:
            for attr, val in ctx.st_attrs[name].items():
                setattr(tensor, attr, val)
        actual_grad = get_edit_distance_ratio_backward_impl(grad_output, actual, expected)
        # Register symbolic grad keyed by the forward tensor's uid,
        # so downstream backward (SymbolicTransform) can retrieve it
        symbolic_grad_registry.register(actual.st_tensor_uid, actual_grad)
        return actual_grad, None


get_edit_distance_ratio = GetEditDistanceRatio.apply


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running get_edit_distance_ratio tests...\n")

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
        with open(path) as f:
            return f.read()

    # Test 1: Forward - identical texts give ratio 0.0
    print("Test 1: Forward - identical texts")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["hello world"], tmpdir)
        expected_t = make_tensor(["hello world"], tmpdir)

        out = get_edit_distance_ratio_impl(actual_t, expected_t)
        run_test("Shape matches", list(out.shape) == [1])
        run_test("dtype bfloat16", out.dtype == torch.bfloat16)
        run_test("Identical => ratio 0.0", out[0].item() == 0.0, 0.0, out[0].item())

    # Test 2: Forward - different texts give ratio > 0
    print("Test 2: Forward - different texts")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["hello world"], tmpdir)
        expected_t = make_tensor(["hello earth"], tmpdir)

        out = get_edit_distance_ratio_impl(actual_t, expected_t)
        run_test("Ratio > 0.0", out[0].item() > 0.0)
        run_test("Ratio <= 1.0", out[0].item() <= 1.0)
        print(f"    ratio = {out[0].item():.4f}")

    # Test 3: Forward - completely different strings
    print("Test 3: Forward - completely different")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["abc"], tmpdir)
        expected_t = make_tensor(["xyz"], tmpdir)

        out = get_edit_distance_ratio_impl(actual_t, expected_t)
        run_test("Ratio == 1.0", out[0].item() == 1.0, 1.0, out[0].item())

    # Test 4: Forward - 2D batch
    print("Test 4: Forward - 2D batch")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor([["same", "differs"]], tmpdir)
        expected_t = make_tensor([["same", "original"]], tmpdir)

        out = get_edit_distance_ratio_impl(actual_t, expected_t)
        run_test("Shape [1, 2]", list(out.shape) == [1, 2])
        run_test("Identical element => 0.0", out[0, 0].item() == 0.0)
        run_test("Different element => > 0.0", out[0, 1].item() > 0.0)

    # Test 5: Backward produces symbolic gradient
    print("Test 5: Backward - symbolic gradient")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["hello world"], tmpdir)
        expected_t = make_tensor(["hello earth"], tmpdir)
        grad_out = torch.tensor([1.0], dtype=torch.bfloat16)

        actual_grad = get_edit_distance_ratio_backward_impl(grad_out, actual_t, expected_t)
        run_test("Grad has st_tensor_uid", hasattr(actual_grad, "st_tensor_uid"))
        run_test("Grad has st_relative_to", hasattr(actual_grad, "st_relative_to"))
        run_test("Grad shape matches actual", list(actual_grad.shape) == list(actual_t.shape))
        diff_text = read_storage(actual_grad, 0)
        run_test("Diff text non-empty", len(diff_text) > 0)
        print(f"    diff (first 80): {repr(diff_text[:80])}")

    # Test 6: Backward - identical texts produce empty diff
    print("Test 6: Backward - identical texts => empty diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["same content"], tmpdir)
        expected_t = make_tensor(["same content"], tmpdir)
        grad_out = torch.tensor([1.0], dtype=torch.bfloat16)

        actual_grad = get_edit_distance_ratio_backward_impl(grad_out, actual_t, expected_t)
        diff_text = read_storage(actual_grad, 0)
        run_test("Identical => empty diff", diff_text == "")

    # Test 7: GetEditDistanceRatio autograd Function
    print("Test 7: GetEditDistanceRatio autograd Function")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["foo bar"], tmpdir)
        expected_t = make_tensor(["foo baz"], tmpdir)

        ctx = type("MockCtx", (), {})()
        ctx.save_for_backward = lambda *ts: setattr(ctx, "saved_tensors", ts)
        fwd = GetEditDistanceRatio.forward(ctx, actual_t, expected_t)
        run_test("Forward returns bfloat16", fwd.dtype == torch.bfloat16)

        grad_out = torch.tensor([0.5], dtype=torch.bfloat16)
        result = GetEditDistanceRatio.backward(ctx, grad_out)
        run_test("Returns tuple of 2", len(result) == 2)
        run_test("Second is None", result[1] is None)
        run_test("First has st_tensor_uid", hasattr(result[0], "st_tensor_uid"))

    print("\nAll tests completed.")
