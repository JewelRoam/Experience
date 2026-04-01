import os
import torch

from experience.symbolic_tensor.tensor_util.slice_tensor import (
    slice_tensor as _raw_slice_tensor,
)
from experience.symbolic_tensor.tensor_util.slice_view import _normalize_slice_element
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function.slice_view import (
    _save_st_attrs,
    _restore_st_attrs,
    _resolve_grad_output,
    _build_per_dim,
    slice_backward,
)
from experience.symbolic_tensor.function import symbolic_grad_registry


class SliceTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slice_list):
        per_dim = _build_per_dim(slice_list, input.size())
        output = _raw_slice_tensor(input, slice_list)

        ctx.save_for_backward(input, output)
        _save_st_attrs(ctx, input=input, output=output)
        ctx.per_dim = per_dim

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        _restore_st_attrs(ctx, input=input, output=output)

        grad_output = _resolve_grad_output(output, grad_output)
        grad_input = slice_backward(grad_output, input, output, ctx.per_dim)

        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)

        return grad_input, None


def slice_tensor(input: torch.Tensor, slice_list: list) -> torch.Tensor:
    """Autograd-aware slice_tensor. Falls back to raw when no grad needed."""
    if not input.requires_grad:
        return _raw_slice_tensor(input, slice_list)
    return SliceTensor.apply(input, slice_list)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
    from experience.symbolic_tensor.function.slice_view import _build_per_dim

    print("Running function/slice_tensor tests...\n")

    def run_test(name, cond, expected=None, actual=None):
        if cond:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None:
                print(f"    expected: {expected}, actual: {actual}")

    def read_out(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: No grad — raw passthrough (copy, not symlink)
    print("Test 1: No grad passthrough")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        r = slice_tensor(t, [0, slice(None)])
        run_test("shape [2]", list(r.shape) == [2])
        run_test("[0]=a", read_out(r, 0) == "a")
        run_test("NOT symlink", not os.path.islink(
            os.path.join(tmpdir, r.st_tensor_uid, "storage", "0", "data")))

    # Test 2: With grad — forward (independent copy)
    print("Test 2: Autograd forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a\n", "b\n"], ["c\n", "d\n"]], tmpdir)
        t.requires_grad_(True)
        r = slice_tensor(t, [1, slice(None)])
        run_test("shape [2]", list(r.shape) == [2])
        run_test("has st_relative_to", hasattr(r, "st_relative_to"))
        run_test("[0]=c", read_out(r, 0) == "c\n")
        run_test("[1]=d", read_out(r, 1) == "d\n")
        # Verify independent copy
        copy_path = os.path.join(tmpdir, r.st_tensor_uid, "storage", "0", "data")
        run_test("NOT symlink", not os.path.islink(copy_path))

    # Test 3: Backward scatter
    print("Test 3: Backward scatter")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["alpha\n", "beta\n", "gamma\n"], tmpdir)
        t.requires_grad_(True)
        # Select indices 0 and 2
        output = SliceTensor.apply(t, [torch.tensor([0, 2])])
        run_test("output shape [2]", list(output.shape) == [2])

        # Modify alpha → ALPHA
        mod_data = [read_out(output, 0).replace("alpha", "ALPHA"), read_out(output, 1)]
        mod_output = make_tensor(mod_data, tmpdir)
        grad_out = get_diff_tensor(output, mod_output)
        per_dim = _build_per_dim([torch.tensor([0, 2])], t.size())

        grad_input = slice_backward(grad_out, t, output, per_dim)
        run_test("grad_input not None", grad_input is not None)
        run_test("grad_input shape [3]", list(grad_input.shape) == [3])
        gi_0 = read_out(grad_input, 0)
        run_test("grad[0] has diff", gi_0 is not None and "---" in gi_0)
        run_test("grad[0] has +ALPHA", "+ALPHA" in gi_0 if gi_0 else False)
        gi_1 = read_out(grad_input, 1)
        run_test("grad[1] is TODO", gi_1 is not None and "TODO" in gi_1)
        gi_2 = read_out(grad_input, 2)
        run_test("grad[2] empty diff", gi_2 is not None and "---" not in gi_2)

    # Test 4: No grad when not required
    print("Test 4: No grad when not required")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["a\n", "b\n"], tmpdir)
        output = _raw_slice_tensor(t, [torch.tensor([0])])
        per_dim = _build_per_dim([torch.tensor([0])], t.size())
        grad_out = make_tensor([""], tmpdir)
        result = slice_backward(grad_out, t, output, per_dim)
        run_test("returns None", result is None)

    print("\nAll tests completed.")
