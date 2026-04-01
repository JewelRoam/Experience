import os
import itertools
import torch
from typing import List, Optional

from experience.symbolic_tensor.tensor_util.slice_view import (
    slice_view as _raw_slice_view,
    _normalize_slice_element,
)
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.st_patched import st_patched
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view as _scalar_view
from experience.symbolic_tensor.function import symbolic_grad_registry


def _save_st_attrs(ctx, **named_tensors):
    ctx.st_attrs = {}
    for name, tensor in named_tensors.items():
        attrs = {}
        for attr in ("st_relative_to", "st_tensor_uid"):
            if hasattr(tensor, attr):
                attrs[attr] = getattr(tensor, attr)
        ctx.st_attrs[name] = attrs


def _restore_st_attrs(ctx, **named_tensors):
    for name, tensor in named_tensors.items():
        for attr, val in ctx.st_attrs[name].items():
            setattr(tensor, attr, val)


def _resolve_grad_output(output, grad_output):
    symbolic_grad = symbolic_grad_registry.pop(output.st_tensor_uid)
    if symbolic_grad is not None:
        return symbolic_grad
    if not hasattr(grad_output, "st_relative_to"):
        wrapped = todo_tensor_like(output)
        wrapped.data.copy_(grad_output.data)
        return wrapped
    return grad_output


def _coords_to_flat(coords, strides):
    return sum(c * s for c, s in zip(coords, strides))


def _build_per_dim(slice_list, input_size):
    """Normalize each slice element and return (indices, collapses) per dim."""
    per_dim = []
    for d, elem in enumerate(slice_list):
        indices, collapses = _normalize_slice_element(elem, input_size[d])
        per_dim.append((indices, collapses))
    return per_dim


def slice_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    per_dim: list,
) -> Optional[torch.Tensor]:
    """Backward for slice_view / slice_tensor: scatter grads back to input positions.

    For each output position, maps it back to the corresponding input position
    via per_dim, computes the diff, and assigns to grad_input.
    """
    if not input.requires_grad:
        return None

    better_version_output = st_patched(output, grad_output)

    grad_input = todo_tensor_like(input)
    grad_input.data.zero_()

    output_shape = list(output.shape)
    output_ranges = [range(s) for s in output_shape] if output_shape else []
    output_strides = output.stride() if output_shape else ()
    input_strides = input.stride()

    for out_coords in itertools.product(*output_ranges) if output_ranges else [()]:
        out_coords = list(out_coords)

        # Map output coords back to input coords
        in_coords = []
        out_dim = 0
        for d, (indices, collapses) in enumerate(per_dim):
            if collapses:
                in_coords.append(indices[0])
            else:
                in_coords.append(indices[out_coords[out_dim]])
                out_dim += 1

        # Diff at this output position
        orig_view = _scalar_view(output, out_coords)
        improved_view = _scalar_view(better_version_output, out_coords)
        diff = get_diff_tensor(orig_view, improved_view)

        # Assign to grad_input at the input position
        grad_view = _scalar_view(grad_input, in_coords)
        assign_tensor(grad_view, diff)

        # Copy improved coefficient
        out_flat = _coords_to_flat(out_coords, output_strides) if out_coords else 0
        in_flat = _coords_to_flat(in_coords, input_strides)
        improved_coeff = better_version_output.data.flatten()[out_flat].item()
        grad_input.data.flatten()[in_flat] = improved_coeff

    return grad_input


class SliceView(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slice_list):
        per_dim = _build_per_dim(slice_list, input.size())
        output = _raw_slice_view(input, slice_list)

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


def slice_view(input: torch.Tensor, slice_list: list) -> torch.Tensor:
    """Autograd-aware slice_view. Falls back to raw when no grad needed."""
    if not input.requires_grad:
        return _raw_slice_view(input, slice_list)
    return SliceView.apply(input, slice_list)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running function/slice_view tests...\n")

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

    # Test 1: No grad — raw passthrough
    print("Test 1: No grad passthrough")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        r = slice_view(t, [0, slice(None)])
        run_test("shape [2]", list(r.shape) == [2])
        run_test("[0]=a", read_out(r, 0) == "a")
        run_test("is symlink", os.path.islink(
            os.path.join(tmpdir, r.st_tensor_uid, "storage", "0", "data")))

    # Test 2: With grad — forward works
    print("Test 2: Autograd forward")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["a\n", "b\n", "c\n"], ["d\n", "e\n", "f\n"]], tmpdir)
        t.requires_grad_(True)
        r = slice_view(t, [0, slice(None)])
        run_test("shape [3]", list(r.shape) == [3])
        run_test("has st_relative_to", hasattr(r, "st_relative_to"))
        run_test("[0]=a", read_out(r, 0) == "a\n")
        run_test("[2]=c", read_out(r, 2) == "c\n")

    # Test 3: With grad — tensor index
    print("Test 3: Autograd tensor index")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["a\n", "b\n", "c\n", "d\n"], tmpdir)
        t.requires_grad_(True)
        r = slice_view(t, [torch.tensor([0, 3])])
        run_test("shape [2]", list(r.shape) == [2])
        run_test("[0]=a", read_out(r, 0) == "a\n")
        run_test("[1]=d", read_out(r, 1) == "d\n")

    # Test 4: Backward — scatter grad back
    print("Test 4: Backward scatter")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["alpha\n", "beta\n", "gamma\n"]], tmpdir)
        t.requires_grad_(True)
        # Select columns 0 and 2
        output = SliceView.apply(t, [slice(None), torch.tensor([0, 2])])
        run_test("output shape [1, 2]", list(output.shape) == [1, 2])
        run_test("out[0,0]=alpha", read_out(output, 0) == "alpha\n")
        run_test("out[0,1]=gamma", read_out(output, 1) == "gamma\n")

        # Modify output: alpha → ALPHA
        mod_data = [[read_out(output, 0).replace("alpha", "ALPHA"), read_out(output, 1)]]
        mod_output = make_tensor(mod_data, tmpdir)
        grad_out = get_diff_tensor(output, mod_output)

        grad_input = slice_backward(grad_out, t, output,
                                    _build_per_dim([slice(None), torch.tensor([0, 2])], t.size()))

        run_test("grad_input not None", grad_input is not None)
        run_test("grad_input shape [1, 3]", list(grad_input.shape) == [1, 3])
        # grad_input[0, 0] should have diff (alpha → ALPHA)
        gi_00 = read_out(grad_input, 0)
        run_test("grad[0,0] has diff", gi_00 is not None and "---" in gi_00)
        run_test("grad[0,0] has +ALPHA", "+ALPHA" in gi_00 if gi_00 else False)
        # grad_input[0, 1] not selected — should be TODO
        gi_01 = read_out(grad_input, 1)
        run_test("grad[0,1] is TODO", gi_01 is not None and "TODO" in gi_01)
        # grad_input[0, 2] unchanged — empty diff
        gi_02 = read_out(grad_input, 2)
        run_test("grad[0,2] empty diff", gi_02 is not None and "---" not in gi_02)

    # Test 5: Backward with int index (dim collapse)
    print("Test 5: Backward int index")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["x\n", "y\n"], ["z\n", "w\n"]], tmpdir)
        t.requires_grad_(True)
        # Select row 1 → shape (2,)
        per_dim = _build_per_dim([1, slice(None)], t.size())
        output = _raw_slice_view(t, [1, slice(None)])
        run_test("output shape [2]", list(output.shape) == [2])

        # Modify z → Z
        mod_data = [read_out(output, 0).replace("z", "Z"), read_out(output, 1)]
        mod_output = make_tensor(mod_data, tmpdir)
        grad_out = get_diff_tensor(output, mod_output)

        grad_input = slice_backward(grad_out, t, output, per_dim)
        run_test("grad shape [2, 2]", list(grad_input.shape) == [2, 2])
        # grad[0, 0] and grad[0, 1] not selected → TODO
        run_test("grad[0,0] TODO", "TODO" in (read_out(grad_input, 0) or ""))
        # grad[1, 0] has diff z→Z
        gi_10 = read_out(grad_input, 2)
        run_test("grad[1,0] has diff", gi_10 is not None and "---" in gi_10)
        run_test("grad[1,0] has +Z", "+Z" in gi_10 if gi_10 else False)

    # Test 6: No grad when not required
    print("Test 6: No grad when not required")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["a\n", "b\n"], tmpdir)
        output = _raw_slice_view(t, [torch.tensor([0])])
        per_dim = _build_per_dim([torch.tensor([0])], t.size())
        grad_out = make_tensor([""], tmpdir)
        result = slice_backward(grad_out, t, output, per_dim)
        run_test("returns None", result is None)

    print("\nAll tests completed.")
