import os
import itertools
import torch
from typing import List, Optional, Type

from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor
from experience.fs_util.text_merger import TextMerger


def _get_storage_path(tensor: torch.Tensor, flat_index: int) -> str:
    """Get storage file path for a flat index."""
    digits = list(str(flat_index))
    return os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )


def _read_storage(tensor: torch.Tensor, flat_index: int) -> Optional[str]:
    """Read the text content at a flat index, resolving symlinks."""
    path = os.path.realpath(_get_storage_path(tensor, flat_index))
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def _write_storage(tensor: torch.Tensor, flat_index: int, content: str) -> None:
    """Write text content to a flat index."""
    path = _get_storage_path(tensor, flat_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def st_reduce_forward(
    input: torch.Tensor,
    axis: int = -1,
    text_merger: Optional[Type] = None,
) -> torch.Tensor:
    """Reduce a symbolic tensor along an axis, analogous to torch.sum.

    For each position in the output (with the reduced axis removed),
    gathers all elements along the reduced axis from input, packs them
    using text_merger.pack into a single merged string, and writes it
    to the output tensor.

    Shape propagation follows torch.sum(input, dim=axis):
    - The reduced axis is removed from the output shape.

    Args:
        input: Symbolic tensor to reduce.
        axis: Dimension to reduce (default -1, i.e. last dim). No tuple support.
        text_merger: Class with pack/unpack class methods. Defaults to TextMerger.

    Returns:
        Symbolic tensor with the reduced axis removed.
    """
    if text_merger is None:
        text_merger = TextMerger

    # Normalize negative axis
    ndim = input.dim()
    if axis < 0:
        axis = ndim + axis
    assert 0 <= axis < ndim, f"axis {axis} out of range for {ndim}D tensor"

    input_shape = list(input.shape)
    reduce_size = input_shape[axis]

    # Output shape: remove the reduced axis
    output_shape = input_shape[:axis] + input_shape[axis + 1:]

    # Handle scalar output (reducing a 1D tensor)
    if not output_shape:
        output_shape = []

    output = make_none_tensor(output_shape if output_shape else [1], input.st_relative_to)

    # If reducing to scalar, we'll reshape at the end
    is_scalar = not output_shape

    # Compute strides for the input
    input_strides = input.stride()

    # Iterate over all output positions
    # Output coordinates are the input coordinates with the reduced axis removed
    if output_shape:
        output_ranges = [range(s) for s in output_shape]
    else:
        output_ranges = [range(1)]

    for out_flat_idx, out_coords in enumerate(itertools.product(*output_ranges)):
        out_coords = list(out_coords)

        # Build frames: gather elements along the reduced axis
        frames = []
        for k in range(reduce_size):
            # Reconstruct input coordinates
            in_coords = out_coords[:axis] + [k] + out_coords[axis:]
            in_flat_index = sum(c * s for c, s in zip(in_coords, input_strides))

            coefficient = input.data.flatten()[in_flat_index].item()
            content = _read_storage(input, in_flat_index)

            if content is not None and coefficient != 0.0:
                frames.append((k, coefficient, content))

        # Pack and write
        if frames:
            merged = text_merger.pack(frames)
            _write_storage(output, out_flat_idx, merged)
            output.data.flatten()[out_flat_idx] = sum(f[1] for f in frames)
        # else: leave as zero (no content)

    if is_scalar:
        uid = output.st_tensor_uid
        rel = output.st_relative_to
        output = output.squeeze(0)
        output.st_relative_to = rel
        output.st_tensor_uid = uid

    return output


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.fs_util.text_merger import TextMerger, kFrameMarker

    print("Running st_reduce_forward tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_out(tensor, flat_index):
        path = _get_storage_path(tensor, flat_index)
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Reduce 1D tensor (3,) -> scalar
    print("Test 1: Reduce 1D -> scalar")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["alpha", "beta", "gamma"], tmpdir)
        result = st_reduce_forward(inp, axis=0)
        run_test("shape is scalar-like", result.numel() == 1)
        content = read_out(result, 0)
        run_test("content not None", content is not None)
        run_test("has 3 markers", content.count(kFrameMarker) == 3)
        # Verify round-trip
        frames = TextMerger.unpack(content)
        run_test("3 frames", len(frames) == 3)
        run_test("frame[0] = 'alpha'", frames[0][2] == "alpha")
        run_test("frame[1] = 'beta'", frames[1][2] == "beta")
        run_test("frame[2] = 'gamma'", frames[2][2] == "gamma")

    # Test 2: Reduce 2D (2, 3) along axis=-1 -> (2,)
    print("Test 2: Reduce 2D along last axis")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
        result = st_reduce_forward(inp, axis=-1)
        run_test("shape [2]", list(result.shape) == [2])
        # Row 0: a, b, c merged
        content0 = read_out(result, 0)
        frames0 = TextMerger.unpack(content0)
        run_test("row 0: 3 frames", len(frames0) == 3)
        run_test("row 0 frame[0] = 'a'", frames0[0][2] == "a")
        run_test("row 0 frame[2] = 'c'", frames0[2][2] == "c")
        # Row 1: d, e, f merged
        content1 = read_out(result, 1)
        frames1 = TextMerger.unpack(content1)
        run_test("row 1: 3 frames", len(frames1) == 3)
        run_test("row 1 frame[0] = 'd'", frames1[0][2] == "d")

    # Test 3: Reduce 2D (2, 3) along axis=0 -> (3,)
    print("Test 3: Reduce 2D along first axis")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
        result = st_reduce_forward(inp, axis=0)
        run_test("shape [3]", list(result.shape) == [3])
        # Col 0: a, d merged
        content0 = read_out(result, 0)
        frames0 = TextMerger.unpack(content0)
        run_test("col 0: 2 frames", len(frames0) == 2)
        run_test("col 0 frame[0] = 'a'", frames0[0][2] == "a")
        run_test("col 0 frame[1] = 'd'", frames0[1][2] == "d")

    # Test 4: Reduce 3D (2, 3, 2) along axis=1 -> (2, 2)
    print("Test 4: Reduce 3D along middle axis")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [
            [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]],
            [["d0", "d1"], ["e0", "e1"], ["f0", "f1"]],
        ]
        inp = make_tensor(data, tmpdir)
        result = st_reduce_forward(inp, axis=1)
        run_test("shape [2, 2]", list(result.shape) == [2, 2])
        # result[0, 0] = merge of inp[0, :, 0] = [a0, b0, c0]
        content00 = read_out(result, 0)
        frames00 = TextMerger.unpack(content00)
        run_test("[0,0]: 3 frames", len(frames00) == 3)
        run_test("[0,0] frame[0] = 'a0'", frames00[0][2] == "a0")
        run_test("[0,0] frame[1] = 'b0'", frames00[1][2] == "b0")
        run_test("[0,0] frame[2] = 'c0'", frames00[2][2] == "c0")

    # Test 5: Numeric coefficients are summed
    print("Test 5: Numeric coefficient sum")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["x", "y", "z"], tmpdir)
        # Coefficients are all 1.0 from make_tensor
        result = st_reduce_forward(inp, axis=0)
        run_test("coefficient sum = 3.0", abs(result.data.flatten()[0].item() - 3.0) < 1e-5)

    # Test 6: Partial content (some None elements)
    print("Test 6: Partial content")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["hello", None, "world"], tmpdir)
        result = st_reduce_forward(inp, axis=0)
        content = read_out(result, 0)
        frames = TextMerger.unpack(content)
        run_test("2 frames (skip None)", len(frames) == 2)
        run_test("frame[0] = 'hello'", frames[0][2] == "hello")
        run_test("frame[1] = 'world'", frames[1][2] == "world")

    # Test 7: Single element axis (no-op merge)
    print("Test 7: Single element axis")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only"]], tmpdir)  # (1, 1)
        result = st_reduce_forward(inp, axis=-1)
        run_test("shape [1]", list(result.shape) == [1])
        content = read_out(result, 0)
        frames = TextMerger.unpack(content)
        run_test("1 frame", len(frames) == 1)
        run_test("content = 'only'", frames[0][2] == "only")

    # Test 8: Default axis is -1
    print("Test 8: Default axis=-1")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        result = st_reduce_forward(inp)
        run_test("shape [2]", list(result.shape) == [2])

    # Test 9: Skip zero-coefficient elements
    print("Test 9: Skip zero-coefficient elements")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["keep", "skip", "also_keep"], tmpdir)
        # Set middle element coefficient to zero
        inp.data[1] = 0.0
        result = st_reduce_forward(inp, axis=0)
        content = read_out(result, 0)
        frames = TextMerger.unpack(content)
        run_test("2 frames (skip zero coeff)", len(frames) == 2)
        run_test("frame[0] = 'keep'", frames[0][2] == "keep")
        run_test("frame[1] = 'also_keep'", frames[1][2] == "also_keep")
        run_test("coeff sum = 2.0", abs(result.data.flatten()[0].item() - 2.0) < 1e-5)

    # Test 10: Custom text_merger
    print("Test 10: Custom text_merger")

    class SimpleMerger:
        @staticmethod
        def pack(frames):
            return " | ".join(f"{idx}:{text}" for idx, _, text in frames)

        @staticmethod
        def unpack(merged):
            parts = merged.split(" | ")
            result = []
            for p in parts:
                idx_s, text = p.split(":", 1)
                result.append((int(idx_s), 1.0, text))
            return result

    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["x", "y"], tmpdir)
        result = st_reduce_forward(inp, axis=0, text_merger=SimpleMerger)
        content = read_out(result, 0)
        run_test("custom format", content == "0:x | 1:y")

    print("\nAll tests completed.")
