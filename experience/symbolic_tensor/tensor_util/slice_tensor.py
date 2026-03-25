import os
import itertools
import torch
from pathlib import Path
from typing import List, Union

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor, NestedList


SliceElement = Union[torch.Tensor, slice, int]


def _get_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> Path:
    """Get the storage file Path for a given set of coordinates."""
    stride = tensor.stride()
    flat_index = sum(c * s for c, s in zip(coordinates, stride))
    digits = list(str(flat_index))
    return Path(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )


def _normalize_slice_element(
    elem: SliceElement, dim_size: int
) -> tuple:
    """Normalize a slice element to (list of indices, collapses_dim)."""
    if isinstance(elem, int):
        return [elem], True
    elif isinstance(elem, slice):
        return list(range(*elem.indices(dim_size))), False
    elif isinstance(elem, torch.Tensor):
        if elem.dim() == 0:
            return [int(elem.item())], True
        else:
            return elem.tolist(), False
    else:
        raise TypeError(f"Unsupported slice element type: {type(elem)}")


def _build_nested(flat_paths: List[Path], shape: List[int]) -> NestedList:
    """Reshape a flat list of Paths into a nested list according to shape."""
    if not shape:
        return flat_paths[0]
    if len(shape) == 1:
        return flat_paths
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    result = []
    for i in range(shape[0]):
        start = i * chunk_size
        result.append(_build_nested(flat_paths[start:start + chunk_size], shape[1:]))
    return result


def slice_tensor(
    input: torch.Tensor,
    slice_tensors: List[SliceElement],
) -> torch.Tensor:
    """
    Create a symbolic tensor by copying (not symlinking) selected elements
    from an input tensor. Same indexing semantics as slice_view, but the
    new tensor has independent storage (file copies, not symlinks).

    This is useful when the result will be modified (e.g., by an LLM)
    without affecting the source tensor.

    Args:
        input: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
        slice_tensors: A list of indexing elements, one per dimension of input.

    Returns:
        A new symbolic tensor with copied storage files.
    """
    assert len(slice_tensors) == len(input.size()), (
        f"Expected {len(input.size())} slice elements, got {len(slice_tensors)}"
    )

    input_size = input.size()
    per_dim = []
    output_shape = []
    for d, elem in enumerate(slice_tensors):
        indices, collapses = _normalize_slice_element(elem, input_size[d])
        per_dim.append((indices, collapses))
        if not collapses:
            output_shape.append(len(indices))

    # Handle empty output (any dimension is 0)
    if 0 in output_shape:
        ret = make_tensor([], input.st_relative_to, symlink=False)
        reshaped = ret.view(output_shape)
        reshaped.st_relative_to = ret.st_relative_to
        reshaped.st_tensor_uid = ret.st_tensor_uid
        return reshaped

    # Generate all coordinate combinations
    all_indices = [idx for idx, _ in per_dim]
    selected_paths: List[Path] = []
    for combo in itertools.product(*all_indices):
        selected_paths.append(_get_storage_path(input, list(combo)))

    # Build nested list matching output shape
    if output_shape:
        nested_paths = _build_nested(selected_paths, output_shape)
    else:
        nested_paths = selected_paths[0]

    # symlink=False -> copy files instead of creating symlinks
    ret = make_tensor(nested_paths, input.st_relative_to, symlink=False)

    count_collapsed = sum(1 for _, collapses in per_dim if collapses)
    expected_ndim = len(input.size()) - count_collapsed
    assert len(ret.size()) == expected_ndim, (
        f"Expected output ndim {expected_ndim}, got {len(ret.size())}"
    )

    return ret


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as mt

    print("Running tests for slice_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Copy, not symlink
    print("Test 1: Copy semantics")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["hello", "world"]
        t = mt(data, tmpdir)
        result = slice_tensor(t, [torch.tensor([0, 1], dtype=torch.long)])
        run_test("Shape is [2]", list(result.shape) == [2])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        f0 = os.path.join(root, "0", "data")
        run_test("File exists", os.path.isfile(f0))
        run_test("NOT a symlink", not os.path.islink(f0))
        with open(f0) as f:
            run_test("Content is 'hello'", f.read() == "hello")

    # Test 2: Modifying copy doesn't affect source
    print("Test 2: Independent storage")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["original"]
        t = mt(data, tmpdir)
        result = slice_tensor(t, [torch.tensor([0], dtype=torch.long)])
        # Modify the copy
        copy_path = os.path.join(tmpdir, result.st_tensor_uid, "storage", "0", "data")
        with open(copy_path, "w") as f:
            f.write("modified")
        # Check source is unchanged
        src_path = os.path.join(tmpdir, t.st_tensor_uid, "storage", "0", "data")
        with open(src_path) as f:
            run_test("Source unchanged", f.read() == "original")

    # Test 3: Int indexing collapses dim
    print("Test 3: Int indexing")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b"], ["c", "d"]]
        t = mt(data, tmpdir)
        result = slice_tensor(t, [0, slice(None)])
        run_test("Shape is [2]", list(result.shape) == [2])
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        with open(os.path.join(root, "0", "data")) as f:
            run_test("(0,0) -> 'a'", f.read() == "a")

    # Test 4: Scalar result
    print("Test 4: Scalar")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["x", "y"], ["z", "w"]]
        t = mt(data, tmpdir)
        result = slice_tensor(t, [1, 0])
        run_test("Shape is []", list(result.shape) == [])
        stored = os.path.join(tmpdir, result.st_tensor_uid, "storage", "0", "data")
        run_test("Not symlink", not os.path.islink(stored))
        with open(stored) as f:
            run_test("(1,0) -> 'z'", f.read() == "z")

    # Test 5: Empty tensor index (corner case)
    print("Test 5: Empty tensor index")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b"], ["c", "d"]]
        t = mt(data, tmpdir)
        result = slice_tensor(t, [torch.tensor([], dtype=torch.long), slice(None)])
        run_test("Shape is [0, 2]", list(result.shape) == [0, 2], [0, 2], list(result.shape))
        run_test("numel is 0", result.numel() == 0)
        run_test("Not symlink storage", not os.path.islink(
            os.path.join(tmpdir, result.st_tensor_uid, "storage")) if os.path.exists(
            os.path.join(tmpdir, result.st_tensor_uid, "storage")) else True)

    print("\nAll tests completed.")
