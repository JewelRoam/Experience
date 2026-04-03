import os
import json
import shutil
import torch
from pathlib import Path
from typing import List, Union
from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor


NestedDataType = Union[None, str, Path]
NestedList = Union[NestedDataType, List["NestedList"]]


def _is_leaf(item) -> bool:
    """Check if item is a leaf node (None, str, or Path, not a list)."""
    return item is None or isinstance(item, (str, Path))


def _get_shape(nested_data: NestedList) -> List[int]:
    """Derive the shape from a consistently nested list."""
    if _is_leaf(nested_data):
        return []
    if len(nested_data) == 0:
        return [0]
    shape = [len(nested_data)]
    inner_shape = _get_shape(nested_data[0])
    return shape + inner_shape


def _assert_consistent_shape(nested_data: NestedList, shape: List[int]) -> None:
    """Assert that all branches of nested_data match the derived shape."""
    if not shape:
        assert _is_leaf(nested_data), (
            f"Expected None, str, or Path at leaf, got {type(nested_data)}"
        )
        return
    assert isinstance(nested_data, list), (
        f"Expected list of length {shape[0]}, got {type(nested_data)}"
    )
    assert len(nested_data) == shape[0], (
        f"Expected length {shape[0]}, got {len(nested_data)}"
    )
    for item in nested_data:
        _assert_consistent_shape(item, shape[1:])


def _flatten(nested_data: NestedList) -> List[NestedDataType]:
    """Flatten a nested list into a flat list of leaf elements."""
    if _is_leaf(nested_data):
        return [nested_data]
    result = []
    for item in nested_data:
        result.extend(_flatten(item))
    return result


def _str_to_digit_list(s: str) -> List[str]:
    """Convert a string representation of an integer into a list of digit strings."""
    return list(s)


def _flat_index_to_coords(flat_index: int, shape: List[int]) -> List[int]:
    """Convert a flat index to multi-dimensional coordinates given a shape."""
    coords = []
    for dim_size in reversed(shape):
        coords.append(flat_index % dim_size)
        flat_index //= dim_size
    coords.reverse()
    return coords


def make_tensor(nested_data: NestedList, relative_to: str, symlink: bool = False) -> torch.Tensor:
    """
    Create a symbolic tensor from a nested list of strings or Paths, persisting each
    element to disk under a hash-based directory structure.

    Args:
        nested_data: A single str/Path (scalar tensor) or a consistently shaped
            nested list of strings or pathlib.Path objects.
        relative_to: Root directory for file storage.
        symlink: If True and element is a Path, create a relative symlink instead of copying.

    Returns:
        A ones-filled torch.Tensor with st_relative_to and st_tensor_uid set.
        Values of 1 indicate content has been persisted to disk.
        File contents are saved under {relative_to}/{tensor_uid}/storage/{digit_dirs}/data.
        Shape is saved as JSON under {relative_to}/{tensor_uid}/shape.
    """
    # Derive shape from nested structure
    shape = _get_shape(nested_data)

    # Assert nested data is consistent with derived shape
    _assert_consistent_shape(nested_data, shape)

    # Create zero tensor via make_none_tensor
    tensor = make_none_tensor(shape, relative_to)

    # Flatten nested data
    flattened_data = _flatten(nested_data)

    # Build root dir for this tensor's storage
    tensor_root_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid)

    # Save each element (string content or Path)
    for i, opt_file_content_or_path in enumerate(flattened_data):
        if opt_file_content_or_path is None:
            continue
        index_digits = _str_to_digit_list(str(i))
        file_path = os.path.join(
            tensor_root_dir,
            "storage",
            os.path.join(*index_digits),
            "data",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(opt_file_content_or_path, Path):
            src_path = str(opt_file_content_or_path)
            if symlink:
                rel_src = os.path.relpath(
                    os.path.realpath(str(opt_file_content_or_path)),
                    os.path.realpath(os.path.dirname(file_path)),
                )
                os.symlink(rel_src, file_path)
            else:
                # Resolve symlinks for existence check; skip missing files
                real_src = os.path.realpath(src_path)
                if not os.path.isfile(real_src):
                    continue
                shutil.copy2(src_path, file_path)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(opt_file_content_or_path)
        # Set coefficient to 1.0 for elements that got a file
        coords = _flat_index_to_coords(i, shape)
        tensor[tuple(coords)] = 1

    # Save shape as JSON
    shape_path = os.path.join(tensor_root_dir, "shape")
    os.makedirs(os.path.dirname(shape_path), exist_ok=True)
    with open(shape_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(shape))

    return tensor


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("Running tests for make_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D list of strings
    print("Test 1: 1D list")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ["hello", "world"]
        t = make_tensor(data, tmpdir)
        run_test("Shape is [2]", list(t.shape) == [2], [2], list(t.shape))
        run_test("All ones", torch.all(t == 1).item())
        run_test("st_relative_to set", t.st_relative_to == tmpdir)
        run_test("st_tensor_uid set", isinstance(t.st_tensor_uid, str) and len(t.st_tensor_uid) > 0)
        root = os.path.join(tmpdir, t.st_tensor_uid)
        run_test("File 0 exists", os.path.isfile(os.path.join(root, "storage", "0", "data")))
        run_test("File 1 exists", os.path.isfile(os.path.join(root, "storage", "1", "data")))
        with open(os.path.join(root, "storage", "0", "data")) as f:
            run_test("File 0 content", f.read() == "hello")
        with open(os.path.join(root, "storage", "1", "data")) as f:
            run_test("File 1 content", f.read() == "world")
        with open(os.path.join(root, "shape")) as f:
            run_test("Shape JSON", json.loads(f.read()) == [2])

    # Test 2: 2D nested list
    print("Test 2: 2D nested list")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [["a", "b", "c"], ["d", "e", "f"]]
        t = make_tensor(data, tmpdir)
        run_test("Shape is [2, 3]", list(t.shape) == [2, 3], [2, 3], list(t.shape))
        run_test("All ones", torch.all(t == 1).item())
        root = os.path.join(tmpdir, t.st_tensor_uid)
        # Index 0 -> "a", index 5 -> "f"
        with open(os.path.join(root, "storage", "0", "data")) as f:
            run_test("Index 0 = 'a'", f.read() == "a")
        with open(os.path.join(root, "storage", "5", "data")) as f:
            run_test("Index 5 = 'f'", f.read() == "f")
        with open(os.path.join(root, "shape")) as f:
            run_test("Shape JSON", json.loads(f.read()) == [2, 3])

    # Test 3: Multi-digit index (10+ items)
    print("Test 3: Multi-digit index")
    with tempfile.TemporaryDirectory() as tmpdir:
        data = [f"item_{i}" for i in range(12)]
        t = make_tensor(data, tmpdir)
        run_test("Shape is [12]", list(t.shape) == [12])
        root = os.path.join(tmpdir, t.st_tensor_uid)
        # Index 11 -> digits "1", "1" -> storage/1/1/data
        path_11 = os.path.join(root, "storage", "1", "1", "data")
        run_test("Index 11 path exists", os.path.isfile(path_11))
        with open(path_11) as f:
            run_test("Index 11 content", f.read() == "item_11")

    # Test 4: Inconsistent shape should raise
    print("Test 4: Inconsistent shape assertion")
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_data = [["a", "b"], ["c"]]  # ragged
        try:
            make_tensor(bad_data, tmpdir)
            run_test("Should have raised", False)
        except AssertionError:
            run_test("AssertionError raised", True)

    # Test 5: Path elements (copy mode)
    print("Test 5: Path elements (copy)")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "source.txt"
        src_file.write_text("path content", encoding="utf-8")
        data = [src_file]
        t = make_tensor(data, tmpdir)
        run_test("Shape is [1]", list(t.shape) == [1])
        root = os.path.join(tmpdir, t.st_tensor_uid)
        stored = os.path.join(root, "storage", "0", "data")
        run_test("File exists", os.path.isfile(stored))
        run_test("Not a symlink", not os.path.islink(stored))
        with open(stored) as f:
            run_test("Content matches", f.read() == "path content")

    # Test 6: Path elements (symlink mode)
    print("Test 6: Path elements (symlink)")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "source.txt"
        src_file.write_text("symlinked content", encoding="utf-8")
        data = [src_file]
        t = make_tensor(data, tmpdir, symlink=True)
        root = os.path.join(tmpdir, t.st_tensor_uid)
        stored = os.path.join(root, "storage", "0", "data")
        run_test("Is a symlink", os.path.islink(stored))
        target = os.readlink(stored)
        run_test("Symlink is relative", not os.path.isabs(target))
        with open(stored) as f:
            run_test("Content via symlink", f.read() == "symlinked content")

    # Test 7: Mixed str and Path elements
    print("Test 7: Mixed str and Path")
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "mixed.txt"
        src_file.write_text("from path", encoding="utf-8")
        data = ["from string", src_file]
        t = make_tensor(data, tmpdir)
        root = os.path.join(tmpdir, t.st_tensor_uid)
        with open(os.path.join(root, "storage", "0", "data")) as f:
            run_test("Str element", f.read() == "from string")
        with open(os.path.join(root, "storage", "1", "data")) as f:
            run_test("Path element", f.read() == "from path")

    print("\nAll tests completed.")
