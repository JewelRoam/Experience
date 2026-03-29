import os
import torch
from typing import List, Optional, Union
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


NestedList = Union[Optional[str], List["NestedList"]]


def _map_nested(nested, fn):
    """Apply fn to each leaf element, preserving nested list structure."""
    if isinstance(nested, list):
        return [_map_nested(item, fn) for item in nested]
    return fn(nested)


def _check_file_path(file_path) -> Optional[str]:
    """Return 'TODO' if file exists and has non-empty content, None otherwise."""
    path = str(file_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content:
            return None
        return "TODO"
    except Exception:
        return None


def todo_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """
    Create a symbolic tensor with the same shape as input.

    Elements are "TODO" where the input has existing non-empty content on disk,
    None (no file created) where the input has no file or empty content.

    Args:
        input: A symbolic tensor with st_relative_to and st_file_paths().

    Returns:
        A new symbolic tensor with the same shape. Positions with content
        get "TODO" strings; positions without content get no file on disk.
    """
    file_paths = input.st_file_paths()
    nested_data = _map_nested(file_paths, _check_file_path)
    return make_tensor(nested_data, input.st_relative_to)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as mt

    print("Running tests for todo_tensor_like...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: All elements have content -> all TODO
    print("Test 1: All elements have content")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["a", "b", "c"], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [3])
        run_test("All ones", torch.all(todo == 1).item())
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        for i in range(3):
            with open(os.path.join(root, str(i), "data")) as f:
                run_test(f"Element {i} is 'TODO'", f.read() == "TODO")

    # Test 2: Mixed content and None
    print("Test 2: Mixed content and None elements")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["hello", None, "world"], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [3])
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        # Element 0: has content -> TODO
        path0 = os.path.join(root, "0", "data")
        run_test("Element 0 has TODO",
                 os.path.exists(path0) and open(path0).read() == "TODO")
        # Element 1: was None -> no file
        path1 = os.path.join(root, "1", "data")
        run_test("Element 1 has no file", not os.path.exists(path1))
        # Element 2: has content -> TODO
        path2 = os.path.join(root, "2", "data")
        run_test("Element 2 has TODO",
                 os.path.exists(path2) and open(path2).read() == "TODO")

    # Test 3: 2D tensor
    print("Test 3: 2D tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt([["x", "y"], ["z", "w"]], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [2, 2])
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        for i in range(4):
            with open(os.path.join(root, str(i), "data")) as f:
                run_test(f"Flat index {i} is 'TODO'", f.read() == "TODO")

    # Test 4: Preserves relative_to
    print("Test 4: Preserves relative_to")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["a"], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Same relative_to", todo.st_relative_to == tmpdir)
        run_test("Different uid", todo.st_tensor_uid != orig.st_tensor_uid)

    # Test 5: Empty file content treated as None
    print("Test 5: Empty file content -> no TODO")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["content"], tmpdir)
        # Overwrite storage file with empty content
        storage_path = os.path.join(
            tmpdir, orig.st_tensor_uid, "storage", "0", "data"
        )
        with open(storage_path, "w") as f:
            f.write("")
        todo = todo_tensor_like(orig)
        todo_path = os.path.join(
            tmpdir, todo.st_tensor_uid, "storage", "0", "data"
        )
        run_test("Empty content -> no file", not os.path.exists(todo_path))

    # Test 6: 2D with None elements
    print("Test 6: 2D with None elements")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt([["a", None], [None, "d"]], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [2, 2])
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        run_test("Index 0 has TODO",
                 os.path.exists(os.path.join(root, "0", "data")))
        run_test("Index 1 no file",
                 not os.path.exists(os.path.join(root, "1", "data")))
        run_test("Index 2 no file",
                 not os.path.exists(os.path.join(root, "2", "data")))
        run_test("Index 3 has TODO",
                 os.path.exists(os.path.join(root, "3", "data")))

    print("\nAll tests completed.")
