import torch
from typing import List, Union
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


NestedList = Union[str, List["NestedList"]]


def _expand(size: torch.Size, fill: str) -> NestedList:
    """Create a nested list matching the given size, filled with a string."""
    if len(size) == 0:
        return fill
    return [_expand(size[1:], fill) for _ in range(size[0])]


def todo_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """
    Create a symbolic tensor with the same shape as input, filled with "TODO" strings.

    Args:
        input: A symbolic tensor with st_relative_to attribute.

    Returns:
        A new symbolic tensor with the same shape, each element storing "TODO".
    """
    nested_data = _expand(input.size(), "TODO")
    return make_tensor(nested_data, input.st_relative_to)


if __name__ == "__main__":
    import os
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor as mt

    print("Running tests for todo_tensor_like...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D
    print("Test 1: 1D todo_tensor_like")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["a", "b", "c"], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [3])
        run_test("All ones", torch.all(todo == 1).item())
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        for i in range(3):
            with open(os.path.join(root, str(i), "data")) as f:
                run_test(f"Element {i} is 'TODO'", f.read() == "TODO")

    # Test 2: 2D
    print("Test 2: 2D todo_tensor_like")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt([["x", "y"], ["z", "w"]], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Shape matches", list(todo.shape) == [2, 2])
        root = os.path.join(tmpdir, todo.st_tensor_uid, "storage")
        for i in range(4):
            with open(os.path.join(root, str(i), "data")) as f:
                run_test(f"Flat index {i} is 'TODO'", f.read() == "TODO")

    # Test 3: Same relative_to
    print("Test 3: Preserves relative_to")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = mt(["a"], tmpdir)
        todo = todo_tensor_like(orig)
        run_test("Same relative_to", todo.st_relative_to == tmpdir)
        run_test("Different uid", todo.st_tensor_uid != orig.st_tensor_uid)

    print("\nAll tests completed.")
