import os
import itertools
import shutil
import torch
from typing import List


def _get_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> str:
    """Get the real storage file path for a given set of coordinates."""
    stride = tensor.stride()
    flat_index = sum(c * s for c, s in zip(coordinates, stride))
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    return os.path.realpath(path)


def assign_tensor(lvalue: torch.Tensor, rvalue: torch.Tensor) -> None:
    """Copy all file contents from rvalue storage to lvalue storage element-wise.

    Both tensors must have the same shape. For each coordinate, the file content
    at rvalue's storage is copied to lvalue's storage.

    Args:
        lvalue: Destination symbolic tensor.
        rvalue: Source symbolic tensor.
    """
    assert lvalue.shape == rvalue.shape, (
        f"Shape mismatch: lvalue {list(lvalue.shape)} != rvalue {list(rvalue.shape)}"
    )

    coords_list = list(itertools.product(*[range(s) for s in lvalue.size()]))
    for coords in coords_list:
        coords = list(coords)
        lvalue_path = _get_storage_path(lvalue, coords)
        rvalue_path = _get_storage_path(rvalue, coords)
        shutil.copy2(rvalue_path, lvalue_path)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like

    print("Running assign_tensor tests...\n")

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

    # Test 1: 1D assign
    print("Test 1: 1D assign")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["hello", "world"], tmpdir)
        dst = make_tensor(["aaa", "bbb"], tmpdir)
        assign_tensor(dst, src)
        run_test("dst[0] == 'hello'", read_storage(dst, 0) == "hello", "hello", read_storage(dst, 0))
        run_test("dst[1] == 'world'", read_storage(dst, 1) == "world", "world", read_storage(dst, 1))

    # Test 2: 2D assign
    print("Test 2: 2D assign")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor([["a", "b"], ["c", "d"]], tmpdir)
        dst = make_tensor([["x", "y"], ["z", "w"]], tmpdir)
        assign_tensor(dst, src)
        run_test("dst[0,0] == 'a'", read_storage(dst, 0) == "a")
        run_test("dst[0,1] == 'b'", read_storage(dst, 1) == "b")
        run_test("dst[1,0] == 'c'", read_storage(dst, 2) == "c")
        run_test("dst[1,1] == 'd'", read_storage(dst, 3) == "d")

    # Test 3: Assign to TODO tensor
    print("Test 3: Assign to TODO tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["content_a", "content_b"], tmpdir)
        dst = todo_tensor_like(src)
        run_test("dst[0] is TODO before", read_storage(dst, 0) == "TODO")
        assign_tensor(dst, src)
        run_test("dst[0] == 'content_a' after", read_storage(dst, 0) == "content_a")
        run_test("dst[1] == 'content_b' after", read_storage(dst, 1) == "content_b")

    # Test 4: Shape mismatch raises assertion
    print("Test 4: Shape mismatch assertion")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["a", "b"], tmpdir)
        dst = make_tensor(["x", "y", "z"], tmpdir)
        try:
            assign_tensor(dst, src)
            run_test("Assertion raised", False)
        except AssertionError:
            run_test("Assertion raised", True)

    # Test 5: Scalar (0D) assign
    print("Test 5: Scalar assign")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor("scalar_src", tmpdir)
        dst = make_tensor("scalar_dst", tmpdir)
        assign_tensor(dst, src)
        run_test("dst == 'scalar_src'", read_storage(dst, 0) == "scalar_src")

    # Test 6: Source unchanged after assign
    print("Test 6: Source unchanged")
    with tempfile.TemporaryDirectory() as tmpdir:
        src = make_tensor(["original"], tmpdir)
        dst = make_tensor(["target"], tmpdir)
        assign_tensor(dst, src)
        run_test("src[0] still 'original'", read_storage(src, 0) == "original")
        run_test("dst[0] == 'original'", read_storage(dst, 0) == "original")

    print("\nAll tests completed.")
