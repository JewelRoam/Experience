import os
import itertools
import pathlib
import torch
from typing import Any, List, Union

NestedList = Union[pathlib.Path, List["NestedList"]]


def _get_storage_path(tensor: torch.Tensor, coordinates: List[int]) -> pathlib.Path:
    """Compute the flat index from coordinates and strides, return the storage file path."""
    flat_index = sum(c * s for c, s in zip(coordinates, tensor.stride()))
    digits = list(str(flat_index))
    return pathlib.Path(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )


def _build_nested(
    tensor: torch.Tensor,
    coordinates_list: List[List[int]],
    shape: List[int],
) -> NestedList:
    """Reshape a flat list of paths into a nested list matching the tensor shape."""
    paths = [_get_storage_path(tensor, coords) for coords in coordinates_list]
    if not shape:
        return paths[0]
    if len(shape) == 1:
        return paths
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _build_nested(
            tensor,
            coordinates_list[i * chunk_size:(i + 1) * chunk_size],
            shape[1:],
        )
        for i in range(shape[0])
    ]


def get_nested_list_file_pathes(tensor: torch.Tensor) -> NestedList:
    """Return a nested list of pathlib.Path matching the tensor's shape.

    Each path points to the storage file for the corresponding tensor element.

    Args:
        tensor: A symbolic tensor with st_relative_to and st_tensor_uid attributes.

    Returns:
        A nested list of pathlib.Path matching tensor.shape.
    """
    shape = list(tensor.size())
    coordinates_list = [list(c) for c in itertools.product(*[range(s) for s in shape])]
    return _build_nested(tensor, coordinates_list, shape)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running get_nested_list_file_pathes tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D tensor
    print("Test 1: 1D tensor [3]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = make_tensor(["a", "b", "c"], tmpdir)
        result = get_nested_list_file_pathes(tensor)
        run_test("Result is a list", isinstance(result, list))
        run_test("Length is 3", len(result) == 3, 3, len(result))
        run_test("Elements are Path", all(isinstance(p, pathlib.Path) for p in result))
        # Verify paths exist and contain correct data
        for i, p in enumerate(result):
            run_test(f"Path [{i}] exists", p.exists(), True, p.exists())
            content = p.read_text()
            expected_content = ["a", "b", "c"][i]
            run_test(f"Path [{i}] content", content == expected_content, expected_content, repr(content))

    # Test 2: 2D tensor
    print("Test 2: 2D tensor [2, 3]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
        result = get_nested_list_file_pathes(tensor)
        run_test("Outer list length 2", len(result) == 2, 2, len(result))
        run_test("Inner list length 3", len(result[0]) == 3, 3, len(result[0]))
        run_test("[0][0] content", result[0][0].read_text() == "a")
        run_test("[0][2] content", result[0][2].read_text() == "c")
        run_test("[1][0] content", result[1][0].read_text() == "d")
        run_test("[1][2] content", result[1][2].read_text() == "f")

    # Test 3: Scalar tensor
    print("Test 3: Scalar tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = make_tensor("hello", tmpdir)
        result = get_nested_list_file_pathes(tensor)
        run_test("Result is a Path", isinstance(result, pathlib.Path))
        run_test("Content", result.read_text() == "hello")

    # Test 4: 3D tensor
    print("Test 4: 3D tensor [2, 1, 2]")
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = make_tensor([[["x", "y"]], [["z", "w"]]], tmpdir)
        result = get_nested_list_file_pathes(tensor)
        run_test("Shape [2][1][2]", len(result) == 2 and len(result[0]) == 1 and len(result[0][0]) == 2)
        run_test("[0][0][0] content", result[0][0][0].read_text() == "x")
        run_test("[0][0][1] content", result[0][0][1].read_text() == "y")
        run_test("[1][0][0] content", result[1][0][0].read_text() == "z")
        run_test("[1][0][1] content", result[1][0][1].read_text() == "w")

    # Test 5: Paths use correct storage layout
    print("Test 5: Storage path structure")
    with tempfile.TemporaryDirectory() as tmpdir:
        tensor = make_tensor(["a", "b"], tmpdir)
        result = get_nested_list_file_pathes(tensor)
        path_str = str(result[0])
        run_test("Path contains 'storage'", "storage" in path_str)
        run_test("Path ends with 'data'", path_str.endswith("data"))
        run_test("Path contains tensor uid", tensor.st_tensor_uid in path_str)

    print("\nAll tests completed.")
