import os
import json
import math
import torch
from typing import List, Union
from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


NestedList = Union[str, List["NestedList"]]


def _str_to_digit_list(s: str) -> List[str]:
    """Convert a string representation of an integer into a list of digit strings."""
    return list(s)


def _assert_is_dumped_tensor_dir(load_dir: str) -> None:
    """Assert that load_dir contains shape, stride, and storage."""
    assert os.path.isfile(os.path.join(load_dir, "shape")), (
        f"Missing 'shape' in {load_dir}"
    )
    assert os.path.isfile(os.path.join(load_dir, "stride")), (
        f"Missing 'stride' in {load_dir}"
    )
    assert os.path.isdir(os.path.join(load_dir, "storage")), (
        f"Missing 'storage' directory in {load_dir}"
    )


def _read_flat_data(load_dir: str, element_count: int) -> List[str]:
    """Read all storage files in flat index order."""
    flat_data = []
    for i in range(element_count):
        digits = _str_to_digit_list(str(i))
        path = os.path.join(load_dir, "storage", os.path.join(*digits), "data")
        with open(path, "r", encoding="utf-8") as f:
            flat_data.append(f.read())
    return flat_data


def _unflatten(flat_data: List[str], shape: List[int]) -> NestedList:
    """Reshape a flat list of strings into a nested list according to shape."""
    if not shape:
        return flat_data[0]
    if len(shape) == 1:
        return flat_data
    inner_size = math.prod(shape[1:])
    result = []
    for i in range(shape[0]):
        start = i * inner_size
        chunk = flat_data[start:start + inner_size]
        result.append(_unflatten(chunk, shape[1:]))
    return result


def load_tensor(load_dir: str, relative_to: str) -> torch.Tensor:
    """
    Load a symbolic tensor from a dumped directory.

    Reads storage files, shape, and stride from load_dir, recreates the tensor
    via make_tensor into relative_to, then applies as_strided to restore the
    original view.

    Args:
        load_dir: A directory previously created by dump_tensor.
        relative_to: Root directory for the new tensor's file storage.

    Returns:
        A symbolic torch.Tensor with st_relative_to and st_tensor_uid set.
    """
    # Assert valid dump directory
    _assert_is_dumped_tensor_dir(load_dir)

    # Load shape and stride from dump
    with open(os.path.join(load_dir, "shape"), "r", encoding="utf-8") as f:
        shape = json.loads(f.read())
    with open(os.path.join(load_dir, "stride"), "r", encoding="utf-8") as f:
        stride = json.loads(f.read())

    # Read nested data from storage files
    element_count = math.prod(shape) if shape else 1
    flat_data = _read_flat_data(load_dir, element_count)
    nested_data = _unflatten(flat_data, shape)

    # Recreate tensor via make_tensor
    tensor = make_tensor(nested_data, relative_to)

    # Preserve symbolic attributes across as_strided (it returns a new view)
    st_relative_to = tensor.st_relative_to
    st_tensor_uid = tensor.st_tensor_uid

    # Restore original view with as_strided
    tensor = tensor.as_strided(shape, stride)
    tensor.st_relative_to = st_relative_to
    tensor.st_tensor_uid = st_tensor_uid

    return tensor


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.dump_tensor import dump_tensor

    print("Running tests for load_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D roundtrip (make -> dump -> load)
    print("Test 1: 1D roundtrip")
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as dump_dir, \
         tempfile.TemporaryDirectory() as load_dir:
        data = ["hello", "world", "foo"]
        t_orig = make_tensor(data, src_dir)
        dump_tensor(t_orig, dump_dir)
        t_loaded = load_tensor(dump_dir, load_dir)

        run_test("Shape matches", list(t_loaded.shape) == [3], [3], list(t_loaded.shape))
        run_test("Stride matches", list(t_loaded.stride()) == list(t_orig.stride()))
        run_test("All ones", torch.all(t_loaded == 1).item())
        run_test("Has st_relative_to", t_loaded.st_relative_to == load_dir)
        run_test("Has st_tensor_uid", isinstance(t_loaded.st_tensor_uid, str))

        # Verify file contents roundtripped correctly
        loaded_root = os.path.join(load_dir, t_loaded.st_tensor_uid)
        for i, content in enumerate(data):
            digits = _str_to_digit_list(str(i))
            path = os.path.join(loaded_root, "storage", os.path.join(*digits), "data")
            with open(path) as f:
                run_test(f"Content {i} roundtrip", f.read() == content)

    # Test 2: 2D roundtrip
    print("Test 2: 2D roundtrip")
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as dump_dir, \
         tempfile.TemporaryDirectory() as load_dir:
        data = [["a", "b", "c"], ["d", "e", "f"]]
        t_orig = make_tensor(data, src_dir)
        dump_tensor(t_orig, dump_dir)
        t_loaded = load_tensor(dump_dir, load_dir)

        run_test("Shape [2,3]", list(t_loaded.shape) == [2, 3], [2, 3], list(t_loaded.shape))
        run_test("Stride matches", list(t_loaded.stride()) == list(t_orig.stride()))

        flat = ["a", "b", "c", "d", "e", "f"]
        loaded_root = os.path.join(load_dir, t_loaded.st_tensor_uid)
        for i, content in enumerate(flat):
            digits = _str_to_digit_list(str(i))
            path = os.path.join(loaded_root, "storage", os.path.join(*digits), "data")
            with open(path) as f:
                run_test(f"Content {i} = '{content}'", f.read() == content)

    # Test 3: Invalid dump dir should raise
    print("Test 3: Invalid dump dir assertion")
    with tempfile.TemporaryDirectory() as bad_dir, \
         tempfile.TemporaryDirectory() as load_dir:
        try:
            load_tensor(bad_dir, load_dir)
            run_test("Should have raised", False)
        except AssertionError:
            run_test("AssertionError raised", True)

    # Test 4: Different relative_to produces new uid
    print("Test 4: New uid on load")
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as dump_dir, \
         tempfile.TemporaryDirectory() as load_dir:
        t_orig = make_tensor(["x"], src_dir)
        dump_tensor(t_orig, dump_dir)
        t_loaded = load_tensor(dump_dir, load_dir)
        run_test("Different uid", t_loaded.st_tensor_uid != t_orig.st_tensor_uid)
        run_test("New relative_to", t_loaded.st_relative_to == load_dir)

    print("\nAll tests completed.")
