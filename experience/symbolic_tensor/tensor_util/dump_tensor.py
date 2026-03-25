import os
import json
import shutil
import torch
from typing import List


def _str_to_digit_list(s: str) -> List[str]:
    """Convert a string representation of an integer into a list of digit strings."""
    return list(s)


def dump_tensor(tensor: torch.Tensor, dump_dir: str) -> None:
    """
    Dump a symbolic tensor's storage files to a target directory.

    Copies all element files from the tensor's source storage into dump_dir,
    and saves shape and stride metadata as JSON.

    Args:
        tensor: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
        dump_dir: The destination directory to dump into.
    """
    element_count = tensor.numel()
    src_root_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid)

    for i in range(element_count):
        index_digits = _str_to_digit_list(str(i))
        relative_path = os.path.join(os.path.join(*index_digits), "data")

        src_path = os.path.join(src_root_dir, "storage", relative_path)
        dst_path = os.path.join(dump_dir, "storage", relative_path)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    # Save shape as JSON
    shape_path = os.path.join(dump_dir, "shape")
    with open(shape_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(list(tensor.shape)))

    # Save stride as JSON
    stride_path = os.path.join(dump_dir, "stride")
    with open(stride_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(list(tensor.stride())))


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running tests for dump_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D tensor dump
    print("Test 1: 1D dump")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        data = ["hello", "world", "foo"]
        t = make_tensor(data, src_dir)
        dump_tensor(t, dst_dir)

        # Check storage files copied
        for i, content in enumerate(data):
            digits = _str_to_digit_list(str(i))
            path = os.path.join(dst_dir, "storage", os.path.join(*digits), "data")
            run_test(f"File {i} exists", os.path.isfile(path))
            with open(path) as f:
                run_test(f"File {i} content", f.read() == content)

        # Check shape
        with open(os.path.join(dst_dir, "shape")) as f:
            shape = json.loads(f.read())
            run_test("Shape JSON", shape == [3], [3], shape)

        # Check stride
        with open(os.path.join(dst_dir, "stride")) as f:
            stride = json.loads(f.read())
            run_test("Stride JSON", stride == list(t.stride()), list(t.stride()), stride)

    # Test 2: 2D tensor dump
    print("Test 2: 2D dump")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        data = [["a", "b"], ["c", "d"]]
        t = make_tensor(data, src_dir)
        dump_tensor(t, dst_dir)

        run_test("Shape is [2, 2]",
                 json.loads(open(os.path.join(dst_dir, "shape")).read()) == [2, 2])
        run_test("Stride saved",
                 json.loads(open(os.path.join(dst_dir, "stride")).read()) == list(t.stride()))

        # Verify all 4 files
        flat = ["a", "b", "c", "d"]
        for i, content in enumerate(flat):
            digits = _str_to_digit_list(str(i))
            path = os.path.join(dst_dir, "storage", os.path.join(*digits), "data")
            with open(path) as f:
                run_test(f"Index {i} = '{content}'", f.read() == content)

    # Test 3: Multi-digit indices
    print("Test 3: Multi-digit indices (12 elements)")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        data = [f"item_{i}" for i in range(12)]
        t = make_tensor(data, src_dir)
        dump_tensor(t, dst_dir)

        # Check index 11 -> 1/1/data
        path_11 = os.path.join(dst_dir, "storage", "1", "1", "data")
        run_test("Index 11 exists", os.path.isfile(path_11))
        with open(path_11) as f:
            run_test("Index 11 content", f.read() == "item_11")

        run_test("Shape is [12]",
                 json.loads(open(os.path.join(dst_dir, "shape")).read()) == [12])

    print("\nAll tests completed.")
