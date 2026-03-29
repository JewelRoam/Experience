import os
import itertools
import subprocess
import tempfile
import torch
from typing import List

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


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


def _ensure_newline(path: str) -> str:
    """Return a temp file path with content of *path* guaranteed to end with newline.

    If the original file already ends with a newline, returns the original path
    (no temp file created). Otherwise returns a new temp path (caller must unlink).
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if content.endswith("\n"):
        return path
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(content + "\n")
    tmp.close()
    return tmp.name


def get_diff_tensor(lvalue: torch.Tensor, rvalue: torch.Tensor) -> torch.Tensor:
    """Compute element-wise unified diff between two symbolic tensors.

    For each coordinate, runs ``diff -u`` on the lvalue and rvalue storage
    files and stores the resulting unified diff in a new tensor.

    Args:
        lvalue: Original symbolic tensor.
        rvalue: Modified symbolic tensor.

    Returns:
        A new symbolic tensor of the same shape where each element contains
        the unified diff between the corresponding lvalue and rvalue elements.
    """
    assert lvalue.shape == rvalue.shape, (
        f"Shape mismatch: lvalue {list(lvalue.shape)} != rvalue {list(rvalue.shape)}"
    )

    coords_list = list(itertools.product(*[range(s) for s in lvalue.size()]))
    diffs = []
    for coords in coords_list:
        coords = list(coords)
        lvalue_path = _get_storage_path(lvalue, coords)
        rvalue_path = _get_storage_path(rvalue, coords)
        # Normalize to newline-terminated temps so diffs never contain
        # '\ No newline at end of file' — keeps patches clean for `patch` cmd
        lv_norm = _ensure_newline(lvalue_path)
        rv_norm = _ensure_newline(rvalue_path)
        try:
            result = subprocess.run(
                ["diff", "-u", "--label", "data", "--label", "data",
                 lv_norm, rv_norm],
                capture_output=True, text=True,
            )
            diff_text = result.stdout
            diffs.append(diff_text if diff_text else None)
        finally:
            if lv_norm != lvalue_path:
                os.unlink(lv_norm)
            if rv_norm != rvalue_path:
                os.unlink(rv_norm)

    # Reshape diffs into nested list matching original shape
    shape = list(lvalue.size())
    diff_nested = _reshape_flat_to_nested(diffs, shape)

    ret = make_tensor(diff_nested, lvalue.st_relative_to)
    return ret


def _reshape_flat_to_nested(flat: list, shape: List[int]):
    """Reshape a flat list into a nested list matching the given shape."""
    if len(shape) == 0:
        return flat[0]
    if len(shape) == 1:
        return flat
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    result = []
    for i in range(shape[0]):
        chunk = flat[i * chunk_size : (i + 1) * chunk_size]
        result.append(_reshape_flat_to_nested(chunk, shape[1:]))
    return result


if __name__ == "__main__":
    import tempfile

    print("Running get_diff_tensor tests...\n")

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
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Identical tensors produce empty diff
    print("Test 1: Identical tensors -> empty diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["hello\n", "world\n"], tmpdir)
        b = make_tensor(["hello\n", "world\n"], tmpdir)
        diff = get_diff_tensor(a, b)
        run_test("shape matches", list(diff.shape) == [2])
        run_test("diff[0] is None (no diff)", read_storage(diff, 0) is None)
        run_test("diff[1] is None (no diff)", read_storage(diff, 1) is None)

    # Test 2: Different content produces unified diff
    print("Test 2: Different content -> unified diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["line1\nline2\n", "same\n"], tmpdir)
        b = make_tensor(["line1\nmodified\n", "same\n"], tmpdir)
        diff = get_diff_tensor(a, b)
        d0 = read_storage(diff, 0)
        run_test("diff[0] contains ---", "---" in d0)
        run_test("diff[0] contains +++", "+++" in d0)
        run_test("diff[0] contains -line2", "-line2" in d0)
        run_test("diff[0] contains +modified", "+modified" in d0)
        run_test("diff[1] is None (same content)", read_storage(diff, 1) is None)

    # Test 3: 2D tensor diff
    print("Test 3: 2D tensor diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor([["a\n", "b\n"], ["c\n", "d\n"]], tmpdir)
        b = make_tensor([["a\n", "B\n"], ["C\n", "d\n"]], tmpdir)
        diff = get_diff_tensor(a, b)
        run_test("shape matches", list(diff.shape) == [2, 2])
        run_test("diff[0,0] None (same)", read_storage(diff, 0) is None)
        run_test("diff[0,1] has diff", "---" in read_storage(diff, 1))
        run_test("diff[1,0] has diff", "---" in read_storage(diff, 2))
        run_test("diff[1,1] None (same)", read_storage(diff, 3) is None)

    # Test 4: Shape mismatch raises assertion
    print("Test 4: Shape mismatch assertion")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["a"], tmpdir)
        b = make_tensor(["x", "y"], tmpdir)
        try:
            get_diff_tensor(a, b)
            run_test("Assertion raised", False)
        except AssertionError:
            run_test("Assertion raised", True)

    print("\nAll tests completed.")
