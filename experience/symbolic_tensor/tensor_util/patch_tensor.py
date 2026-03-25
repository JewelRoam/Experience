import os
import itertools
import subprocess
import tempfile
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


def patch_tensor(lvalue: torch.Tensor, rvalue: torch.Tensor) -> dict:
    """Apply unified diffs from rvalue onto lvalue's storage in-place.

    For each coordinate, rvalue contains a unified diff patch which is applied
    to the corresponding lvalue storage file using ``patch``.

    Args:
        lvalue: Target symbolic tensor whose files will be patched.
        rvalue: Patch symbolic tensor containing unified diffs.

    Returns:
        Stats dict with keys: applied, rejected, fuzzed, skipped, rej_files.
    """
    assert lvalue.shape == rvalue.shape, (
        f"Shape mismatch: lvalue {list(lvalue.shape)} != rvalue {list(rvalue.shape)}"
    )

    stats = {"applied": 0, "rejected": 0, "fuzzed": 0, "skipped": 0, "rej_files": 0}

    coords_list = list(itertools.product(*[range(s) for s in lvalue.size()]))
    for coords in coords_list:
        coords = list(coords)
        lvalue_path = _get_storage_path(lvalue, coords)
        rvalue_path = _get_storage_path(rvalue, coords)

        # Read the patch content; skip if empty or TODO
        with open(rvalue_path) as f:
            patch_content = f.read()
        stripped = patch_content.strip()
        if not stripped or stripped == "TODO":
            stats["skipped"] += 1
            continue

        # Ensure target file ends with newline (patch requires it)
        with open(lvalue_path, "r", encoding="utf-8") as f:
            lvalue_content = f.read()
        if not lvalue_content.endswith("\n"):
            with open(lvalue_path, "w", encoding="utf-8") as f:
                f.write(lvalue_content + "\n")

        # Clean up .rej files from previous iterations
        rej_path = lvalue_path + ".rej"
        if os.path.isfile(rej_path):
            os.unlink(rej_path)

        # Normalize patch content to end with newline
        normalized = patch_content if patch_content.endswith("\n") else patch_content + "\n"

        # Write patch to a temp file and apply
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as pf:
            pf.write(normalized)
            patch_file = pf.name

        try:
            result = subprocess.run(
                ["patch", "--no-backup-if-mismatch", "--fuzz=3",
                 "-i", patch_file, lvalue_path],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                stats["rejected"] += 1
            else:
                stats["applied"] += 1
                if "fuzz" in result.stdout.lower():
                    stats["fuzzed"] += 1
        finally:
            os.unlink(patch_file)

        rej_path_after = lvalue_path + ".rej"
        if os.path.isfile(rej_path_after):
            stats["rej_files"] += 1

    return stats


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor

    print("Running patch_tensor tests...\n")

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

    # Test 1: Roundtrip - diff then patch recovers original->modified
    print("Test 1: Roundtrip diff+patch")
    with tempfile.TemporaryDirectory() as tmpdir:
        original = make_tensor(["line1\nline2\nline3\n", "alpha\nbeta\n"], tmpdir)
        modified = make_tensor(["line1\nchanged\nline3\n", "alpha\ngamma\n"], tmpdir)
        # Take a copy of original to patch
        target = make_tensor(["line1\nline2\nline3\n", "alpha\nbeta\n"], tmpdir)

        diff = get_diff_tensor(original, modified)
        patch_tensor(target, diff)

        run_test("target[0] matches modified[0]",
                 read_storage(target, 0) == "line1\nchanged\nline3\n",
                 "line1\nchanged\nline3\n", read_storage(target, 0))
        run_test("target[1] matches modified[1]",
                 read_storage(target, 1) == "alpha\ngamma\n",
                 "alpha\ngamma\n", read_storage(target, 1))

    # Test 2: Empty diff (no change) leaves target intact
    print("Test 2: Empty diff -> no change")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["same\n"], tmpdir)
        b = make_tensor(["same\n"], tmpdir)
        target = make_tensor(["same\n"], tmpdir)
        diff = get_diff_tensor(a, b)
        patch_tensor(target, diff)
        run_test("target unchanged", read_storage(target, 0) == "same\n")

    # Test 3: 2D roundtrip
    print("Test 3: 2D roundtrip")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = make_tensor([["x\n", "y\n"], ["z\n", "w\n"]], tmpdir)
        mod = make_tensor([["x\n", "Y\n"], ["Z\n", "w\n"]], tmpdir)
        target = make_tensor([["x\n", "y\n"], ["z\n", "w\n"]], tmpdir)
        diff = get_diff_tensor(orig, mod)
        patch_tensor(target, diff)
        run_test("[0,0] unchanged", read_storage(target, 0) == "x\n")
        run_test("[0,1] patched to Y", read_storage(target, 1) == "Y\n",
                 "Y\n", read_storage(target, 1))
        run_test("[1,0] patched to Z", read_storage(target, 2) == "Z\n",
                 "Z\n", read_storage(target, 2))
        run_test("[1,1] unchanged", read_storage(target, 3) == "w\n")

    # Test 4: Shape mismatch raises assertion
    print("Test 4: Shape mismatch assertion")
    with tempfile.TemporaryDirectory() as tmpdir:
        a = make_tensor(["a"], tmpdir)
        b = make_tensor(["x", "y"], tmpdir)
        try:
            patch_tensor(a, b)
            run_test("Assertion raised", False)
        except AssertionError:
            run_test("Assertion raised", True)

    # Test 5: Full roundtrip via st_get_diff and st_patch (registered ops)
    print("Test 5: Roundtrip via registered tensor ops")
    import experience.symbolic_tensor.tensor_util.register_tensor_ops  # noqa: register ops
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = make_tensor(["foo\nbar\n", "baz\n"], tmpdir)
        mod = make_tensor(["foo\nBAR\n", "baz\n"], tmpdir)
        target = make_tensor(["foo\nbar\n", "baz\n"], tmpdir)
        diff = orig.st_get_diff(mod)
        target.st_patch(diff)
        run_test("target[0] patched via ops", read_storage(target, 0) == "foo\nBAR\n",
                 "foo\nBAR\n", read_storage(target, 0))
        run_test("target[1] unchanged via ops", read_storage(target, 1) == "baz\n")

    print("\nAll tests completed.")
