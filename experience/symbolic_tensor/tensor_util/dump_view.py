import os
import itertools
import torch
from typing import List, Tuple


def _str_to_digit_list(s: str) -> List[str]:
    """Convert a string representation of an integer into a list of digit strings."""
    return list(s)


def _get_coordinates(size: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for a given tensor shape."""
    ranges = [range(s) for s in size]
    return [list(coord) for coord in itertools.product(*ranges)]


def _flat_index_from_coordinates(coordinates: List[int], stride: Tuple[int, ...]) -> int:
    """Compute flat storage index from coordinates and stride."""
    return sum(c * s for c, s in zip(coordinates, stride))


def dump_view(tensor: torch.Tensor, dump_dir: str, extension: str) -> None:
    """
    Create a coordinate-based symlink view of a symbolic tensor's storage.

    For each element, creates a relative symlink from a human-readable
    coordinate path to the flat-indexed source file.

    Args:
        tensor: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
        dump_dir: The destination directory for the symlink tree.
        extension: File extension to append (e.g. "py", "txt").
    """
    coordinates_list = _get_coordinates(tensor.size())

    for coordinates in coordinates_list:
        # Compute flat storage index from coordinates and stride
        flat_index = _flat_index_from_coordinates(coordinates, tensor.stride())

        # Source file path: {relative_to}/{uid}/storage/{digit_dirs}/data
        src_index_digits = _str_to_digit_list(str(flat_index))
        src_file_path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*src_index_digits),
            "data",
        )

        # Destination file path: {dump_dir}/{coord_dirs}/data.{extension}
        if coordinates:
            coord_dirs = os.path.join(*[str(c) for c in coordinates])
            dst_file_path = os.path.join(dump_dir, coord_dirs, f"data.{extension}")
        else:
            # Scalar tensor: no coordinate directories
            dst_file_path = os.path.join(dump_dir, f"data.{extension}")

        # Create parent directories for destination
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

        # Create relative symlink from dst to src
        rel_src = os.path.relpath(src_file_path, os.path.dirname(dst_file_path))
        os.symlink(rel_src, dst_file_path)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running tests for dump_view...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: 1D view
    print("Test 1: 1D view")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as view_dir:
        data = ["hello", "world"]
        t = make_tensor(data, src_dir)
        dump_view(t, view_dir, "txt")

        for i, content in enumerate(data):
            link_path = os.path.join(view_dir, str(i), "data.txt")
            run_test(f"Symlink {i} exists", os.path.islink(link_path))
            run_test(f"Symlink {i} resolves", os.path.isfile(link_path))
            with open(link_path) as f:
                run_test(f"Content {i} via symlink", f.read() == content)

    # Test 2: 2D view — coordinates map correctly with stride
    print("Test 2: 2D view")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as view_dir:
        data = [["a", "b", "c"], ["d", "e", "f"]]
        t = make_tensor(data, src_dir)
        dump_view(t, view_dir, "py")

        expected = {(0, 0): "a", (0, 1): "b", (0, 2): "c",
                    (1, 0): "d", (1, 1): "e", (1, 2): "f"}
        for (r, c), content in expected.items():
            link_path = os.path.join(view_dir, str(r), str(c), "data.py")
            run_test(f"({r},{c}) symlink exists", os.path.islink(link_path))
            with open(link_path) as f:
                run_test(f"({r},{c}) = '{content}'", f.read() == content)

    # Test 3: Verify symlinks are relative
    print("Test 3: Relative symlinks")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as view_dir:
        data = ["only"]
        t = make_tensor(data, src_dir)
        dump_view(t, view_dir, "md")

        link_path = os.path.join(view_dir, "0", "data.md")
        target = os.readlink(link_path)
        run_test("Symlink target is relative", not os.path.isabs(target))
        run_test("Symlink resolves to file", os.path.isfile(link_path))

    # Test 4: Multi-digit indices
    print("Test 4: Multi-digit flat indices (12 elements)")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as view_dir:
        data = [f"item_{i}" for i in range(12)]
        t = make_tensor(data, src_dir)
        dump_view(t, view_dir, "txt")

        # Index 11 -> coord (11,) -> view_dir/11/data.txt, links to storage/1/1/data
        link_11 = os.path.join(view_dir, "11", "data.txt")
        run_test("Index 11 symlink exists", os.path.islink(link_11))
        with open(link_11) as f:
            run_test("Index 11 content", f.read() == "item_11")

    print("\nAll tests completed.")
