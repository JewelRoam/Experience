import os
from pathlib import Path
from typing import List, Optional

import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor


def _get_all_file_paths(root_dir: str, extension: Optional[str] = None) -> List[str]:
    """Recursively collect all file paths under root_dir, optionally filtered by extension."""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if extension is not None and not fname.endswith(extension):
                continue
            paths.append(os.path.join(dirpath, fname))
    paths.sort()
    return paths


class SoleFileBatchDataLoader:
    """
    Yields symbolic tensors from files in a directory.
    Each element's storage contains a symlink to the original file.

    Args:
        root_dir: Root directory to scan for files.
        extension: File extension filter (None = all files).
        batch_size: Number of files per batch.
    """

    def __init__(self, root_dir: str, extension: Optional[str] = None, batch_size: int = 1):
        self.root_dir = root_dir
        self.extension = extension
        self.batch_size = batch_size
        self.file_paths = _get_all_file_paths(root_dir, extension)

    def __iter__(self):
        batch = []
        for fpath in self.file_paths:
            batch.append(Path(fpath))
            if len(batch) == self.batch_size:
                tensor = make_tensor(batch, self.root_dir, symlink=True)
                tensor.requires_grad_(False)
                yield tensor
                batch = []
        if batch:
            tensor = make_tensor(batch, self.root_dir, symlink=True)
            tensor.requires_grad_(False)
            yield tensor

    def __len__(self):
        return (len(self.file_paths) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    import tempfile

    print("Running SoleFileBatchDataLoader tests...\n")

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

    def storage_path(tensor, flat_index):
        digits = list(str(flat_index))
        return os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )

    # Test 1: Basic loading
    print("Test 1: Basic loading")
    with tempfile.TemporaryDirectory() as tmpdir:
        files = {"a.txt": "Hello", "b.txt": "World"}
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        loader = SoleFileBatchDataLoader(tmpdir, batch_size=2)
        batches = list(loader)
        run_test("1 batch", len(batches) == 1, 1, len(batches))
        run_test("Shape [2]", list(batches[0].shape) == [2])
        run_test("st_relative_to", batches[0].st_relative_to == tmpdir)
        run_test("Has st_tensor_uid", hasattr(batches[0], "st_tensor_uid"))

    # Test 2: Content via symlinks
    print("Test 2: Content via symlinks")
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "hello.txt"), "w") as f:
            f.write("hello content")

        loader = SoleFileBatchDataLoader(tmpdir, batch_size=1)
        batch = next(iter(loader))
        run_test("Is symlink", os.path.islink(storage_path(batch, 0)))
        run_test("Content readable", read_storage(batch, 0) == "hello content")

    # Test 3: Extension filter
    print("Test 3: Extension filter")
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["a.txt", "b.py", "c.txt"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(name)

        loader = SoleFileBatchDataLoader(tmpdir, extension=".txt", batch_size=10)
        batches = list(loader)
        run_test("1 batch", len(batches) == 1)
        run_test("Shape [2]", list(batches[0].shape) == [2], [2], list(batches[0].shape))

    # Test 4: Multiple batches with remainder
    print("Test 4: Multiple batches")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            with open(os.path.join(tmpdir, f"{i}.txt"), "w") as f:
                f.write(f"file {i}")

        loader = SoleFileBatchDataLoader(tmpdir, batch_size=2)
        batches = list(loader)
        run_test("3 batches", len(batches) == 3, 3, len(batches))
        run_test("Batch 0 shape [2]", list(batches[0].shape) == [2])
        run_test("Batch 2 shape [1]", list(batches[2].shape) == [1])
        run_test("__len__ == 3", len(loader) == 3)

    # Test 5: Subdirectories
    print("Test 5: Subdirectories")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "sub"))
        with open(os.path.join(tmpdir, "top.txt"), "w") as f:
            f.write("top")
        with open(os.path.join(tmpdir, "sub", "nested.txt"), "w") as f:
            f.write("nested")

        loader = SoleFileBatchDataLoader(tmpdir, batch_size=10)
        batch = next(iter(loader))
        run_test("Shape [2]", list(batch.shape) == [2])
        contents = {read_storage(batch, i) for i in range(2)}
        run_test("Contains top", "top" in contents)
        run_test("Contains nested", "nested" in contents)

    # Test 6: Empty directory
    print("Test 6: Empty directory")
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = SoleFileBatchDataLoader(tmpdir, batch_size=1)
        batches = list(loader)
        run_test("No batches", len(batches) == 0)
        run_test("__len__ == 0", len(loader) == 0)

    print("\nAll tests completed.")
