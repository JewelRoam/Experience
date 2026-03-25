import os
import torch

from experience.symbolic_tensor.llm_client.pack_dir import pack_dir


def pack_tensor(tensor: torch.Tensor) -> str:
    """Pack a symbolic tensor's storage directory into a string representation.

    Builds the tensor directory path from st_relative_to/st_tensor_uid,
    then delegates to pack_dir to produce a repomix-like text dump.

    Args:
        tensor: A symbolic tensor with st_relative_to and st_tensor_uid attributes.

    Returns:
        A string containing the directory structure and all file contents.
    """
    tensor_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid)
    return pack_dir(tensor_dir)


if __name__ == "__main__":
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running pack_tensor tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Basic scalar tensor
    print("Test 1: Scalar tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor("hello world", tmpdir)
        result = pack_tensor(t)
        run_test("Contains directory structure", "# Directories structure" in result)
        run_test("Contains files section", "# Files" in result)
        run_test("Contains tensor content", "hello world" in result)

    # Test 2: 1D tensor
    print("Test 2: 1D tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["alpha", "beta"], tmpdir)
        result = pack_tensor(t)
        run_test("Contains alpha", "alpha" in result)
        run_test("Contains beta", "beta" in result)
        run_test("Contains storage in structure", "storage" in result)

    # Test 3: 2D tensor (experience-like)
    print("Test 3: 2D tensor")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor([["q0", "k0", "v0"], ["q1", "k1", "v1"]], tmpdir)
        result = pack_tensor(t)
        run_test("Contains q0", "q0" in result)
        run_test("Contains v1", "v1" in result)

    # Test 4: Has correct tensor_uid in path
    print("Test 4: Tensor UID")
    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor("test", tmpdir)
        result = pack_tensor(t)
        run_test("Has st_tensor_uid", hasattr(t, "st_tensor_uid"))
        run_test("Result is non-empty string", isinstance(result, str) and len(result) > 0)

    print("\nAll tests completed.")
