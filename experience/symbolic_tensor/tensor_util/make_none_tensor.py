import uuid
import torch
from typing import List

def make_none_like_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return make_none_tensor(tensor.shape, tensor.st_relative_to, tensor.dtype)

def make_none_tensor(shape: List[int], relative_to: str, dtype=torch.bfloat16) -> torch.Tensor:
    """
    Create a zero-filled symbolic tensor with metadata attributes.

    Args:
        shape: The shape of the tensor to create.
        relative_to: The root directory path to associate with this tensor.
        dtype: The tensor dtype (default: torch.bfloat16).

    Returns:
        A zero-filled torch.Tensor with st_relative_to and st_tensor_uid attributes.
    """
    ret = torch.zeros(shape, dtype=dtype)
    ret.st_relative_to = relative_to
    ret.st_tensor_uid = str(uuid.uuid4())
    return ret


if __name__ == "__main__":
    import tempfile

    print("Running tests for make_none_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            if expected is not None and actual is not None:
                print(f"  expected: {expected}")
                print(f"  actual:   {actual}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Basic creation
        t = make_none_tensor([2, 3, 256], tmpdir)
        run_test("Test1: Shape correct", t.shape == (2, 3, 256), (2, 3, 256), tuple(t.shape))
        run_test("Test1: dtype bfloat16", t.dtype == torch.bfloat16)
        run_test("Test1: All zeros", torch.all(t == 0).item())
        run_test("Test1: st_relative_to set", t.st_relative_to == tmpdir)
        run_test("Test1: st_tensor_uid is string", isinstance(t.st_tensor_uid, str))
        run_test("Test1: st_tensor_uid non-empty", len(t.st_tensor_uid) > 0)

        # Test 2: Each call produces a unique uid
        t2 = make_none_tensor([1], tmpdir)
        run_test("Test2: Unique uids", t.st_tensor_uid != t2.st_tensor_uid)

        # Test 3: Custom dtype
        t3 = make_none_tensor([4, 4], tmpdir, dtype=torch.float32)
        run_test("Test3: Custom dtype float32", t3.dtype == torch.float32)

    print("\nAll tests completed.")
