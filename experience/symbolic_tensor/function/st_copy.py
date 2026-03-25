import os
import tempfile
import torch
from torch.autograd import Function

from experience.symbolic_tensor.tensor_util.dump_tensor import dump_tensor
from experience.symbolic_tensor.tensor_util.load_tensor import load_tensor


def copy_impl(input_tensor: torch.Tensor, dst_relative_to: str) -> torch.Tensor:
    """
    Copy a symbolic tensor to a new location with a new uid.

    Dumps input_tensor's storage to a temp dir, then loads it into dst_relative_to,
    producing a new tensor with independent storage and a fresh uid.

    Args:
        input_tensor: A symbolic tensor with st_relative_to and st_tensor_uid.
        dst_relative_to: Destination root directory for the copied tensor.

    Returns:
        A new symbolic tensor with the same shape/content under dst_relative_to.
    """
    with tempfile.TemporaryDirectory() as tmp_dump_dir:
        dump_tensor(input_tensor, tmp_dump_dir)
        return load_tensor(tmp_dump_dir, dst_relative_to)


class Copy(Function):
    @staticmethod
    def forward(ctx, input_tensor, dst_relative_to):
        ctx.save_for_backward(input_tensor)
        return copy_impl(input_tensor, dst_relative_to)

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        grad_input = copy_impl(grad_output, input_tensor.st_relative_to)
        return grad_input, None


copy = Copy.apply


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running copy tests...\n")

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

    # Test 1: copy_impl forward
    print("Test 1: copy_impl")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        data = ["hello", "world"]
        t = make_tensor(data, src_dir)
        copied = copy_impl(t, dst_dir)

        run_test("Shape matches", list(copied.shape) == list(t.shape))
        run_test("Different uid", copied.st_tensor_uid != t.st_tensor_uid)
        run_test("dst relative_to", copied.st_relative_to == dst_dir)
        run_test("Content 0", read_storage(copied, 0) == "hello")
        run_test("Content 1", read_storage(copied, 1) == "world")

    # Test 2: copy_impl 2D
    print("Test 2: copy_impl 2D")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        data = [["a", "b"], ["c", "d"]]
        t = make_tensor(data, src_dir)
        copied = copy_impl(t, dst_dir)

        run_test("Shape [2,2]", list(copied.shape) == [2, 2])
        run_test("Stride matches", list(copied.stride()) == list(t.stride()))
        for i, content in enumerate(["a", "b", "c", "d"]):
            run_test(f"Content {i} = '{content}'", read_storage(copied, i) == content)

    # Test 3: Copy.backward returns (grad, None)
    print("Test 3: Copy autograd backward")
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as dst_dir, \
         tempfile.TemporaryDirectory() as grad_dir:
        t = make_tensor(["source"], src_dir)
        grad_out = make_tensor(["grad text"], grad_dir)

        # Simulate autograd: forward to populate ctx, then backward
        ctx = type('MockCtx', (), {})()
        ctx.save_for_backward = lambda *ts: setattr(ctx, 'saved_tensors', ts)
        Copy.forward(ctx, t, dst_dir)
        result = Copy.backward(ctx, grad_out)

        run_test("Returns tuple of 2", len(result) == 2)
        run_test("Second is None", result[1] is None)
        run_test("grad relative_to matches input", result[0].st_relative_to == src_dir)
        run_test("grad content copied", read_storage(result[0], 0) == "grad text")

    # Test 4: Modifying copy doesn't affect source
    print("Test 4: Independent storage")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        t = make_tensor(["original"], src_dir)
        copied = copy_impl(t, dst_dir)

        # Modify the copy's storage
        copy_path = os.path.join(
            dst_dir, copied.st_tensor_uid, "storage", "0", "data"
        )
        with open(copy_path, "w") as f:
            f.write("modified")

        run_test("Copy modified", read_storage(copied, 0) == "modified")
        run_test("Source unchanged", read_storage(t, 0) == "original")

    print("\nAll tests completed.")
