import os
import tempfile
import torch
from typing import Optional, Type

from experience.symbolic_tensor.function.merge_forward import merge_forward
from experience.symbolic_tensor.function.merge_backward import merge_backward
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.fs_util.text_merger import TextMerger


class Merge(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        axis: int = -1,
        text_merger: Optional[Type] = None,
    ) -> torch.Tensor:
        output = merge_forward(input, axis, text_merger)

        # Save tensors for backward
        ctx.save_for_backward(input, output)
        # save_for_backward strips custom attributes; preserve them manually
        ctx.st_attrs = {}
        for name, tensor in [("input", input), ("output", output)]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        # Save non-tensor state
        ctx.axis = axis
        ctx.text_merger = text_merger

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors

        # Restore custom st_* attributes stripped by save_for_backward
        for name, tensor in [("input", input), ("output", output)]:
            for attr, val in ctx.st_attrs[name].items():
                setattr(tensor, attr, val)

        # Check if a symbolic gradient was registered by an upstream backward
        symbolic_grad = symbolic_grad_registry.pop(output.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            # No registered symbolic grad and no st_* attrs: wrap as TODO
            symbolic_grad_output = todo_tensor_like(output)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        grad_input = merge_backward(
            grad_output, input, output,
            ctx.axis, ctx.text_merger,
        )

        # Register symbolic grad keyed by input tensor uid
        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)

        # Return grads for (input, axis, text_merger)
        return grad_input, None, None


merge = Merge.apply


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
    from experience.fs_util.text_merger import kFrameMarker

    print("Running Merge (autograd.Function) tests...\n")

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
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Forward pass via Merge.apply
    print("Test 1: Forward via Merge.apply")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a\n", "b\n", "c\n"], ["d\n", "e\n", "f\n"]], tmpdir)
        inp.requires_grad_(True)

        output = Merge.apply(inp, -1)

        run_test("Output shape [2]", list(output.shape) == [2])
        run_test("Output has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("Output has st_tensor_uid", hasattr(output, "st_tensor_uid"))
        content0 = read_storage(output, 0)
        run_test("output[0] not None", content0 is not None)
        run_test("output[0] has 3 markers", content0.count(kFrameMarker) == 3)
        frames0 = TextMerger.unpack(content0)
        run_test("output[0] 3 frames", len(frames0) == 3)
        run_test("frame[0] = 'a'", frames0[0][2] == "a")
        run_test("frame[2] = 'c'", frames0[2][2] == "c")

    # Test 2: merge shorthand
    print("Test 2: merge shorthand")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["x\n", "y\n"], tmpdir)

        output = merge(inp, 0)
        run_test("Output is scalar-like", output.numel() == 1)
        content = read_storage(output, 0)
        run_test("content not None", content is not None)
        frames = TextMerger.unpack(content)
        run_test("2 frames", len(frames) == 2)

    # Test 3: Forward with axis=0
    print("Test 3: Forward axis=0")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a\n", "b\n"], ["c\n", "d\n"]], tmpdir)
        inp.requires_grad_(True)

        output = Merge.apply(inp, 0)

        run_test("Output shape [2]", list(output.shape) == [2])
        content0 = read_storage(output, 0)
        frames0 = TextMerger.unpack(content0)
        run_test("col 0: 2 frames", len(frames0) == 2)
        run_test("col 0 frame[0] = 'a'", frames0[0][2] == "a")
        run_test("col 0 frame[1] = 'c'", frames0[1][2] == "c")

    # Test 4: Forward + backward (direct)
    print("Test 4: Forward + backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha\n", "beta\n"], ["gamma\n", "delta\n"]], tmpdir)
        inp.requires_grad_(True)

        output = Merge.apply(inp, -1)
        run_test("Output shape [2]", list(output.shape) == [2])

        # Create modified output: change "alpha" to "ALPHA" in row 0
        out_0 = read_storage(output, 0)
        out_1 = read_storage(output, 1)
        modified_0 = out_0.replace("alpha", "ALPHA")
        modified_output = make_tensor([modified_0, out_1], tmpdir)
        grad_out = get_diff_tensor(output, modified_output)

        grad_input = merge_backward(grad_out, inp, output, axis=-1)

        run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])
        gi_00 = read_storage(grad_input, 0)
        run_test("grad_input[0,0] has diff", gi_00 is not None and "---" in gi_00)
        run_test("grad_input[0,0] +ALPHA", "+ALPHA" in gi_00 if gi_00 else False)
        gi_01 = read_storage(grad_input, 1)
        run_test("grad_input[0,1] empty diff", gi_01 is not None and "---" not in gi_01)
        gi_10 = read_storage(grad_input, 2)
        run_test("grad_input[1,0] empty diff", gi_10 is not None and "---" not in gi_10)

    # Test 5: Coefficient propagation through autograd
    print("Test 5: Coefficient propagation")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x\n", "y\n"]], tmpdir)
        inp.requires_grad_(True)

        output = Merge.apply(inp, -1)  # shape (1,)
        run_test("Output coeff = 2.0",
                 abs(output.data.flatten()[0].item() - 2.0) < 1e-5)

    # Test 6: Empty input
    print("Test 6: Single element axis")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only\n"]], tmpdir)
        inp.requires_grad_(True)

        output = Merge.apply(inp, -1)
        run_test("Output shape [1]", list(output.shape) == [1])
        content = read_storage(output, 0)
        frames = TextMerger.unpack(content)
        run_test("1 frame", len(frames) == 1)
        run_test("content = 'only'", frames[0][2] == "only")

    # Test 7: Custom text_merger
    print("Test 7: Custom text_merger")

    class SimpleMerger:
        @staticmethod
        def pack(frames):
            return " | ".join(f"{idx}:{text}" for idx, _, text in frames)

        @staticmethod
        def unpack(merged):
            parts = merged.split(" | ")
            result = []
            for p in parts:
                idx_s, text = p.split(":", 1)
                result.append((int(idx_s), 1.0, text))
            return result

    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor(["p\n", "q\n"], tmpdir)

        output = Merge.apply(inp, 0, SimpleMerger)
        content = read_storage(output, 0)
        run_test("custom format", content == "0:p\n | 1:q\n")

    print("\nAll tests completed.")
