import os
import subprocess
import tempfile
import torch
from typing import Callable, Dict, Optional

from experience.symbolic_tensor.function.slice_attention_forward import (
    slice_attention_forward,
)
from experience.symbolic_tensor.function.slice_attention_backward import (
    slice_attention_backward,
)
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry


class SliceAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        attention_mask: torch.Tensor,
        return_view: bool = False,
        grad_input_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        llm_method: str = "raw_llm_api",
        llm_env: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        output = slice_attention_forward(input, attention_mask, return_view)

        # Save tensors for backward
        ctx.save_for_backward(input, output, attention_mask)
        # save_for_backward strips custom attributes; preserve them manually
        ctx.st_attrs = {}
        for name, tensor in [("input", input), ("output", output)]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        # Save non-tensor state
        ctx.grad_input_prompt = grad_input_prompt
        ctx.task_prompt = task_prompt
        ctx.llm_method = llm_method
        ctx.llm_env = llm_env

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, attention_mask = ctx.saved_tensors

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

        grad_input = slice_attention_backward(
            grad_output, input, output, attention_mask,
            ctx.grad_input_prompt, ctx.task_prompt,
            llm_method=ctx.llm_method, llm_env=ctx.llm_env,
        )

        # Register symbolic grad keyed by input tensor uid
        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)

        # Return grads for (input, attention_mask, return_view,
        #                    grad_input_prompt, task_prompt, llm_method, llm_env)
        return grad_input, None, None, None, None, None, None


slice_attention = SliceAttention.apply


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running SliceAttention (autograd.Function) tests...\n")

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

    # Test 1: Forward pass via .apply()
    print("Test 1: Forward pass via SliceAttention.apply")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["hello", "world"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = SliceAttention.apply(inp, mask)

        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])
        run_test("Output has st_relative_to", hasattr(output, "st_relative_to"))
        run_test("Output has st_tensor_uid", hasattr(output, "st_tensor_uid"))
        run_test("[0,0,0]='hello'", read_storage(output, 0) == "hello")
        run_test("[0,1,0]='hello'", read_storage(output, 2) == "hello")
        run_test("[0,1,1]='world'", read_storage(output, 3) == "world")
        run_test("attended are 1.0", output[mask].eq(1.0).all().item())
        run_test("non-attended are 0.0", output[~mask].eq(0.0).all().item())

    # Test 2: Forward with return_view=True
    print("Test 2: Forward with return_view=True")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha", "beta"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = SliceAttention.apply(inp, mask, True)

        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])
        run_test("[0,0,0]='alpha'", read_storage(output, 0) == "alpha")
        run_test("[0,1,0]='alpha'", read_storage(output, 2) == "alpha")
        run_test("[0,1,1]='beta'", read_storage(output, 3) == "beta")

    # Test 3: slice_attention shorthand
    print("Test 3: slice_attention shorthand")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.ones(1, 2, 2, dtype=torch.bool)

        output = slice_attention(inp, mask)
        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])
        run_test("[0,0,0]='a'", read_storage(output, 0) == "a")

    # Test 4: Causal mask
    print("Test 4: Causal mask 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))

        output = slice_attention(inp, mask)
        run_test("shape (1, 3, 3)", list(output.shape) == [1, 3, 3])
        run_test("[0,0,0]='x'", read_storage(output, 0) == "x")
        run_test("[0,1,0]='x'", read_storage(output, 3) == "x")
        run_test("[0,1,1]='y'", read_storage(output, 4) == "y")
        run_test("[0,2,0]='x'", read_storage(output, 6) == "x")
        run_test("[0,2,1]='y'", read_storage(output, 7) == "y")
        run_test("[0,2,2]='z'", read_storage(output, 8) == "z")

    # Test 5: Empty mask
    print("Test 5: Empty mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        mask = torch.zeros(1, 2, 2, dtype=torch.bool)

        output = slice_attention(inp, mask)
        run_test("all zero", (output == 0).all().item())

    # Test 6: Forward + backward with LLM
    print("Test 6: Forward + backward with LLM")

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.LLM_config.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["Hello world", "Goodbye"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tensor([[[True, False], [True, True]]])

        output = SliceAttention.apply(
            inp, mask, False,
            None,  # grad_input_prompt
            "Improve text formality",  # task_prompt
            "raw_llm_api",
        )

        run_test("Output shape (1, 2, 2)", list(output.shape) == [1, 2, 2])

        # Construct symbolic grad_output
        grad_out = make_tensor(
            [[["Make more formal", "unused"],
              ["Keep greeting formal", "Make farewell formal"]]],
            tmpdir,
        )
        grad_out.data[mask] = 1.0

        grad_input = slice_attention_backward(
            grad_out, inp, output, mask,
            task_prompt="Improve text formality",
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])
        for i in range(2):
            gi_text = read_storage(grad_input, i)
            run_test(f"grad_input[{i}] not None", gi_text is not None)
            print(f"  grad_input[{i}]: {repr(gi_text[:80]) if gi_text else 'None'}")

    print("\nAll tests completed.")
