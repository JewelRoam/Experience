import os
import subprocess
import tempfile
import torch
from typing import Any, Callable, Optional, Tuple

from experience.symbolic_tensor.function.symbolic_transform_forward import (
    symbolic_transform_forward,
    default_prompt_for_output,
)
from experience.symbolic_tensor.function.symbolic_transform_backward import (
    symbolic_transform_backward,
    default_prompt_for_grad_input,
    default_prompt_for_grad_exp_key,
    default_prompt_for_grad_exp_value,
)
from experience.symbolic_tensor.function.get_query_tensor import default_prompt_for_query
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function import symbolic_grad_registry


class SymbolicTransform(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        experience: torch.Tensor,
        output_prompt: Optional[Callable[..., str]] = None,
        query_prompt: Optional[Callable[..., str]] = None,
        grad_input_prompt: Optional[Callable[..., str]] = None,
        grad_exp_key_prompt: Optional[Callable[..., str]] = None,
        grad_exp_value_prompt: Optional[Callable[..., str]] = None,
        task_prompt: str = "",
        topk: int = 16,
        llm_method: str = "raw_llm_api",
    ) -> Tuple[torch.Tensor, Any]:
        output, selected_experience_qkv_indexes_list = symbolic_transform_forward(
            input, experience, output_prompt, query_prompt, task_prompt, topk, llm_method=llm_method
        )

        # Save tensors for backward
        ctx.save_for_backward(input, output, experience)
        # save_for_backward strips custom attributes; preserve them manually
        ctx.st_attrs = {}
        for name, tensor in [("input", input), ("output", output), ("experience", experience)]:
            attrs = {}
            for attr in ("st_relative_to", "st_tensor_uid"):
                if hasattr(tensor, attr):
                    attrs[attr] = getattr(tensor, attr)
            ctx.st_attrs[name] = attrs
        # Save non-tensor state
        ctx.selected_experience_qkv_indexes_list = selected_experience_qkv_indexes_list
        ctx.grad_input_prompt = grad_input_prompt
        ctx.grad_exp_key_prompt = grad_exp_key_prompt
        ctx.grad_exp_value_prompt = grad_exp_value_prompt
        ctx.task_prompt = task_prompt
        ctx.topk = topk
        ctx.llm_method = llm_method

        return output, selected_experience_qkv_indexes_list

    @staticmethod
    def backward(ctx, grad_output, grad_selected_indexes=None):
        input, output, experience = ctx.saved_tensors

        # Restore custom st_* attributes stripped by save_for_backward
        for name, tensor in [("input", input), ("output", output), ("experience", experience)]:
            for attr, val in ctx.st_attrs[name].items():
                setattr(tensor, attr, val)

        # Check if a symbolic gradient was registered by an upstream backward
        # (autograd strips st_* attrs when passing gradients between Function nodes)
        # Look up by output tensor's uid (registered by GetEditDistanceRatio.backward)
        symbolic_grad = symbolic_grad_registry.pop(output.st_tensor_uid)
        if symbolic_grad is not None:
            grad_output = symbolic_grad
        elif not hasattr(grad_output, "st_relative_to"):
            # No registered symbolic grad and no st_* attrs: wrap as TODO
            symbolic_grad_output = todo_tensor_like(output)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output,
            input,
            output,
            experience,
            ctx.selected_experience_qkv_indexes_list,
            ctx.grad_input_prompt,
            ctx.grad_exp_key_prompt,
            ctx.grad_exp_value_prompt,
            ctx.task_prompt,
            ctx.topk,
            llm_method=ctx.llm_method,
        )

        # Register symbolic grads keyed by the original parameter tensor uids
        # so the optimizer can retrieve them (autograd strips st_* attrs)
        if grad_input is not None:
            symbolic_grad_registry.register(input.st_tensor_uid, grad_input)
        symbolic_grad_registry.register(experience.st_tensor_uid, grad_experience)

        # Return grads for (input, experience, output_prompt, query_prompt, grad_input_prompt,
        #                    grad_exp_key_prompt, grad_exp_value_prompt, task_prompt, topk, llm_method)
        return grad_input, grad_experience, None, None, None, None, None, None, None, None


symbolic_transform = SymbolicTransform.apply


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running SymbolicTransform (autograd.Function) tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Forward pass via .apply()
    print("Test 1: Forward pass via SymbolicTransform.apply")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello world in English"]
        input_tensor = make_tensor(input_data, tmpdir)
        input_tensor.requires_grad_(True)

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        output, selected_indexes = SymbolicTransform.apply(
            input_tensor, experience_tensor,
            None,  # output_prompt
            None,  # query_prompt
            None,  # grad_input_prompt
            None,  # grad_exp_key_prompt
            None,  # grad_exp_value_prompt
            2,     # topk
            "raw_llm_api",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Selected indexes returned", selected_indexes is not None)
        run_test("Output requires grad", output.requires_grad)

        # Check output content
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        path = os.path.join(root, "0", "data")
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
            run_test("Output not TODO", "TODO" not in content)
            print(f"  Output: {repr(content[:120])}")

    # Test 2: Forward + backward (direct call)
    print("\nTest 2: Forward + backward (direct call)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world in English"], tmpdir)

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        output, selected_indexes = symbolic_transform_forward(
            input_tensor, experience_tensor,
            output_prompt=None,
            topk=2,
            llm_method="raw_llm_api",
        )

        run_test("Output has st attrs", hasattr(output, "st_relative_to"))

        # Construct a symbolic grad_output with text diff
        grad_output = make_tensor(
            ["The translation should use formal French: 'Bonjour le monde' -> 'Bonjour au monde'"],
            tmpdir,
        )
        grad_output.data.fill_(1.0)

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output, input_tensor, output, experience_tensor,
            selected_experience_qkv_indexes_list=selected_indexes,
            topk=2,
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape matches", list(grad_input.shape) == list(input_tensor.shape))
        run_test("grad_experience shape matches", list(grad_experience.shape) == list(experience_tensor.shape))

        # Check grad_input text diff
        root = os.path.join(tmpdir, grad_input.st_tensor_uid, "storage")
        path = os.path.join(root, "0", "data")
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
            run_test("grad_input text not TODO", "TODO" not in content)
            print(f"  grad_input text: {repr(content[:120])}")

        # Check grad_experience text diffs
        root = os.path.join(tmpdir, grad_experience.st_tensor_uid, "storage")
        for i in range(grad_experience.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                print(f"  grad_experience[{i}]: TODO={'TODO' == content.strip()} {repr(content[:80])}")

        # Numeric channel
        run_test("grad_input coeff == 1.0", grad_input.data[0].item() == 1.0)
        run_test("grad_experience coeff accumulated", grad_experience.data[0, 0].item() == 1.0)

    print("\nAll tests completed.")
