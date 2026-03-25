import os
import subprocess
import tempfile
import torch
import torch.nn as nn
from typing import List, Tuple, Any

from experience.symbolic_tensor.function.symbolic_transform import symbolic_transform
from experience.symbolic_tensor.tensor_util.make_none_tensor import make_none_tensor


class SymbolicTransformModule(nn.Module):
    """
    torch.nn.Module wrapping symbolic_transform.

    Experience is created inside __init__ like nn.Linear creates its weight.
    Initialized as a zero-coefficient symbolic tensor; load or assign content
    before use.

    Args:
        experience_shape: Shape of the experience tensor, e.g. [num_entries, 3].
        forward_prompt: Prompt describing the transformation task.
        topk: Number of experience entries to select per input element.
    """

    experience: torch.Tensor

    def __init__(
        self,
        experience_shape: List[int],
        forward_prompt: str = "",
        topk: int = 16,
    ):
        super().__init__()
        self.forward_prompt = forward_prompt
        self.topk = topk
        self._experience_dir = tempfile.mkdtemp()
        self.experience = make_none_tensor(experience_shape, self._experience_dir)
        self.experience.requires_grad_(True)

    def parameters(self, recurse: bool = True):
        yield self.experience

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return symbolic_transform(input, self.experience, self.forward_prompt, self.topk)


if __name__ == "__main__":
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.function.st_copy import copy_impl

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

    print("Running SymbolicTransformModule tests...\n")

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

    # Test 1: Construction
    print("Test 1: Construction")
    model = SymbolicTransformModule(
        experience_shape=[2, 3],
        forward_prompt="Translate to French.",
        topk=2,
    )
    run_test("experience shape", list(model.experience.shape) == [2, 3])
    run_test("experience has st_relative_to", hasattr(model.experience, "st_relative_to"))
    run_test("experience has st_tensor_uid", hasattr(model.experience, "st_tensor_uid"))
    run_test("experience requires_grad", model.experience.requires_grad)
    run_test("forward_prompt stored", model.forward_prompt == "Translate to French.")
    run_test("topk stored", model.topk == 2)
    params = list(model.parameters())
    run_test("1 parameter", len(params) == 1)
    run_test("Parameter is experience", params[0] is model.experience)

    # Test 2: Load experience content via copy
    print("Test 2: Load experience content")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir"],
        ]
        src = make_tensor(experience_data, tmpdir)

        model = SymbolicTransformModule(
            experience_shape=[2, 3],
            forward_prompt="Translate to French.",
            topk=2,
        )
        # Load content: copy src storage into model's experience
        loaded = copy_impl(src, model._experience_dir)
        model.experience = loaded
        model.experience.requires_grad_(True)

        run_test("Content loaded", read_storage(model.experience, 0) == "greeting\nhello\nworld")
        run_test("Content[2]", read_storage(model.experience, 2) == "Bonjour le monde")

    # Test 3: Forward pass with loaded experience
    print("Test 3: Forward pass")
    with tempfile.TemporaryDirectory() as tmpdir:
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir"],
        ]
        src = make_tensor(experience_data, tmpdir)

        model = SymbolicTransformModule(
            experience_shape=[2, 3],
            forward_prompt="Translate to French.",
            topk=2,
        )
        loaded = copy_impl(src, model._experience_dir)
        model.experience = loaded
        model.experience.requires_grad_(True)

        input_tensor = make_tensor(["Hello world in English"], tmpdir)
        output, selected_indexes = model(input_tensor)

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Output has st_tensor_uid", hasattr(output, "st_tensor_uid"))
        run_test("Selected indexes returned", selected_indexes is not None)

        output_text = read_storage(output, 0)
        run_test("Output not empty", len(output_text) > 0)
        print(f"  Output: {repr(output_text[:120])}")

    # Test 4: Works with optimizer
    print("Test 4: Compatible with optimizer")
    from experience.symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD

    model = SymbolicTransformModule(
        experience_shape=[2, 3],
        forward_prompt="Translate.",
        topk=1,
    )
    optimizer = SymbolicSGD(model.parameters(), lr=1.0, step_prompt="Update translations.")
    run_test("Optimizer accepts model.parameters()", len(optimizer.param_groups) == 1)
    run_test("Optimizer has 1 param", len(optimizer.param_groups[0]["params"]) == 1)

    print("\nAll tests completed.")
