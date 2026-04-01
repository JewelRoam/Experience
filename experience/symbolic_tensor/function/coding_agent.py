import os
import tempfile
import itertools
import shutil
import torch
from typing import Callable, Dict, List, Optional, Tuple

from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _build_nested_result(flat_results: List, shape: List[int]):
    """Reshape a flat list into a nested list matching the given shape."""
    if not shape:
        return flat_results[0]
    if len(shape) == 1:
        return flat_results
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _build_nested_result(flat_results[i * chunk_size:(i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]


def _copy_back_to_storage_view(mutable_dir: str, view_tensor: torch.Tensor) -> None:
    """Copy LLM results from mutable workspace dir back through view tensor's symlinks."""
    coords_list = [list(coord) for coord in itertools.product(*[range(s) for s in view_tensor.size()])]
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, view_tensor.stride()))
        digits = list(str(flat_index))
        view_storage_path = os.path.join(
            view_tensor.st_relative_to,
            view_tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        real_storage_path = os.path.realpath(view_storage_path)
        if coords:
            coord_dirs = os.path.join(*[str(c) for c in coords])
            mutable_file = os.path.join(mutable_dir, coord_dirs, "data.txt")
        else:
            mutable_file = os.path.join(mutable_dir, "data.txt")
        if os.path.isfile(mutable_file):
            with open(mutable_file, "r", encoding="utf-8") as f:
                content = f.read()
            with open(real_storage_path, "w", encoding="utf-8") as f:
                f.write(content)


def default_prompt_for_output(
    task_prompt: str,
    workspace_dir: str,
    const_input_view: str,
    mutable_output_dir: str,
) -> str:
    return (
        "You are a coding agent.\n\n"
        f"{task_prompt}\n\n"
        f"Input: \"{const_input_view}\"\n"
        f"Output: \"{mutable_output_dir}\"\n\n"
        f"Replace TODO in \"{mutable_output_dir}\" with result.\n"
    )


def coding_agent(
    input: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Forward pass of coding agent: input → prompt → LLM → output.

    Mini version of st_moe_forward without autograd or experience.

    Args:
        input: A symbolic tensor to process.
        output_prompt: Callable(task_prompt, workspace_dir, const_input_view,
            mutable_output_dir) -> str. None uses default_prompt_for_output.
        task_prompt: High-level task description.
        llm_method: LLM backend ("coding_agent" or "raw_llm_api").
        llm_env: Optional environment variables for LLM.

    Returns:
        output: The processed symbolic tensor (same shape as input).
    """
    output = todo_tensor_like(input)

    coords_list = _scalar_slice_indices(input.size())
    flat_tasks: List[AgentTask] = []
    flat_copyback_info: List[Tuple[str, torch.Tensor]] = []

    for coords in coords_list:
        scalar_input_view = slice_view(input, coords)
        scalar_output_view = slice_view(output, coords)
        scalar_output_value = slice_tensor(output, coords)

        workspace_dir = tempfile.mkdtemp()
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_dir = os.path.join(workspace_dir, "mutable_output_dir")

        dump_view(scalar_input_view, input_view_dir, "txt")
        dump_view(scalar_output_value, output_dir, "txt")

        prompt = (output_prompt or default_prompt_for_output)(
            task_prompt, workspace_dir, input_view_dir, output_dir,
        )

        flat_tasks.append(AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir="mutable_output_dir",
            prompt=prompt,
        ))
        flat_copyback_info.append((output_dir, scalar_output_view))

    all_tasks = _build_nested_result(flat_tasks, list(input.size()))
    TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    for output_dir, scalar_output_view in flat_copyback_info:
        _copy_back_to_storage_view(output_dir, scalar_output_view)

    for task in flat_tasks:
        shutil.rmtree(task.workspace_dir, ignore_errors=True)

    return output


if __name__ == "__main__":
    import subprocess
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

    print("Running coding_agent tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_output(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Simple coding agent (raw_llm_api)
    print("Test 1: Simple task (llm_method=raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Write a Python hello world one-liner"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        output = coding_agent(
            input_tensor,
            task_prompt="You are a Python expert. Write concise code.",
            llm_method="raw_llm_api",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        content = read_output(output, 0)
        run_test("Output file exists", content is not None)
        if content:
            run_test("Output not TODO", "TODO" not in content)
            print(f"  Output: {repr(content)}")

    # Test 2: Simple coding agent (coding_agent)
    print("Test 2: Simple task (llm_method=coding_agent)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Write a Python hello world one-liner"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        output = coding_agent(
            input_tensor,
            task_prompt="You are a Python expert. Write concise code.",
            llm_method="coding_agent",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        content = read_output(output, 0)
        run_test("Output file exists", content is not None)
        if content:
            run_test("Output not TODO", "TODO" not in content)
            print(f"  Output: {repr(content)}")

    # Test 3: Multi-element tensor
    print("Test 3: Multi-element (llm_method=raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["sum of 1+2", "product of 3*4"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        output = coding_agent(
            input_tensor,
            task_prompt="Compute the arithmetic result. Output only the number.",
            llm_method="raw_llm_api",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        for i in range(output.numel()):
            content = read_output(output, i)
            run_test(f"Output[{i}] exists", content is not None)
            if content:
                run_test(f"Output[{i}] not TODO", "TODO" not in content)
                print(f"  Output[{i}]: {repr(content)}")

    # Test 4: Custom prompt
    print("Test 4: Custom prompt (llm_method=raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello"]
        input_tensor = make_tensor(input_data, tmpdir)

        def custom_prompt(task_prompt, workspace_dir, const_input_view, mutable_output_dir):
            return (
                f"{task_prompt}\n\n"
                f"Read input from \"{const_input_view}\".\n"
                f"Write the uppercase version to \"{mutable_output_dir}\".\n"
                f"Replace TODO with result.\n"
            )

        output = coding_agent(
            input_tensor,
            output_prompt=custom_prompt,
            task_prompt="Convert text to uppercase.",
            llm_method="raw_llm_api",
        )

        run_test("Output shape matches", list(output.shape) == list(input_tensor.shape))
        content = read_output(output, 0)
        run_test("Output exists", content is not None)
        if content:
            run_test("Output not TODO", "TODO" not in content)
            print(f"  Output: {repr(content)}")

    print("\nAll tests completed.")
