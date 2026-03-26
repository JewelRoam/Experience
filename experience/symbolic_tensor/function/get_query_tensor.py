import os
import itertools
import tempfile
import torch
from typing import Callable, Dict, Optional

from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def _copy_back_to_storage_view(mutable_dir: str, output: torch.Tensor) -> None:
    """Copy results from workspace view back to tensor storage."""
    coords_list = [list(coord) for coord in itertools.product(*[range(s) for s in output.size()])]
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, output.stride()))
        digits = list(str(flat_index))
        storage_path = os.path.join(
            output.st_relative_to,
            output.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        if coords:
            coord_dirs = os.path.join(*[str(c) for c in coords])
            view_file = os.path.join(mutable_dir, coord_dirs, "data.txt")
        else:
            view_file = os.path.join(mutable_dir, "data.txt")
        if os.path.isfile(view_file):
            with open(view_file, "r", encoding="utf-8") as f:
                content = f.read()
            with open(storage_path, "w", encoding="utf-8") as f:
                f.write(content)


def default_prompt_for_query(
    task_prompt: str,
    input_view_dir: str,
    mutable_output_dir: str,
) -> str:
    return (
        "You are a semantic grep keyword generator.\n\n"
        f"{task_prompt}\n\n"
        f"Given the symbolic tensor view in \"{input_view_dir}\",\n"
        f"please generate the query keywords into corresponding files in \"{mutable_output_dir}\".\n"
        "Each line in an output file should contain only a single summary keyword "
        "that would be used for grep/query-like operations.\n"
        "The keywords of files are used for calculating similarity between files.\n"
        "All \"TODO\" in output files should be replaced with keywords.\n"
        "The length of keywords list should be kept small for it's a semantic summary.\n"
    )


def get_query_tensor(
    input: torch.Tensor,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "coding_agent",
    llm_env: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """
    Generate a query keyword tensor from an input symbolic tensor.

    Creates a TODO-filled output tensor matching the input shape, dumps both
    as views, then invokes a task handler to replace each TODO file
    with grep/query keywords (one keyword per line).

    Args:
        input: A symbolic tensor with st_relative_to and st_tensor_uid attributes.
        query_prompt: Callable that builds the prompt. None uses default.
        task_prompt: High-level task description.
        llm_method: LLM method to use ("coding_agent" or "raw_llm_api").
        llm_env: Environment variable dict for LLM client. None uses os.environ defaults.

    Returns:
        The output symbolic tensor whose storage files now contain keywords.
    """
    output = todo_tensor_like(input)

    # Full slice copy of output for LLM to write to freely
    full_slices = [slice(None)] * len(output.size())
    copied_output = slice_tensor(output, full_slices)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_view_dir = os.path.join(tmp_dir, "input_view")
        output_view_dir = os.path.join(tmp_dir, "mutable_output_dir")

        dump_view(input, input_view_dir, "txt")
        dump_view(copied_output, output_view_dir, "txt")

        prompt = (query_prompt or default_prompt_for_query)(
            task_prompt, input_view_dir, output_view_dir,
        )

        agent_task = AgentTask(
            workspace_dir=tmp_dir,
            output_relative_dir="mutable_output_dir",
            prompt=prompt,
        )
        TaskHandler()([agent_task], llm_method, llm_env=llm_env)

        # Copy back results to output tensor storage
        _copy_back_to_storage_view(output_view_dir, output)

    return output


if __name__ == "__main__":
    import subprocess
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor

    # Source env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running get_query_tensor tests...\n")

    data = ["def hello():\n    print('hello world')", "class Foo:\n    pass"]

    for method in ["coding_agent", "raw_llm_api"]:
        print(f"--- method={method} ---")
        with tempfile.TemporaryDirectory() as tmpdir:
            t = make_tensor(data, tmpdir)
            print(f"  Input shape: {list(t.shape)}")

            result_tensor = get_query_tensor(t, llm_method=method)
            print(f"  Output shape: {list(result_tensor.shape)}")

            assert list(result_tensor.shape) == list(t.shape), "Shape mismatch"

            root = os.path.join(tmpdir, result_tensor.st_tensor_uid, "storage")
            for i in range(result_tensor.numel()):
                digits = list(str(i))
                path = os.path.join(root, os.path.join(*digits), "data")
                if os.path.isfile(path):
                    with open(path) as f:
                        content = f.read()
                    assert "TODO" not in content, f"Output {i} still contains TODO"
                    print(f"  Output {i}: {repr(content)}")
                else:
                    print(f"  Output {i}: FILE MISSING")

        print()

    # Test with custom prompt
    print("--- custom prompt ---")
    def my_prompt(task_prompt, input_view_dir, mutable_output_dir):
        return (
            f"{task_prompt}\n\n"
            f"Read files in \"{input_view_dir}\" and write keywords to \"{mutable_output_dir}\".\n"
            "One keyword per line. Replace TODO.\n"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        t = make_tensor(["import torch"], tmpdir)
        result_tensor = get_query_tensor(
            t, query_prompt=my_prompt, task_prompt="Extract Python keywords",
            llm_method="raw_llm_api",
        )
        root = os.path.join(tmpdir, result_tensor.st_tensor_uid, "storage")
        path = os.path.join(root, "0", "data")
        if os.path.isfile(path):
            with open(path) as f:
                print(f"  Custom prompt output: {repr(f.read())}")

    print("\nAll tests passed.")
