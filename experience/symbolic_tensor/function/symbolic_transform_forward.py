import os
import tempfile
import itertools
import shutil
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.symbolic_tensor.function.get_query_tensor import get_query_tensor, default_prompt_for_query
from experience.symbolic_tensor.function.select_qkv_indexes import select_qkv_indexes
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _replace_last_tensor_with_full_slice(
    index_tensors: List[torch.Tensor],
    last_dim_size: int,
) -> List[Union[torch.Tensor, slice]]:
    """Replace the last index tensor with a full slice to keep all q/k/v."""
    result: List[Union[torch.Tensor, slice]] = list(index_tensors[:-1])
    result.append(slice(None))  # full slice on last dim (q, k, v)
    return result


def _read_file_content(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the file content for a given flat index from tensor storage."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_nested_result(flat_results: List[Any], shape: List[int]) -> Any:
    """Reshape a flat list of results into a nested list matching the given shape."""
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
    """Copy LLM results from mutable workspace dir back through view tensor's symlinks.

    The mutable dir contains files written by the LLM (independent copies, no symlinks).
    The view_tensor was created by slice_view and has symlink storage pointing to the
    parent tensor. We resolve symlinks to find the real parent storage path.
    """
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
    const_experience_view: str,
    const_input_view: str,
    mutable_output_dir: str,
) -> str:
    """Default prompt for the forward pass (semantic translation).

    Args:
        task_prompt: High-level task description (e.g. "Translate Python To Viba").
        workspace_dir: Root workspace directory.
        const_experience_view: Path to read-only experience QKV view.
        const_input_view: Path to read-only input view.
        mutable_output_dir: Path to mutable output directory (LLM writes here).

    Returns:
        The prompt string for the LLM agent.
    """
    return (
        "You are a semantic translator.\n\n"
        f"{task_prompt}\n\n"
        "Experience mappings are defined as:\n"
        "  1) file \"<root_dir>/<experience_coordinate>.../0/data.xxx\" means query file of <experience_coordinate>...\n"
        "  2) file \"<root_dir>/<experience_coordinate>.../1/data.xxx\" means key file of <experience_coordinate>...\n"
        "  3) file \"<root_dir>/<experience_coordinate>.../2/data.xxx\" means value file of <experience_coordinate>...\n\n"
        "You need read all the key => value pairs to get the experiences.\n"
        "The value files show the EXACT target format and syntax.\n"
        "You MUST faithfully follow the syntax patterns, structure, and style from the value files.\n"
        "Do NOT invent your own syntax or formatting — copy the patterns from the experience values.\n\n"
        f"Conducted by \"{const_experience_view}\",\n"
        f"please translate source semantic text \"{const_input_view}\"\n"
        f"to target semantic text \"{mutable_output_dir}\".\n\n"
        f"Replace TODO in \"{mutable_output_dir}\" with target semantic text.\n"
    )


def symbolic_transform_forward(
    input: torch.Tensor,
    experience: torch.Tensor,
    output_prompt: Optional[Callable[..., str]] = None,
    query_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    topk: int = 16,
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> Tuple[torch.Tensor, Any]:
    """
    Forward pass of the symbolic transform: translate input to output using
    experience (q/k/v mappings) and an LLM coding agent.

    For each scalar element of the input tensor:
      1. Generate query keywords from input
      2. Select top-k experience entries by Jaccard similarity
      3. Slice experience to get relevant q/k/v mappings
      4. Dump views of experience, input element, and TODO output element
      5. Build a SymbolicTransformRequest

    Then dispatch all requests to GenerateOutput[method] for batch LLM processing,
    and copy results back to tensor storage.

    Args:
        input: A symbolic tensor to translate.
        experience: An Experience tensor (last dim=3: query, key, value).
        output_prompt: Callable that builds the forward prompt. Receives
            (workspace_dir, const_experience_view, const_input_view, mutable_output_dir)
            and returns a prompt string. None uses default_prompt_for_output.
        topk: Number of top experience entries to use per element.
        llm_method: LLM backend to use (default "raw_llm_api").

    Returns:
        A tuple of:
        - output: The translated symbolic tensor (same shape as input).
        - selected_experience_qkv_indexes_list: A nested list matching input's
          shape, where each leaf is a list[torch.Tensor[int]] of selected
          experience coordinates.
    """
    # Create TODO-filled output tensor
    output = todo_tensor_like(input)

    # Generate input query keywords
    input_query = get_query_tensor(input, query_prompt=query_prompt, task_prompt=task_prompt, llm_method=llm_method, llm_env=llm_env)

    # Phase 1: Build requests per scalar element
    coords_list = _scalar_slice_indices(input.size())
    flat_selected_indexes: List[List[torch.Tensor]] = []
    flat_tasks: List[AgentTask] = []
    flat_copyback_info: List[Tuple[str, torch.Tensor]] = []

    for coords in coords_list:
        int_slices = [c for c in coords]

        # Create scalar views: view (symlink for copy-back) + value (copy for LLM)
        scalar_input_view = slice_view(input, int_slices)
        scalar_output_view = slice_view(output, int_slices)
        scalar_output_value = slice_tensor(output, int_slices)

        # Read the input file content for this element
        stride = input.stride()
        flat_index = sum(c * s for c, s in zip(coords, stride))
        batch_input_file_content = _read_file_content(input_query, flat_index)
        query_key_words = [w for w in batch_input_file_content.strip().split("\n") if w.strip()]
        # print(f"{query_key_words=}")
        # Select top-k experience entries by similarity
        select_experience_query_indexes = select_qkv_indexes(
            experience, query_key_words, topk
        )
        # Record selected indexes (list of tensors, one per dim)
        flat_selected_indexes.append(select_experience_query_indexes)
        # Replace last index tensor with full slice to keep q/k/v together
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            select_experience_query_indexes, experience.size()[-1]
        )
        # print(f"{select_experience_indexes=}")

        # Slice experience to get relevant entries
        experience_sliced_view = slice_view(experience, select_experience_indexes)

        # Create workspace and dump views
        workspace_dir = tempfile.mkdtemp()
        exp_view_dir = os.path.join(workspace_dir, "const_experiance_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_dir = os.path.join(workspace_dir, "mutable_output_dir")

        dump_view(experience_sliced_view, exp_view_dir, "txt")
        dump_view(scalar_input_view, input_view_dir, "txt")
        # Dump the copy (not the view) — LLM writes here freely
        dump_view(scalar_output_value, output_dir, "txt")

        prompt = (output_prompt or default_prompt_for_output)(
            task_prompt, workspace_dir, exp_view_dir, input_view_dir, output_dir,
        )

        flat_tasks.append(AgentTask(workspace_dir=workspace_dir, output_relative_dir="mutable_output_dir", prompt=prompt))
        flat_copyback_info.append((output_dir, scalar_output_view))

    # Phase 2: Dispatch to TaskHandler
    all_tasks = _build_nested_result(flat_tasks, list(input.size()))

    TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    # Phase 3: Copy back results and cleanup
    for output_dir, scalar_output_view in flat_copyback_info:
        _copy_back_to_storage_view(output_dir, scalar_output_view)

    for task in flat_tasks:
        shutil.rmtree(task.workspace_dir, ignore_errors=True)

    # Build nested structure matching input shape
    selected_experience_qkv_indexes_list = _build_nested_result(
        flat_selected_indexes, list(input.size())
    )

    return output, selected_experience_qkv_indexes_list


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

    print("Running symbolic_transform_forward test...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    print("Test 1: English to French translation (llm_method=coding_agent)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello world in English"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        print(f"  Experience shape: {list(experience_tensor.shape)}")

        output, selected_indexes = symbolic_transform_forward(
            input_tensor, experience_tensor,
            output_prompt=None,
            topk=2,
            llm_method="coding_agent",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Selected indexes is a list", isinstance(selected_indexes, list))

        # Read output storage
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        for i in range(output.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            run_test(f"Output file {i} exists", os.path.isfile(path))
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                run_test(f"Output {i} not TODO", "TODO" not in content)
                print(f"  Output {i}: {repr(content)}")

    print("\nTest 2: English to French translation (llm_method=raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello world in English"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        print(f"  Experience shape: {list(experience_tensor.shape)}")

        output, selected_indexes = symbolic_transform_forward(
            input_tensor, experience_tensor,
            output_prompt=None,
            topk=2,
            llm_method="raw_llm_api",
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Selected indexes is a list", isinstance(selected_indexes, list))

        # Read output storage
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        for i in range(output.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            run_test(f"Output file {i} exists", os.path.isfile(path))
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                run_test(f"Output {i} not TODO", "TODO" not in content)
                print(f"  Output {i}: {repr(content)}")

    print("\nAll tests completed.")
