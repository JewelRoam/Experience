import os
import shutil
import tempfile
import itertools
import torch
from typing import Any, List, Tuple, Union

from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from symbolic_tensor.tensor_util.empty_tensor_like import empty_tensor_like
from symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from symbolic_tensor.tensor_util.slice_view import slice_view
from symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from symbolic_tensor.tensor_util.dump_view import dump_view
from symbolic_tensor.llm_client.agent_task import AgentTask
from symbolic_tensor.llm_client.task_handler import TaskHandler
from symbolic_tensor.sparse_util.convert_nested_list_coordinates_to_pairs_coordinates import (
    convert_nested_list_coordinates_to_pairs_coordinates,
)
from symbolic_tensor.sparse_util.transpose_pairs_coordinates import (
    transpose_pairs_coordinates,
)


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


def _flatten_nested_indexes(
    nested: Any,
    shape: List[int],
) -> List[List[torch.Tensor]]:
    """Flatten a nested list of index tensor lists matching the given shape."""
    if not shape:
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested_indexes(item, shape[1:]))
    return result


def _pad_indexes_to_topk_with_none_experience_indexes(
    select_experience_query_indexes: List[torch.Tensor],
    topk: int,
    experience: torch.Tensor,
) -> List[torch.Tensor]:
    """Pad index tensors to topk length with zero-index when fewer entries were selected."""
    if not select_experience_query_indexes:
        ndim = len(experience.size())
        return [torch.zeros(topk, dtype=torch.long) for _ in range(ndim)]

    current_len = len(select_experience_query_indexes[0])
    if current_len >= topk:
        return select_experience_query_indexes

    padded = []
    for idx_tensor in select_experience_query_indexes:
        pad_count = topk - current_len
        pad_tensor = torch.zeros(pad_count, dtype=idx_tensor.dtype)
        padded.append(torch.cat([idx_tensor, pad_tensor]))
    return padded


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


def symbolic_transform_backward_grad_input(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    experience: torch.Tensor,
    selected_experience_qkv_indexes_list: Any,
    forward_prompt: str = "",
    topk: int = 16,
    llm_method: str = "raw_llm_api",
) -> Union[torch.Tensor, None]:
    """Compute grad_input by iterating per input scalar element.

    Only runs if input.requires_grad; returns None otherwise.
    """
    if not input.requires_grad:
        return None

    grad_input = todo_tensor_like(input)

    # ── Numeric channel ──
    grad_input.data.copy_(grad_output.data)

    # ── Symbolic channel ──
    input_shape = list(input.size())
    coords_list = _scalar_slice_indices(input.size())
    flat_selected_indexes = _flatten_nested_indexes(
        selected_experience_qkv_indexes_list, input_shape
    )

    # Collect all tasks, then batch-call TaskHandler
    all_tasks = []
    task_contexts = []  # (workspace_dir, grad_input_dir, scalar_grad_input_view)

    for coords, select_experience_query_indexes in zip(coords_list, flat_selected_indexes):
        int_slices = [c for c in coords]

        scalar_grad_output = slice_view(grad_output, int_slices)
        scalar_input = slice_view(input, int_slices)
        scalar_output = slice_view(output, int_slices)
        scalar_grad_input_view = slice_view(grad_input, int_slices)
        scalar_grad_input_value = slice_tensor(grad_input, int_slices)

        padded_indexes = _pad_indexes_to_topk_with_none_experience_indexes(
            select_experience_query_indexes, topk, experience
        )
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            padded_indexes, experience.size()[-1]
        )
        experience_sliced_view = slice_view(experience, select_experience_indexes)

        workspace_dir = tempfile.mkdtemp()
        grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_view_dir = os.path.join(workspace_dir, "const_output_view")
        experience_view_dir = os.path.join(workspace_dir, "const_experience_view")
        grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")

        dump_view(scalar_grad_output, grad_output_view_dir, "txt")
        dump_view(scalar_input, input_view_dir, "txt")
        dump_view(scalar_output, output_view_dir, "txt")
        dump_view(experience_sliced_view, experience_view_dir, "txt")
        dump_view(scalar_grad_input_value, grad_input_dir, "txt")

        prompt = (
            "You are a symbolic gradient calculator for backward pass.\n\n"
            f"{forward_prompt}\n\n"
            "During forward pass, the input was translated to output using experience entries.\n"
            "Now given the output gradient (how output should change), compute gradients for\n"
            "input.\n\n"
            "Context (read-only):\n"
            f"- Output gradient (text diff): \"{grad_output_view_dir}\"\n"
            f"- Original input: \"{input_view_dir}\"\n"
            f"- Original output: \"{output_view_dir}\"\n"
            f"- Experience entries used during forward: \"{experience_view_dir}\"\n"
            "  where .../0/data.xxx = query, .../1/data.xxx = key, .../2/data.xxx = value\n"
            "  Each line in query files only contains one summary key word for current experience,\n"
            "  which is used for calculating similarity between inputs and experience.\n"
            "  Key files contain source domain semantics.\n"
            "  Value files contain target domain semantics.\n\n"
            f"Compute and write:\n"
            f"1. Input gradient in \"{grad_input_dir}\":\n"
            "   How should the input text change to improve the output?\n\n"
            "Replace all TODO with computed Gradient files.\n\n"
            "Gradient files format:\n"
            "1. Gradient files must be like output of `diff -u --label data --label data original.txt modified.txt`.\n"
            "2. Gradient files will be applied by cmd `patch -i backward.diff /forward/location/data`\n"
            f"3. Gradient files of \"{grad_input_dir}/<xxx>/data\" must be able to be applied to \"{input_view_dir}/<xxx>/data\"\n"
        )

        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_input_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        task_contexts.append((workspace_dir, grad_input_dir, scalar_grad_input_view))

    # Batch LLM call
    if all_tasks:
        TaskHandler()(all_tasks, llm_method)

    # Copy results back and clean up
    for workspace_dir, grad_input_dir, scalar_grad_input_view in task_contexts:
        _copy_back_to_storage_view(grad_input_dir, scalar_grad_input_view)
        shutil.rmtree(workspace_dir)

    return grad_input


def symbolic_transform_backward_grad_experience(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    experience: torch.Tensor,
    selected_experience_qkv_indexes_list: Any,
    forward_prompt: str = "",
    topk: int = 16,
    llm_method: str = "raw_llm_api",
) -> torch.Tensor:
    """Compute grad_experience by iterating per experience entry via transposed sparse coordinates.

    Uses convert_nested_list_coordinates_to_pairs_coordinates + transpose_pairs_coordinates
    to invert the forward's index mapping: for each experience entry, gather all input elements
    that used it and present them together to the LLM.
    """
    # Pad each leaf's indexes to topk before transposing.
    # When experience is empty and forward selects nothing, padding fills with
    # index 0 so gradient still flows to experience entries.
    input_shape = list(input.size())
    flat_indexes = _flatten_nested_indexes(
        selected_experience_qkv_indexes_list, input_shape
    )
    padded_flat = [
        _pad_indexes_to_topk_with_none_experience_indexes(idx, topk, experience)
        for idx in flat_indexes
    ]
    padded_nested = _build_nested_result(padded_flat, input_shape)

    # Convert padded indexes to pairs, then transpose
    pairs = convert_nested_list_coordinates_to_pairs_coordinates(padded_nested)
    transposed = transpose_pairs_coordinates(pairs)

    # Create grad_experience with empty strings (unselected entries stay "")
    grad_experience = empty_tensor_like(experience)

    # ── Numeric channel ──
    grad_experience.data.zero_()

    # Collect all tasks, then batch-call TaskHandler
    all_tasks = []
    task_contexts = []  # (workspace_dir, grad_experience_dir, grad_experience_sliced_view)

    for sole_exp_point, multi_output_points in transposed:
        # Convert scalar tensors to ints, replace last dim with full slice for q/k/v
        int_exp_point = [t.item() for t in sole_exp_point]
        select_experience_indexes = int_exp_point[:-1] + [slice(None)]

        # Numeric: set coefficient to 1.0 for this experience entry
        grad_experience.data[tuple(select_experience_indexes)] = 1.0

        # Symbolic channel: slice all relevant tensors
        grad_output_sliced = slice_view(grad_output, multi_output_points)
        input_sliced = slice_view(input, multi_output_points)
        output_sliced = slice_view(output, multi_output_points)
        experience_sliced_view = slice_view(experience, select_experience_indexes)
        grad_experience_sliced_view = slice_view(grad_experience, select_experience_indexes)
        # Assign TODO to the selected portion of grad_experience (unselected stays "")
        assign_tensor(grad_experience_sliced_view, todo_tensor_like(grad_experience_sliced_view))
        grad_experience_sliced_value = slice_tensor(grad_experience, select_experience_indexes)

        workspace_dir = tempfile.mkdtemp()
        grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_view_dir = os.path.join(workspace_dir, "const_output_view")
        experience_view_dir = os.path.join(workspace_dir, "const_experience_view")
        grad_experience_dir = os.path.join(workspace_dir, "mutable_grad_experience_dir")

        dump_view(grad_output_sliced, grad_output_view_dir, "txt")
        dump_view(input_sliced, input_view_dir, "txt")
        dump_view(output_sliced, output_view_dir, "txt")
        dump_view(experience_sliced_view, experience_view_dir, "txt")
        dump_view(grad_experience_sliced_value, grad_experience_dir, "txt")

        prompt = (
            "You are a symbolic gradient calculator for backward pass.\n\n"
            f"{forward_prompt}\n\n"
            "During forward pass, the input was translated to output using experience entries.\n"
            "Now given the output gradient (how output should change), compute gradients for\n"
            "experience.\n\n"
            "Context (read-only):\n"
            f"- Output gradient (text diff): \"{grad_output_view_dir}\"\n"
            f"- Original input: \"{input_view_dir}\"\n"
            f"- Original output: \"{output_view_dir}\"\n"
            f"- Experience entries used during forward: \"{experience_view_dir}\"\n"
            "  where .../0/data.xxx = query, .../1/data.xxx = key, .../2/data.xxx = value\n"
            "  Each line in query files only contains one summary key word for current experience,\n"
            "  which is used for calculating similarity between inputs and experience.\n"
            "  Key files contain source domain semantics.\n"
            "  Value files contain target domain semantics.\n\n"
            f"Compute and write:\n"
            f"1. Experience gradients in \"{grad_experience_dir}\":\n"
            "   How should each experience entry (query, key, value) change to improve the output?\n"
            "   Notice, there maybe be multiple grad_output. You should merge them into grad_experience.\n\n"
            "Replace all TODO with computed Gradient files.\n\n"
            "Gradient files format:\n"
            "1. Gradient files must be like output of `diff -u --label data --label data original.txt modified.txt`.\n"
            "2. Gradient files will be applied by cmd `patch -i backward.diff /forward/location/data`\n"
            f"3. Gradient files of \"{grad_experience_dir}/<xxx>/data\" must be able to be applied to \"{experience_view_dir}/<xxx>/data\"\n"
        )

        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_experience_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        task_contexts.append((workspace_dir, grad_experience_dir, grad_experience_sliced_view))

    # Batch LLM call
    if all_tasks:
        TaskHandler()(all_tasks, llm_method)

    # Copy results back and clean up
    for workspace_dir, grad_experience_dir, grad_experience_sliced_view in task_contexts:
        _copy_back_to_storage_view(grad_experience_dir, grad_experience_sliced_view)
        shutil.rmtree(workspace_dir)

    return grad_experience


def symbolic_transform_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    experience: torch.Tensor,
    selected_experience_qkv_indexes_list: Any,
    forward_prompt: str = "",
    topk: int = 16,
    llm_method: str = "raw_llm_api",
) -> Tuple[Union[torch.Tensor, None], torch.Tensor]:
    """Backward pass of the symbolic transform.

    Computes grad_input and grad_experience separately:
    - grad_input: iterates per input element, LLM computes input text diffs
    - grad_experience: iterates per experience entry via transposed sparse coordinates,
      LLM sees all relevant grad_outputs merged for each experience entry

    Args:
        grad_output: Gradient w.r.t. forward output (symbolic tensor with text diffs).
        input: Original input tensor (saved from forward ctx).
        output: Original output tensor (saved from forward ctx).
        experience: Experience tensor (saved from forward ctx, last dim=3: q/k/v).
        selected_experience_qkv_indexes_list: Nested list of index tensors from forward.
        forward_prompt: The prompt used during forward pass.
        topk: Number of top experience entries used per element.
        llm_method: LLM backend to use.

    Returns:
        (grad_input, grad_experience) — grad_input is None if input doesn't require grad.
    """
    grad_input = symbolic_transform_backward_grad_input(
        grad_output, input, output, experience,
        selected_experience_qkv_indexes_list,
        forward_prompt, topk, llm_method,
    )

    grad_experience = symbolic_transform_backward_grad_experience(
        grad_output, input, output, experience,
        selected_experience_qkv_indexes_list,
        forward_prompt, topk, llm_method,
    )

    return grad_input, grad_experience


if __name__ == "__main__":
    import subprocess
    from symbolic_tensor.tensor_util.make_tensor import make_tensor

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

    print("Running symbolic_transform_backward tests...\n")

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
            tensor.st_relative_to, tensor.st_tensor_uid,
            "storage", os.path.join(*digits), "data",
        )
        with open(path) as f:
            return f.read()

    # Test 1: Helper — _pad_indexes_to_topk_with_none_experience_indexes
    print("Test 1: Padding indexes to topk")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_data = [["q0", "k0", "v0"]]
        exp_tensor = make_tensor(exp_data, tmpdir)
        indexes = [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)]
        padded = _pad_indexes_to_topk_with_none_experience_indexes(indexes, 3, exp_tensor)
        run_test("Padded to length 3", len(padded[0]) == 3, 3, len(padded[0]))
        run_test("Original index preserved", padded[0][0].item() == 0)
        run_test("Pad values are 0", padded[0][1].item() == 0 and padded[0][2].item() == 0)

    # Test 2: Helper — no padding needed
    print("Test 2: No padding when already at topk")
    indexes = [torch.tensor([0, 1], dtype=torch.long)]
    padded = _pad_indexes_to_topk_with_none_experience_indexes(indexes, 2, exp_tensor)
    run_test("Length unchanged", len(padded[0]) == 2)

    # Test 3: Helper — _flatten_nested_indexes
    print("Test 3: Flatten nested indexes")
    nested = [[torch.tensor([0]), torch.tensor([1])], [torch.tensor([2]), torch.tensor([3])]]
    flat = _flatten_nested_indexes(nested, [2])
    run_test("Flat length is 2", len(flat) == 2)

    # Test 4: Numeric channel — grad_input pass-through, grad_experience zeros + set 1.0
    print("Test 4: Numeric channel")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["text_a", "text_b"]
        input_tensor = make_tensor(input_data, tmpdir)

        exp_data = [["q0", "k0", "v0"], ["q1", "k1", "v1"]]
        exp_tensor = make_tensor(exp_data, tmpdir)

        grad_out = todo_tensor_like(input_tensor)
        grad_out.data.fill_(2.0)

        # grad_input: pass-through
        grad_input = todo_tensor_like(input_tensor)
        grad_input.data.copy_(grad_out.data)
        run_test("grad_input coeff copied", torch.all(grad_input.data == 2.0).item())

        # grad_experience: zeros + set 1.0 for used entries
        sel_indexes = [
            [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long)],
        ]
        pairs = convert_nested_list_coordinates_to_pairs_coordinates(sel_indexes)
        transposed = transpose_pairs_coordinates(pairs)

        grad_experience = todo_tensor_like(exp_tensor)
        grad_experience.data.zero_()
        for sole_exp_point, multi_output_points in transposed:
            int_exp_point = [t.item() for t in sole_exp_point]
            select_exp_idx = int_exp_point[:-1] + [slice(None)]
            grad_experience.data[tuple(select_exp_idx)] = 1.0

        run_test("exp[0] coeff is 1.0", grad_experience.data[0, 0].item() == 1.0,
                 1.0, grad_experience.data[0, 0].item())
        run_test("exp[1] coeff is 1.0", grad_experience.data[1, 0].item() == 1.0,
                 1.0, grad_experience.data[1, 0].item())

    # Test 5: Full backward with LLM (raw_llm_api only, single batch)
    print("Test 5: Full backward with LLM (single input)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world in English"], tmpdir)
        input_tensor.requires_grad_(True)
        exp_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        exp_tensor = make_tensor(exp_data, tmpdir)
        output_tensor = make_tensor(["Bonjour le monde en francais"], tmpdir)
        grad_output_tensor = make_tensor(
            ["The translation should use formal French: 'Bonjour le monde en francais' -> 'Bonjour au monde en francais'"],
            tmpdir,
        )
        grad_output_tensor.data.fill_(1.0)

        sel_indexes = [[torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long)]]

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output_tensor, input_tensor, output_tensor, exp_tensor,
            selected_experience_qkv_indexes_list=sel_indexes,
            forward_prompt="Translate the English text to French.",
            topk=2,
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape matches input", list(grad_input.shape) == list(input_tensor.shape))
        run_test("grad_experience shape matches experience", list(grad_experience.shape) == list(exp_tensor.shape))

        # Check grad_input text diff
        gi_text = read_storage(grad_input, 0)
        run_test("grad_input text diff not TODO", "TODO" not in gi_text)
        print(f"  grad_input text: {repr(gi_text[:120])}")

        # Check grad_experience text diffs
        for i in range(grad_experience.numel()):
            ge_text = read_storage(grad_experience, i)
            print(f"  grad_experience[{i}]: TODO={'TODO' == ge_text.strip()} {repr(ge_text[:80])}")

        # Numeric checks
        run_test("grad_input coeff == 1.0", grad_input.data[0].item() == 1.0)
        run_test("grad_experience coeff for used entries", grad_experience.data[0, 0].item() == 1.0)

    # Test 6: Multi-batch backward (2 inputs, raw_llm_api)
    print("Test 6: Multi-batch backward (2 inputs)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world", "Goodbye world"], tmpdir)
        input_tensor.requires_grad_(True)
        exp_data = [
            ["greeting\nhello", "Hello world", "Bonjour le monde"],
            ["farewell\ngoodbye", "Goodbye world", "Au revoir le monde"],
        ]
        exp_tensor = make_tensor(exp_data, tmpdir)
        output_tensor = make_tensor(["Bonjour le monde", "Au revoir le monde"], tmpdir)
        grad_output_tensor = make_tensor(
            ["Fix: Bonjour le monde -> Bonjour au monde",
             "Fix: Au revoir le monde -> Au revoir au monde"],
            tmpdir,
        )
        grad_output_tensor.data.fill_(1.0)

        # Input 0 selected exp[0] and exp[1], Input 1 selected exp[1] and exp[0]
        sel_indexes = [
            [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long)],
            [torch.tensor([1, 0], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long)],
        ]

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output_tensor, input_tensor, output_tensor, exp_tensor,
            selected_experience_qkv_indexes_list=sel_indexes,
            forward_prompt="Translate English to French.",
            topk=2,
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape [2]", list(grad_input.shape) == [2])
        run_test("grad_experience shape [2, 3]", list(grad_experience.shape) == [2, 3])

        # Both experience entries should have coeff 1.0 (both used)
        run_test("exp[0] coeff 1.0", grad_experience.data[0, 0].item() == 1.0)
        run_test("exp[1] coeff 1.0", grad_experience.data[1, 0].item() == 1.0)

        # Both grad_inputs should have text diffs
        for i in range(2):
            gi_text = read_storage(grad_input, i)
            print(f"  grad_input[{i}]: TODO={'TODO' == gi_text.strip()} {repr(gi_text[:80])}")
        for i in range(grad_experience.numel()):
            ge_text = read_storage(grad_experience, i)
            print(f"  grad_experience[{i}]: TODO={'TODO' == ge_text.strip()} {repr(ge_text[:80])}")

    print("\nAll tests completed.")
