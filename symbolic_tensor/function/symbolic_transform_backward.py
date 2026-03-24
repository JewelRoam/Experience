import os
import tempfile
import itertools
import torch
from typing import Any, List, Tuple, Union

from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from symbolic_tensor.tensor_util.slice_view import slice_view
from symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from symbolic_tensor.tensor_util.dump_view import dump_view
from symbolic_tensor.llm_client.agent_task import AgentTask
from symbolic_tensor.llm_client.task_handler import TaskHandler


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


def _copy_back_to_storage_view(mutable_dir: str, view_tensor: torch.Tensor) -> None:
    """Copy LLM results from mutable workspace dir back through view tensor's symlinks.

    The mutable dir contains files written by the LLM (independent copies, no symlinks).
    The view_tensor was created by slice_view and has symlink storage pointing to the
    parent tensor. Writing to the view's storage files writes through to the parent.
    """
    coords_list = [list(coord) for coord in itertools.product(*[range(s) for s in view_tensor.size()])]
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, view_tensor.stride()))
        digits = list(str(flat_index))
        # View tensor's storage — symlinks to parent
        view_storage_path = os.path.join(
            view_tensor.st_relative_to,
            view_tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        # Resolve to real parent storage path
        real_storage_path = os.path.realpath(view_storage_path)
        # Mutable workspace file written by LLM
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
    """
    Backward pass of the symbolic transform.

    Two gradient channels:
      a) Numeric channel (coefficient): grad_input copies from grad_output;
         grad_experience accumulates grad_output coefficients per used experience entry.
      b) Symbolic channel (text diff): LLM infers how input and experience texts
         should change given the output gradient.

    Uses slice_view (symlinks) for copy-back targets and slice_tensor (copies) for
    LLM-writable mutable dirs. This avoids the symlink breakage issue.

    Processes each scalar element sequentially.

    Args:
        grad_output: Gradient w.r.t. forward output (symbolic tensor with text diffs).
        input: Original input tensor (saved from forward ctx).
        output: Original output tensor (saved from forward ctx).
        experience: Experience tensor (saved from forward ctx, last dim=3: q/k/v).
        selected_experience_qkv_indexes_list: Nested list of index tensors from forward.
        forward_prompt: The prompt used during forward pass.
        topk: Number of top experience entries used per element.

    Returns:
        A tuple of:
        - grad_input: Gradient w.r.t. input (same shape as input).
        - grad_experience: Gradient w.r.t. experience (same shape as experience).
    """
    # Initialize gradient tensors
    grad_input = todo_tensor_like(input) if input.requires_grad else None
    grad_experience = todo_tensor_like(experience)

    # ── Numeric channel (coefficient) ──
    # grad_input coefficient: pass-through copy from grad_output
    if input.requires_grad:
        grad_input.data.copy_(grad_output.data)
    # grad_experience coefficient: zero-init, then scatter-add
    grad_experience.data.zero_()

    # Flatten nested indexes to iterate per scalar element
    input_shape = list(input.size())
    coords_list = _scalar_slice_indices(input.size())
    flat_selected_indexes = _flatten_nested_indexes(
        selected_experience_qkv_indexes_list, input_shape
    )

    # Scatter-add grad_output coefficients to grad_experience for each element
    for coords, select_experience_query_indexes in zip(coords_list, flat_selected_indexes):
        scalar_grad_output_coeff = grad_output[tuple(coords)].item()

        padded_select_indexes = _pad_indexes_to_topk_with_none_experience_indexes(
            select_experience_query_indexes, topk, experience
        )
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            padded_select_indexes, experience.size()[-1]
        )
        # scatter-add: grad_experience[select_experience_indexes] += scalar_grad_output_coeff
        grad_experience.data[select_experience_indexes] += scalar_grad_output_coeff

    # ── Symbolic channel (text diff) ──
    # Process each scalar element sequentially
    for coords, select_experience_query_indexes in zip(coords_list, flat_selected_indexes):
        int_slices = [c for c in coords]

        # Slice const tensors to scalar views (symlinks, read-only)
        scalar_grad_output = slice_view(grad_output, int_slices)
        scalar_input = slice_view(input, int_slices)
        scalar_output = slice_view(output, int_slices)

        # Slice grad_input: view (symlink) for copy-back, value (copy) for LLM to write
        if input.requires_grad:
            scalar_grad_input_view = slice_view(grad_input, int_slices)
            scalar_grad_input_value = slice_tensor(grad_input, int_slices)

        # Pad indexes to topk
        padded_select_indexes = _pad_indexes_to_topk_with_none_experience_indexes(
            select_experience_query_indexes, topk, experience
        )
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            padded_select_indexes, experience.size()[-1]
        )

        # Slice experience (const view) and grad_experience (view + value)
        experience_sliced_view = slice_view(experience, select_experience_indexes)
        grad_experience_sliced_view = slice_view(grad_experience, select_experience_indexes)
        grad_experience_sliced_value = slice_tensor(grad_experience, select_experience_indexes)

        # Create workspace with dump views
        with tempfile.TemporaryDirectory() as workspace_dir:
            grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
            input_view_dir = os.path.join(workspace_dir, "const_input_view")
            output_view_dir = os.path.join(workspace_dir, "const_output_view")
            experience_view_dir = os.path.join(workspace_dir, "const_experience_view")
            grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")
            grad_experience_dir = os.path.join(workspace_dir, "mutable_grad_experience_dir")

            # Dump const views (read-only context for LLM)
            dump_view(scalar_grad_output, grad_output_view_dir, "txt")
            dump_view(scalar_input, input_view_dir, "txt")
            dump_view(scalar_output, output_view_dir, "txt")
            dump_view(experience_sliced_view, experience_view_dir, "txt")

            # Dump mutable copies (LLM writes gradients here — no symlinks to break)
            if input.requires_grad:
                dump_view(scalar_grad_input_value, grad_input_dir, "txt")
            dump_view(grad_experience_sliced_value, grad_experience_dir, "txt")

            compute_instructions = "Compute and write:\n"
            if input.requires_grad:
                compute_instructions += (
                    f"1. Input gradient in \"{grad_input_dir}\":\n"
                    "   How should the input text change to improve the output?\n"
                )
            compute_instructions += (
                f"{'2' if input.requires_grad else '1'}. Experience gradients in \"{grad_experience_dir}\":\n"
                "   How should each experience entry (query, key, value) change to improve the output?\n"
                "   Notice, it's possible that there are existed Experience gradients accumulated in\n"
                "   mutable_grad_experience_dir in previous iteration. You should merge them.\n\n"
                "Replace all TODO with computed Gradient files.\n\n"
                "Gradient files format:\n"
                "1. Gradient files must be like output of `diff -u --label data --label data original.txt modified.txt`.\n"
                "2. Gradient files will be applied by cmd `patch -i backward.diff /forward/location/data`\n"
                f"3. Gradient files of \"{grad_input_dir}/<xxx>/data\" must be able to be applied to \"{input_view_dir}/<xxx>/data\"\n"
                f"4. Gradient files of \"{grad_experience_dir}/<xxx>/data\" must be able to be applied to \"{experience_view_dir}/<xxx>/data\"\n"
            )

            prompt = (
                "You are a symbolic gradient calculator for backward pass.\n\n"
                f"{forward_prompt}\n\n"
                "During forward pass, the input was translated to output using experience entries.\n"
                "Now given the output gradient (how output should change), compute gradients for\n"
                "input and experience.\n\n"
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
                + compute_instructions
            )

            output_relative_dirs = ["mutable_grad_experience_dir"]
            if input.requires_grad:
                output_relative_dirs.insert(0, "mutable_grad_input_dir")

            agent_task = AgentTask(
                workspace_dir=workspace_dir,
                output_relative_dir=output_relative_dirs,
                prompt=prompt,
            )
            TaskHandler()([agent_task], llm_method)

            # Copy results back from mutable dir through view symlinks to parent storage
            if input.requires_grad:
                _copy_back_to_storage_view(grad_input_dir, scalar_grad_input_view)
            _copy_back_to_storage_view(grad_experience_dir, grad_experience_sliced_view)

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

    # Test 1: _pad_indexes_to_topk_with_none_experience_indexes
    print("Test 1: Padding indexes to topk")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_data = [["q0", "k0", "v0"]]
        exp_tensor = make_tensor(exp_data, tmpdir)
        indexes = [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)]
        padded = _pad_indexes_to_topk_with_none_experience_indexes(indexes, 3, exp_tensor)
        run_test("Padded to length 3", len(padded[0]) == 3, 3, len(padded[0]))
        run_test("Original index preserved", padded[0][0].item() == 0)
        run_test("Pad values are 0", padded[0][1].item() == 0 and padded[0][2].item() == 0)

    # Test 2: No padding needed
    print("Test 2: No padding when already at topk")
    indexes = [torch.tensor([0, 1], dtype=torch.long)]
    padded = _pad_indexes_to_topk_with_none_experience_indexes(indexes, 2, exp_tensor)
    run_test("Length unchanged", len(padded[0]) == 2)

    # Test 3: _flatten_nested_indexes
    print("Test 3: Flatten nested indexes")
    nested = [[torch.tensor([0]), torch.tensor([1])], [torch.tensor([2]), torch.tensor([3])]]
    flat = _flatten_nested_indexes(nested, [2])
    run_test("Flat length is 2", len(flat) == 2)

    # Test 4: Numeric channel — coefficient pass-through and scatter-add
    print("Test 4: Numeric channel")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["text_a", "text_b"]
        input_tensor = make_tensor(input_data, tmpdir)

        exp_data = [["q0", "k0", "v0"], ["q1", "k1", "v1"]]
        exp_tensor = make_tensor(exp_data, tmpdir)

        grad_out = todo_tensor_like(input_tensor)
        grad_out.data.fill_(2.0)

        sel_indexes = [
            [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long)],
        ]

        grad_input = todo_tensor_like(input_tensor)
        grad_experience = todo_tensor_like(exp_tensor)

        # Numeric channel: pass-through
        grad_input.data.copy_(grad_out.data)
        run_test("grad_input coeff copied", torch.all(grad_input.data == 2.0).item())

        # Numeric channel: scatter-add
        grad_experience.data.zero_()
        coords_list_t = _scalar_slice_indices(input_tensor.size())
        for coords, sel_idx in zip(coords_list_t, sel_indexes):
            coeff = grad_out[tuple(coords)].item()
            padded = _pad_indexes_to_topk_with_none_experience_indexes(sel_idx, 2, exp_tensor)
            sel_exp_idx = _replace_last_tensor_with_full_slice(padded, exp_tensor.size()[-1])
            grad_experience.data[sel_exp_idx] += coeff

        run_test("exp[0] accumulated", grad_experience.data[0, 0].item() == 4.0, 4.0, grad_experience.data[0, 0].item())
        run_test("exp[1] from element 1", grad_experience.data[1, 0].item() == 2.0, 2.0, grad_experience.data[1, 0].item())

    # Test 5: Full backward (symbolic channel, requires LLM)
    for method in ["coding_agent", "raw_llm_api"]:
        print(f"Test 5: Full backward with LLM (llm_method={method})")
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
                llm_method=method,
            )

            run_test("grad_input shape matches input", list(grad_input.shape) == list(input_tensor.shape))
            run_test("grad_experience shape matches experience", list(grad_experience.shape) == list(exp_tensor.shape))

            # Check grad_input text diff
            root = os.path.join(tmpdir, grad_input.st_tensor_uid, "storage")
            path = os.path.join(root, "0", "data")
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                run_test("grad_input text diff not TODO", "TODO" not in content)
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

            # Numeric channel checks
            run_test("grad_input coeff == 1.0", grad_input.data[0].item() == 1.0)
            run_test("grad_experience coeff accumulated", grad_experience.data[0, 0].item() == 1.0)

    print("\nAll tests completed.")
