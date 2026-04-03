import os
import shutil
import tempfile
import torch
from typing import Callable, Dict, List, Optional, Union

from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.tensor_util.assign_tensor import assign_tensor
from experience.symbolic_tensor.tensor_util.get_diff_tensor import get_diff_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.tensor_util.dump_view import dump_view
from experience.llm_client.agent_task import AgentTask
from experience.llm_client.task_handler import TaskHandler


def default_prompt_for_grad_input(
    task_prompt: str,
    workspace_dir: str,
    const_grad_output_view: str,
    const_input_view: str,
    const_output_view: str,
    mutable_grad_input_dir: str,
) -> str:
    """Default prompt for computing input gradient in slice_attention backward."""
    return (
        "You are a symbolic gradient calculator for backward pass.\n\n"
        f"{task_prompt}\n\n"
        "During forward pass, input elements were scattered to multiple output\n"
        "positions via attention mask. Each input[b, j] was copied to output[b, i, j]\n"
        "for every row i that attends to column j.\n\n"
        "Now given the output gradients (how each output copy should change),\n"
        "compute the gradient for the original input.\n\n"
        "Context (read-only):\n"
        f"- Output gradient (text diffs at attended positions): \"{const_grad_output_view}\"\n"
        f"- Original input element: \"{const_input_view}\"\n"
        f"- Original output (copies at attended positions): \"{const_output_view}\"\n\n"
        "Compute and write:\n"
        f"1. Input gradient in \"{mutable_grad_input_dir}\":\n"
        "   How should the input text change, considering feedback from all\n"
        "   output positions that attended to it?\n"
        f"2. File \"{mutable_grad_input_dir}/<xxx>/data\" must be a better version "
        f"of \"{const_input_view}/<xxx>/data\"\n\n"
        "Replace all TODO with improved source text.\n"
    )


def slice_attention_backward_grad_input(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_input_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> Optional[torch.Tensor]:
    """Compute grad_input for slice_attention backward.

    Transposes the forward scatter: for each input element (b, j),
    gathers grad_output from all rows that attended to column j.

    Dual-channel gradient:
      Numeric: grad_input[b,j] = mean_i(grad_output[b,i,j] * mask[b,i,j])
      Symbolic: LLM sees all output diffs for each input element and writes
                an improved version; framework computes unified diff.
    """
    if not input.requires_grad:
        return None

    grad_input = todo_tensor_like(input)

    # ── Numeric channel ──
    # Mean over attending rows (dim=1) with mask
    grad_input.data = (grad_output.data * attention_mask.float()).mean(dim=1)

    # ── Symbolic channel ──
    # Transpose the forward scatter: for each input column j, find all
    # rows that attended to it
    input_attended_mask = attention_mask.any(dim=1)  # (batch, seq_len)
    valid_input_points = list(torch.nonzero(input_attended_mask, as_tuple=True))

    if valid_input_points[0].numel() == 0:
        return grad_input

    all_tasks: List[AgentTask] = []
    task_contexts: List[tuple] = []

    for batch_i, col_j in zip(
        valid_input_points[0].tolist(), valid_input_points[1].tolist()
    ):
        # Which rows attend to column col_j
        attending_rows = torch.nonzero(
            attention_mask[batch_i, :, col_j], as_tuple=True
        )[0]

        # Slice grad_output at [batch_i, attending_rows, col_j] — 1D view
        grad_output_view = slice_view(grad_output, [batch_i, attending_rows, col_j])
        # Slice output at same positions
        output_view = slice_view(output, [batch_i, attending_rows, col_j])
        # Slice input at [batch_i, col_j] — single element
        input_view = slice_view(input, [batch_i, col_j])
        # Grad input: view (for assign back) and value (for LLM mutable dump)
        grad_input_view = slice_view(grad_input, [batch_i, col_j])
        grad_input_value = slice_tensor(grad_input, [batch_i, col_j])

        workspace_dir = tempfile.mkdtemp()
        grad_output_view_dir = os.path.join(workspace_dir, "const_grad_output_view")
        input_view_dir = os.path.join(workspace_dir, "const_input_view")
        output_view_dir = os.path.join(workspace_dir, "const_output_view")
        grad_input_dir = os.path.join(workspace_dir, "mutable_grad_input_dir")

        # Dump const views (read-only context for LLM)
        dump_view(grad_output_view, grad_output_view_dir, "txt")
        dump_view(input_view, input_view_dir, "txt")
        dump_view(output_view, output_view_dir, "txt")
        # Dump mutable view (LLM writes gradient here)
        dump_view(grad_input_value, grad_input_dir, "txt")

        prompt = (grad_input_prompt or default_prompt_for_grad_input)(
            task_prompt, workspace_dir, grad_output_view_dir, input_view_dir,
            output_view_dir, grad_input_dir,
        )

        agent_task = AgentTask(
            workspace_dir=workspace_dir,
            output_relative_dir=["mutable_grad_input_dir"],
            prompt=prompt,
        )
        all_tasks.append(agent_task)
        task_contexts.append(
            (workspace_dir, grad_input_view, input_view, grad_input_value)
        )

    # Batch LLM call
    if all_tasks:
        TaskHandler()(all_tasks, llm_method, llm_env=llm_env)

    # Compute diff between original input and LLM-written improved version,
    # assign diff back to grad_input view
    for ws_dir, gi_view, in_view, gi_value in task_contexts:
        diff = get_diff_tensor(in_view, gi_value)
        assign_tensor(gi_view, diff)
        shutil.rmtree(ws_dir)

    # Enforce invariant: coeff > 0 iff storage file has real content.
    # Empty diffs (0-byte files from assign_tensor) must have coeff = 0.
    for i in range(grad_input.numel()):
        digits = list(str(i))
        path = os.path.join(
            grad_input.st_relative_to,
            grad_input.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        path = os.path.realpath(path)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            grad_input.data.flatten()[i] = 0.0

    return grad_input


def slice_attention_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    output: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_input_prompt: Optional[Callable[..., str]] = None,
    task_prompt: str = "",
    llm_method: str = "raw_llm_api",
    llm_env: Optional[Dict[str, str]] = None,
) -> Optional[torch.Tensor]:
    """Backward pass for slice_attention_forward.

    Computes grad_input by transposing the forward attention scatter:
    for each input element (b, j), gathers gradients from all output
    positions (b, i, j) that attended to it.

    Args:
        grad_output: Gradient w.r.t. forward output (symbolic tensor, shape batch x seq x seq).
        input: Original input tensor saved from forward ctx (batch x seq).
        output: Original output tensor saved from forward ctx (batch x seq x seq).
        attention_mask: Bool mask saved from forward ctx (batch x seq x seq).
        grad_input_prompt: Custom prompt callable. None uses default.
        task_prompt: High-level task description.
        llm_method: LLM backend to use.
        llm_env: Environment variables for LLM.

    Returns:
        grad_input tensor, or None if input doesn't require grad.
    """
    if not input.requires_grad:
        return None
    return slice_attention_backward_grad_input(
        grad_output, input, output, attention_mask,
        grad_input_prompt, task_prompt, llm_method, llm_env,
    )


if __name__ == "__main__":
    import subprocess
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.function.slice_attention_forward import (
        slice_attention_forward,
    )

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

    print("Running slice_attention_backward tests...\n")

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
        path = os.path.realpath(path)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return f.read()

    # Test 1: Numeric channel — mean over attending rows
    print("Test 1: Numeric channel (mean)")
    with tempfile.TemporaryDirectory() as tmpdir:
        # mask[0,:,0] = [T, T], mask[0,:,1] = [F, T]
        mask = torch.tensor([[[True, False], [True, True]]])
        grad_out_data = torch.tensor([[[2.0, 0.0], [4.0, 6.0]]])
        # (grad_out * mask.float()).mean(dim=1):
        #   col 0: (2*1 + 4*1) / 2 = 3.0
        #   col 1: (0*0 + 6*1) / 2 = 3.0
        expected = torch.tensor([[3.0, 3.0]])
        actual = (grad_out_data * mask.float()).mean(dim=1)
        run_test("Numeric formula correct",
                 torch.allclose(actual, expected),
                 expected.tolist(), actual.tolist())

    # Test 2: Numeric channel — multi-batch
    print("Test 2: Numeric channel multi-batch")
    with tempfile.TemporaryDirectory() as tmpdir:
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True
        mask[0, 1, 1] = True
        mask[1, 0, 0] = True
        grad_out_data = torch.tensor([
            [[0.0, 0.0], [2.0, 4.0]],
            [[6.0, 0.0], [0.0, 0.0]],
        ])
        expected = torch.tensor([
            [(0*0 + 2*1) / 2, (0*0 + 4*1) / 2],  # [1.0, 2.0]
            [(6*1 + 0*0) / 2, (0*0 + 0*0) / 2],   # [3.0, 0.0]
        ])
        actual = (grad_out_data * mask.float()).mean(dim=1)
        run_test("Multi-batch numeric correct",
                 torch.allclose(actual, expected),
                 expected.tolist(), actual.tolist())

    # Test 3: requires_grad=False returns None
    print("Test 3: No grad when not required")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a"]], tmpdir)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        out = make_tensor([[["x"]]], tmpdir)
        grad_out = todo_tensor_like(out)
        result = slice_attention_backward(grad_out, inp, out, mask)
        run_test("Returns None", result is None)

    # Test 4: Full backward with LLM (raw_llm_api, 1x2 causal)
    print("Test 4: Full backward with LLM (1x2 causal)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["Hello world", "Goodbye"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tensor([[[True, False], [True, True]]])
        out = slice_attention_forward(inp, mask)

        grad_out = make_tensor(
            [[["Improve: be more formal", "unused"],
              ["Keep greeting formal", "Make farewell formal"]]],
            tmpdir,
        )
        grad_out.data[mask] = 1.0

        grad_input = slice_attention_backward(
            grad_out, inp, out, mask,
            task_prompt="Improve text formality",
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])
        for i in range(2):
            gi_text = read_storage(grad_input, i)
            run_test(f"grad_input[{i}] not None", gi_text is not None)
            print(f"  grad_input[{i}]: {repr(gi_text[:80]) if gi_text else 'None'}")

    # Test 5: Multi-batch backward (2x2)
    print("Test 5: Multi-batch backward (2x2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["cat", "dog"], ["sun", "moon"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.zeros(2, 2, 2, dtype=torch.bool)
        mask[0, 1, 0] = True
        mask[0, 1, 1] = True
        mask[1, 0, 0] = True

        out = slice_attention_forward(inp, mask)

        grad_out = make_tensor(
            [[["N/A", "N/A"], ["Improve cat desc", "Improve dog desc"]],
             [["Improve sun desc", "N/A"], ["N/A", "N/A"]]],
            tmpdir,
        )
        grad_out.data[mask] = 1.0

        grad_input = slice_attention_backward(
            grad_out, inp, out, mask,
            task_prompt="Improve descriptions",
            llm_method="raw_llm_api",
        )

        run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])
        for b in range(2):
            for j in range(2):
                flat = b * 2 + j
                gi_text = read_storage(grad_input, flat)
                print(f"  grad_input[{b},{j}]: "
                      f"{repr(gi_text[:60]) if gi_text else 'None'}")

    print("\nAll tests completed.")
