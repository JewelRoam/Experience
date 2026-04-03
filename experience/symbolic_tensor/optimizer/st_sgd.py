import os
import pathlib
import subprocess
import tempfile
import itertools
import torch
from typing import Callable, Dict, List, Optional, Union

from experience.symbolic_tensor.function import symbolic_grad_registry
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.tensor_util.patch_tensor import patch_tensor
from experience.fs_util.get_nested_list_file_pathes import get_nested_list_file_pathes


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _reset_grad_text_to_todo(param: torch.Tensor) -> None:
    """Reset all text storage files of param.grad to 'TODO'."""
    grad = param.grad
    if grad is None or not hasattr(grad, "st_tensor_uid"):
        return
    root = os.path.join(grad.st_relative_to, grad.st_tensor_uid)
    coords_list = _scalar_slice_indices(grad.size())
    for coords in coords_list:
        flat_index = sum(c * s for c, s in zip(coords, grad.stride()))
        digits = list(str(flat_index))
        storage_path = os.path.join(root, "storage", os.path.join(*digits), "data")
        real_path = os.path.realpath(storage_path)
        if os.path.isfile(real_path):
            with open(real_path, "w", encoding="utf-8") as f:
                f.write("TODO")


def _replace_last_tensor_with_slice(
    index_tensors: List[torch.Tensor],
    last_dim_slice: slice,
) -> List[Union[torch.Tensor, slice]]:
    """Replace the last index tensor with a specific slice."""
    result: List[Union[torch.Tensor, slice]] = list(index_tensors[:-1])
    result.append(last_dim_slice)
    return result


def _flatten_nested_paths(nested, result=None):
    """Flatten a nested list of pathlib.Path into a flat list."""
    if result is None:
        result = []
    if isinstance(nested, pathlib.Path):
        result.append(nested)
    elif isinstance(nested, list):
        for item in nested:
            _flatten_nested_paths(item, result)
    return result


def _get_nonzero_points(grad: torch.Tensor) -> List[torch.Tensor]:
    """Get the nonzero coordinate tensors from grad.data."""
    nz = torch.nonzero(grad.data, as_tuple=True)
    return list(nz)


class StSGD(torch.optim.Optimizer):
    """
    Symbolic SGD optimizer. Two-channel update:
      a) Numeric (coefficient): param.data = (1 - lr) * param.data + lr * grad.data
      b) Symbolic (text): apply unified diff patches from grad storage to param storage
         Only patches elements where grad.data != 0 (key+value dims).
      c) Query auto-update: after patching key+value, derive query content from
         updated key+value file text (sort unique lines, join with newline).

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.01).
        llm_env: Environment variable dict for LLM client. None uses os.environ defaults.
    """

    def __init__(self, params, lr: float = 0.01, llm_env: Optional[Dict[str, str]] = None):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.llm_env = llm_env

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        self._last_step_stats = {"applied": 0, "rejected": 0, "fuzzed": 0, "skipped": 0, "rej_files": 0}

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Restore symbolic attributes stripped by autograd
                if hasattr(param, "st_tensor_uid"):
                    symbolic_grad = symbolic_grad_registry.pop(param.st_tensor_uid)
                    if symbolic_grad is not None:
                        grad = symbolic_grad
                        param.grad = grad

                # ── Numeric channel ──
                param.data.mul_(1.0 - lr).add_(grad.data, alpha=lr)

                # ── Symbolic channel ──
                if not hasattr(grad, "st_tensor_uid"):
                    continue

                # Find non-zero gradient points and get unique row indexes
                raw_selected_param_points = _get_nonzero_points(grad)
                if not raw_selected_param_points or len(raw_selected_param_points[0]) == 0:
                    continue

                # Unique rows: drop last dim (q/k/v), get unique combinations
                row_points = raw_selected_param_points[:-1]  # all dims except last
                if row_points:
                    # Stack row coords, find unique rows
                    stacked = torch.stack(row_points, dim=1)
                    unique_rows = torch.unique(stacked, dim=0)
                    unique_row_points = [unique_rows[:, d] for d in range(unique_rows.shape[1])]
                else:
                    unique_row_points = []

                # kv_points: unique rows × key+value slice (1:3)
                kv_points = list(unique_row_points) + [slice(1, 3, None)]

                # Get storage paths for kv_points elements
                kv_param = slice_view(param, kv_points)
                kv_grad = slice_view(grad, kv_points)

                # Apply unified diff patches from grad onto param via st_patch
                stats = patch_tensor(kv_param, kv_grad)
                self._last_step_stats["applied"] += stats["applied"]
                self._last_step_stats["rejected"] += stats["rejected"]
                self._last_step_stats["fuzzed"] += stats["fuzzed"]
                self._last_step_stats["skipped"] += stats["skipped"]
                self._last_step_stats["rej_files"] += stats["rej_files"]

                # ── Update queries from key+value content ──
                self._update_queries(param, unique_row_points)

        return loss

    def _update_queries(self, param: torch.Tensor, unique_row_points: List[torch.Tensor]) -> None:
        """After patching key+value, auto-derive query content from updated key+value text.

        Flow per viba:
        1. Run get_query_tensor on kv_param → LLM generates keywords per element
        2. Read keyword files for each (key, value) pair
        3. Merge, sort, unique, join with newline
        4. Write to query file
        """
        if not hasattr(param, "st_tensor_uid"):
            return

        from experience.symbolic_tensor.function.get_query_tensor import get_query_tensor

        # kv_points: key(1) + value(2) dims
        kv_points = list(unique_row_points) + [slice(1, 3, None)]
        kv_param = slice_view(param, kv_points)

        # Step 1: Run get_query_tensor on kv_param to generate LLM keywords
        kv_queries = get_query_tensor(kv_param, llm_method="raw_llm_api", llm_env=self.llm_env)

        # Step 2: Get file paths from kv_queries (LLM-generated keywords)
        kv_query_file_paths = get_nested_list_file_pathes(kv_queries)
        flat_kv_query_paths = _flatten_nested_paths(kv_query_file_paths)

        # query_points: query(0) dim
        query_points = list(unique_row_points) + [slice(0, 1, None)]
        query_view = slice_view(param, query_points)

        query_file_paths = get_nested_list_file_pathes(query_view)
        flat_query_paths = _flatten_nested_paths(query_file_paths)

        # Step 3-4: For each experience entry, read keyword files for key+value,
        # merge, sort, unique, write to query file
        n_entries = len(flat_query_paths)
        for i in range(n_entries):
            # Each entry has 2 kv keyword files: from_key at 2*i, from_value at 2*i+1
            key_query_path = flat_kv_query_paths[2 * i] if 2 * i < len(flat_kv_query_paths) else None
            value_query_path = flat_kv_query_paths[2 * i + 1] if 2 * i + 1 < len(flat_kv_query_paths) else None
            query_path = flat_query_paths[i]

            # Read LLM-generated keywords from both key and value query files
            all_lines = []
            for p in [key_query_path, value_query_path]:
                if p is not None and p.exists():
                    text = p.read_text(encoding="utf-8").strip()
                    if text:
                        all_lines.extend(text.splitlines())

            # Sort, unique, join with newline
            unique_lines = sorted(set(all_lines))
            query_content = "\n".join(unique_lines)
            if query_content:
                query_content += "\n"

            # Write to query file
            real_query_path = os.path.realpath(str(query_path))
            with open(real_query_path, "w", encoding="utf-8") as f:
                f.write(query_content)

    def get_last_step_stats(self) -> dict:
        """Return patch application stats from the last optimizer step."""
        return dict(self._last_step_stats)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients. If set_to_none=False, also resets grad text storage to 'TODO'."""
        if not set_to_none:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        _reset_grad_text_to_todo(param)
        super().zero_grad(set_to_none=set_to_none)


if __name__ == "__main__":
    import subprocess
    import tempfile
    from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
    from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like

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

    print("Running StSGD tests...\n")

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

    # Test 1: Constructor
    print("Test 1: Constructor")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        opt = StSGD([exp], lr=0.1)
        run_test("param_groups has 1 group", len(opt.param_groups) == 1)
        run_test("lr is 0.1", opt.param_groups[0]["lr"] == 0.1)

    # Test 2: Numeric channel (coefficient update)
    print("Test 2: Numeric channel")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        opt = StSGD([exp], lr=0.5)

        # Manually set a plain gradient (no st_ attrs)
        exp.grad = torch.ones_like(exp) * 2.0
        orig_data = exp.data.clone()
        opt.step()
        # param.data = (1 - 0.5) * orig + 0.5 * 2.0 = 0.5 * orig + 1.0
        expected = 0.5 * orig_data + 1.0
        run_test("Coefficient updated", torch.allclose(exp.data, expected),
                 expected.tolist(), exp.data.tolist())

    # Test 3: zero_grad with set_to_none=True
    print("Test 3: zero_grad(set_to_none=True)")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        exp.grad = torch.ones_like(exp)
        opt = StSGD([exp], lr=0.1)
        opt.zero_grad(set_to_none=True)
        run_test("grad is None", exp.grad is None)

    # Test 4: zero_grad with set_to_none=False resets text
    print("Test 4: zero_grad(set_to_none=False) resets text")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["q", "k", "v"]], tmpdir)
        exp.requires_grad_(True)
        grad = make_tensor([["grad_q", "grad_k", "grad_v"]], tmpdir)
        grad.data.fill_(1.0)
        exp.grad = grad
        run_test("grad text before reset", read_storage(exp.grad, 0) == "grad_q")
        opt = StSGD([exp], lr=0.1)
        opt.zero_grad(set_to_none=False)
        run_test("grad coeff zeroed", exp.grad.data[0, 0].item() == 0.0)
        run_test("grad text is TODO", read_storage(exp.grad, 0) == "TODO")

    # Test 5: Symbolic channel patches only kv elements (key+value)
    print("Test 5: Patch kv elements only")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Experience: [query, key, value]
        exp = make_tensor([["query_word", "hello world", "bonjour monde"]], tmpdir)
        exp.requires_grad_(True)

        # Grad: only key(1) and value(2) have nonzero coefficients
        diff_key = (
            "--- data\n"
            "+++ data\n"
            "@@ -1 +1 @@\n"
            "-hello world\n"
            "+goodbye world\n"
        )
        diff_value = (
            "--- data\n"
            "+++ data\n"
            "@@ -1 +1 @@\n"
            "-bonjour monde\n"
            "+au revoir monde\n"
        )
        grad = make_tensor([["", diff_key, diff_value]], tmpdir)
        # Only key and value get nonzero grad coefficients
        grad.data.fill_(0.0)
        grad.data[0, 1] = 1.0
        grad.data[0, 2] = 1.0
        exp.grad = grad

        opt = StSGD([exp], lr=1.0)
        opt.step()

        stats = opt.get_last_step_stats()
        run_test("2 patches applied", stats["applied"] == 2, 2, stats["applied"])
        run_test("key patched", read_storage(exp, 1).strip() == "goodbye world",
                 "goodbye world", repr(read_storage(exp, 1)))
        run_test("value patched", read_storage(exp, 2).strip() == "au revoir monde",
                 "au revoir monde", repr(read_storage(exp, 2)))
        # Query should be auto-updated from key+value content
        query_after = read_storage(exp, 0)
        run_test("query auto-updated (not original)", query_after.strip() != "query_word",
                 "not query_word", repr(query_after))
        # Query should contain sorted unique lines from key+value
        query_lines = query_after.strip().splitlines()
        run_test("query has sorted unique lines", query_lines == sorted(set(query_lines)))

    # Test 6: Skip TODO and empty grads
    print("Test 6: Skip TODO grads")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([["query", "original key", "original value"]], tmpdir)
        exp.requires_grad_(True)

        grad = make_tensor([["", "TODO", "TODO"]], tmpdir)
        grad.data.fill_(0.0)
        grad.data[0, 1] = 1.0
        grad.data[0, 2] = 1.0
        exp.grad = grad

        opt = StSGD([exp], lr=0.5)
        opt.step()

        stats = opt.get_last_step_stats()
        run_test("2 patches skipped", stats["skipped"] == 2, 2, stats["skipped"])
        run_test("key unchanged", read_storage(exp, 1) == "original key")
        run_test("value unchanged", read_storage(exp, 2) == "original value")

    # Test 7: Multi-entry experience query auto-update
    print("Test 7: Multi-entry query auto-update")
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = make_tensor([
            ["old_query_0", "key content zero", "value content zero"],
            ["old_query_1", "key content one", "value content one"],
        ], tmpdir)
        exp.requires_grad_(True)

        diff_key_0 = (
            "--- data\n+++ data\n@@ -1 +1 @@\n"
            "-key content zero\n+key updated zero\n"
        )
        diff_value_0 = (
            "--- data\n+++ data\n@@ -1 +1 @@\n"
            "-value content zero\n+value updated zero\n"
        )
        diff_key_1 = (
            "--- data\n+++ data\n@@ -1 +1 @@\n"
            "-key content one\n+key updated one\n"
        )
        diff_value_1 = (
            "--- data\n+++ data\n@@ -1 +1 @@\n"
            "-value content one\n+value updated one\n"
        )
        grad = make_tensor([
            ["", diff_key_0, diff_value_0],
            ["", diff_key_1, diff_value_1],
        ], tmpdir)
        grad.data.fill_(0.0)
        grad.data[0, 1] = 1.0
        grad.data[0, 2] = 1.0
        grad.data[1, 1] = 1.0
        grad.data[1, 2] = 1.0
        exp.grad = grad

        opt = StSGD([exp], lr=1.0)
        opt.step()

        stats = opt.get_last_step_stats()
        run_test("4 patches applied", stats["applied"] == 4, 4, stats["applied"])

        # Check keys and values updated
        run_test("key[0] updated", read_storage(exp, 1).strip() == "key updated zero")
        run_test("value[0] updated", read_storage(exp, 2).strip() == "value updated zero")
        run_test("key[1] updated", read_storage(exp, 4).strip() == "key updated one")
        run_test("value[1] updated", read_storage(exp, 5).strip() == "value updated one")

        # Check queries auto-derived
        q0 = read_storage(exp, 0).strip()
        q1 = read_storage(exp, 3).strip()
        run_test("query[0] auto-derived from key+value", q0 != "old_query_0")
        run_test("query[1] auto-derived from key+value", q1 != "old_query_1")
        # Query should have sorted unique lines from key+value text
        q0_lines = q0.splitlines()
        run_test("query[0] lines are sorted unique", q0_lines == sorted(set(q0_lines)))
        print(f"  query[0] = {repr(q0)}")
        print(f"  query[1] = {repr(q1)}")

    print("\nAll tests completed.")
