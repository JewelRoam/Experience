"""
Training script for NaiveModel: Translate Python to Viba.

auto_train: configurable training function for AutoML search.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_view import slice_view
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.example.naive_symbolic_transform_model.model import NaiveModel
from experience.symbolic_tensor.function import symbolic_grad_registry


EXAMPLE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(EXAMPLE_DIR, "dataset")
TASK_PROMPT = "Translate Python To Viba"

DATASET_PAIRS = [
    "seq", "branch", "loop",
    "recursion", "higher_order", "data_struct",
    "default_arg", "list_comp", "format_str",
    "guard", "accumulator", "closure",
]


def _read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def _load_dataset(tmpdir):
    """Load .py/.viba pairs into input and expected tensors."""
    py_paths = [Path(DATASET_DIR) / f"{name}.py" for name in DATASET_PAIRS]
    viba_contents = [(Path(DATASET_DIR) / f"{name}.viba").read_text() for name in DATASET_PAIRS]

    input_tensor = make_tensor(
        [Path(p) for p in py_paths], tmpdir, symlink=True,
    )
    expected_tensor = make_tensor(viba_contents, tmpdir)
    return input_tensor, expected_tensor


def _print_header(title, char="="):
    print(f"\n{char * 60}")
    print(title)
    print(f"{char * 60}")


def _format_patch_summary(patch_stats):
    """Format aggregate patch application stats as a string."""
    lines = []
    total = {k: sum(s[k] for s in patch_stats) for k in patch_stats[0]}
    attempts = total["applied"] + total["rejected"]

    lines.append("Patch Application Analysis")
    lines.append(f"  Total attempted: {attempts}")
    lines.append(f"  Applied cleanly: {total['applied'] - total['fuzzed']}")
    lines.append(f"  Applied with fuzz: {total['fuzzed']}")
    lines.append(f"  Rejected: {total['rejected']}")
    lines.append(f"  Skipped (TODO/empty): {total['skipped']}")
    lines.append(f"  .rej files: {total['rej_files']}")
    if attempts > 0:
        lines.append(f"  Success rate: {total['applied'] / attempts * 100:.1f}%")
    lines.append("")
    for i, s in enumerate(patch_stats, 1):
        lines.append(f"  Iter {i}: applied={s['applied']} rejected={s['rejected']} "
                     f"fuzzed={s['fuzzed']} skipped={s['skipped']} rej={s['rej_files']}")
    return "\n".join(lines)


def auto_train(
    num_experience: int = 24,
    topk: int = 1,
    forward_prompt: Optional[Callable[..., str]] = None,
    lr: float = 1.0,
    llm_model: Optional[str] = None,
    retrieval_method: Optional[Callable[[str, str], float]] = None,
    num_iterations: int = 5,
    batch_size: int = 12,
) -> Tuple[List[float], Dict[str, str]]:
    """
    Configurable training function for AutoML search.

    Args:
        num_experience: Number of experience rows (each row is [query, key, value]).
        topk: Number of top experience entries to select per input.
        forward_prompt: Callable that builds the forward prompt. None uses default.
        lr: Learning rate for StSGD.
        llm_model: LLM model name override. None uses os.environ default.
        retrieval_method: Callable(query_file_content, key_file_content) -> float.
            None uses default Jaccard similarity.
        num_iterations: Number of training iterations.
        batch_size: Number of dataset samples per training batch.

    Returns:
        A tuple of:
        - losses: List of mean loss values, one per iteration.
        - logs: Dict mapping log type to log content string.
            Keys: "training", "patch_summary", "loss_trajectory", "convergence".
    """
    llm_env = None
    if llm_model is not None:
        llm_env = {
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", ""),
            "LLM_BASE_URL": os.environ.get("LLM_BASE_URL", ""),
            "LLM_MODEL": llm_model,
        }

    training_lines = []

    def log(msg=""):
        training_lines.append(msg)
        print(msg)

    log(f"auto_train(num_experience={num_experience}, topk={topk}, lr={lr}, "
        f"llm_model={llm_model!r}, num_iterations={num_iterations}, batch_size={batch_size})")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Dataset ──
        input_tensor, expected_tensor = _load_dataset(tmpdir)
        n = len(DATASET_PAIRS)

        log(f"\nDataset: {n} pairs")
        for i, name in enumerate(DATASET_PAIRS):
            log(f"  [{i}] {name}.py -> {name}.viba")

        # ── Experience (empty, learned during training) ──
        experience_tensor = make_tensor([[""] * 3 for _ in range(num_experience)], tmpdir)

        # ── Model & optimizer ──
        model = NaiveModel(
            output_prompt=forward_prompt,
            task_prompt=TASK_PROMPT,
            topk=topk,
            retrieval_method=retrieval_method,
            llm_env=llm_env,
        )
        model.load_experience(experience_tensor)
        optimizer = StSGD(model.parameters(), lr=lr)

        log(f"\nExperience: {list(experience_tensor.shape)}")
        log(f"Input:      {list(input_tensor.shape)}")
        log(f"Expected:   {list(expected_tensor.shape)}")

        # ── Training loop ──
        losses = []
        patch_stats = []

        for iteration in range(1, num_iterations + 1):
            log(f"\n{'─' * 60}")
            log(f"Iteration {iteration}/{num_iterations}")
            log(f"{'─' * 60}")
            optimizer.zero_grad()

            # Batch slicing
            start = ((iteration - 1) * batch_size) % n
            end = min(start + batch_size, n)
            if start == 0 and end == n:
                batch_input = input_tensor
                batch_expected = expected_tensor
            else:
                batch_input = slice_view(input_tensor, [slice(start, end)])
                batch_expected = slice_view(expected_tensor, [slice(start, end)])

            # Forward
            log("\n  [Forward]")
            output, selected_indexes = model(batch_input)
            for i in range(output.numel()):
                log(f"    output[{i}]: {repr(_read_storage(output, i)[:80])}")

            # Loss
            log("\n  [Loss]")
            loss = get_edit_distance_ratio(output, batch_expected)
            mean_loss = loss.mean().item()
            losses.append(mean_loss)
            log(f"    Per-sample: {[f'{v:.4f}' for v in loss.tolist()]}")
            log(f"    Mean: {mean_loss:.4f}")

            # Backward
            log("\n  [Backward]")
            loss.mean().backward()

            grad_exp = model.transform.experience.grad
            symbolic_grad = symbolic_grad_registry.peek(
                model.transform.experience.st_tensor_uid
            )
            if symbolic_grad is not None:
                grad_exp = symbolic_grad
            if grad_exp is not None and hasattr(grad_exp, "st_tensor_uid"):
                for i in range(min(grad_exp.numel(), 9)):
                    log(f"    grad_exp[{i}]: {repr(_read_storage(grad_exp, i)[:60])}")
            else:
                log("    (no symbolic gradient)")

            # Append-only: zero out grad for non-empty experience rows
            # so the optimizer skips them (it only patches nonzero-grad rows)
            exp = model.transform.experience
            for row_idx in range(exp.shape[0]):
                key_text = _read_storage(exp, row_idx * 3 + 1)  # dim 1 = key
                if key_text.strip():
                    symbolic_grad = symbolic_grad_registry.peek(exp.st_tensor_uid)
                    if symbolic_grad is not None:
                        symbolic_grad.data[row_idx] = 0.0

            # Optimizer step
            log("\n  [Step]")
            optimizer.step()

            stats = optimizer.get_last_step_stats()
            patch_stats.append(stats)
            log(f"    Patches: applied={stats['applied']} rejected={stats['rejected']} "
                f"fuzzed={stats['fuzzed']} skipped={stats['skipped']}")

            # Experience snapshot
            log("    Experience after step:")
            for i in range(experience_tensor.numel()):
                log(f"      [{i}]: {repr(_read_storage(experience_tensor, i)[:80])}")

            log(f"\n  Iteration {iteration} loss: {mean_loss:.4f}")

        # ── Build logs ──
        loss_trajectory = f"Loss trajectory: {[f'{v:.4f}' for v in losses]}"
        log(f"\n{'=' * 60}")
        log("Training Complete")
        log(f"{'=' * 60}")
        log(f"\n{loss_trajectory}")

        patch_summary = _format_patch_summary(patch_stats) if patch_stats else ""
        log(f"\n{patch_summary}")

        if len(losses) > 1 and losses[-1] < losses[0]:
            convergence = "Loss CONVERGED (final < initial)"
        else:
            convergence = "Loss did NOT converge (final >= initial)"
        log(convergence)

        logs = {
            "training": "\n".join(training_lines),
            "patch_summary": patch_summary,
            "loss_trajectory": loss_trajectory,
            "convergence": convergence,
        }

    return losses, logs


if __name__ == "__main__":
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

    _print_header("NaiveModel Training: Translate Python To Viba")

    losses, logs = auto_train(
        num_experience=2 * len(DATASET_PAIRS),
        topk=1,
        forward_prompt=None,
        lr=1.0,
        llm_model=None,
        retrieval_method=None,
        num_iterations=5,
        batch_size=len(DATASET_PAIRS),
    )

    # Save losses
    loss_log = "/tmp/loss.log"
    with open(loss_log, "w") as f:
        for i, v in enumerate(losses, 1):
            f.write(f"iteration {i}: {v:.6f}\n")
        converged = losses[-1] < losses[0] if len(losses) > 1 else "N/A"
        f.write(f"\nConverged: {converged}\n")
    print(f"\nLosses saved to {loss_log}")
