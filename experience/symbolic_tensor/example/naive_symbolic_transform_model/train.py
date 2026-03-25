"""
Training script for NaiveModel: Translate Python to Viba.

5 iterations, validate loss convergence.
Save loss of each iteration to /tmp/loss.log.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from experience.symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD
from experience.symbolic_tensor.example.naive_symbolic_transform_model.model import NaiveModel
from experience.symbolic_tensor.function import symbolic_grad_registry

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


EXAMPLE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(EXAMPLE_DIR, "dataset")
LOSS_LOG = "/tmp/loss.log"
NUM_ITERATIONS = 5
FORWARD_PROMPT = "Translate Python To Viba"

DATASET_PAIRS = [
    "seq", "branch", "loop",
    "recursion", "higher_order", "data_struct",
    "default_arg", "list_comp", "format_str",
    "guard", "accumulator", "closure",
]


def read_storage(tensor, flat_index):
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


def load_dataset(tmpdir):
    """Load .py/.viba pairs into input and expected tensors."""
    py_paths = [Path(DATASET_DIR) / f"{name}.py" for name in DATASET_PAIRS]
    viba_contents = [(Path(DATASET_DIR) / f"{name}.viba").read_text() for name in DATASET_PAIRS]

    input_tensor = make_tensor(
        [Path(p) for p in py_paths], tmpdir, symlink=True,
    )
    expected_tensor = make_tensor(viba_contents, tmpdir)
    return input_tensor, expected_tensor


def print_header(title, char="="):
    print(f"\n{char * 60}")
    print(title)
    print(f"{char * 60}")


def print_patch_summary(patch_stats):
    """Print aggregate patch application stats."""
    total = {k: sum(s[k] for s in patch_stats) for k in patch_stats[0]}
    attempts = total["applied"] + total["rejected"]

    print_header("Patch Application Analysis", "─")
    print(f"  Total attempted: {attempts}")
    print(f"  Applied cleanly: {total['applied'] - total['fuzzed']}")
    print(f"  Applied with fuzz: {total['fuzzed']}")
    print(f"  Rejected: {total['rejected']}")
    print(f"  Skipped (TODO/empty): {total['skipped']}")
    print(f"  .rej files: {total['rej_files']}")
    if attempts > 0:
        print(f"  Success rate: {total['applied'] / attempts * 100:.1f}%")
    print()
    for i, s in enumerate(patch_stats, 1):
        print(f"  Iter {i}: applied={s['applied']} rejected={s['rejected']} "
              f"fuzzed={s['fuzzed']} skipped={s['skipped']} rej={s['rej_files']}")


def main():
    print_header("NaiveModel Training: Translate Python To Viba")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Dataset ──
        input_tensor, expected_tensor = load_dataset(tmpdir)
        n = len(DATASET_PAIRS)

        print(f"\nDataset: {n} pairs")
        for i, name in enumerate(DATASET_PAIRS):
            print(f"  [{i}] {name}.py -> {name}.viba")

        # ── Experience (empty, learned during training) ──
        experience_tensor = make_tensor([[""] * 3 for _ in range(n)], tmpdir)

        # ── Model & optimizer ──
        model = NaiveModel(forward_prompt=FORWARD_PROMPT, topk=1)
        model.load_experience(experience_tensor)
        optimizer = SymbolicSGD(model.parameters(), lr=1.0)

        print(f"\nExperience: {list(experience_tensor.shape)}")
        print(f"Input:      {list(input_tensor.shape)}")
        print(f"Expected:   {list(expected_tensor.shape)}")

        # ── Training loop ──
        losses = []
        patch_stats = []

        for iteration in range(1, NUM_ITERATIONS + 1):
            print_header(f"Iteration {iteration}/{NUM_ITERATIONS}", "─")
            optimizer.zero_grad()

            # Forward
            print("\n  [Forward]")
            output, selected_indexes = model(input_tensor)
            for i in range(output.numel()):
                print(f"    output[{i}]: {repr(read_storage(output, i)[:80])}")

            # Loss
            print("\n  [Loss]")
            loss = get_edit_distance_ratio(output, expected_tensor)
            mean_loss = loss.mean().item()
            losses.append(mean_loss)
            print(f"    Per-sample: {[f'{v:.4f}' for v in loss.tolist()]}")
            print(f"    Mean: {mean_loss:.4f}")

            # Backward
            print("\n  [Backward]")
            loss.mean().backward()

            grad_exp = model.transform.experience.grad
            symbolic_grad = symbolic_grad_registry.peek(
                model.transform.experience.st_tensor_uid
            )
            if symbolic_grad is not None:
                grad_exp = symbolic_grad
            if grad_exp is not None and hasattr(grad_exp, "st_tensor_uid"):
                for i in range(min(grad_exp.numel(), 9)):
                    print(f"    grad_exp[{i}]: {repr(read_storage(grad_exp, i)[:60])}")
            else:
                print("    (no symbolic gradient)")

            # Optimizer step
            print("\n  [Step]")
            optimizer.step()

            stats = optimizer.get_last_step_stats()
            patch_stats.append(stats)
            print(f"    Patches: applied={stats['applied']} rejected={stats['rejected']} "
                  f"fuzzed={stats['fuzzed']} skipped={stats['skipped']}")

            # Experience snapshot
            print("    Experience after step:")
            for i in range(experience_tensor.numel()):
                print(f"      [{i}]: {repr(read_storage(experience_tensor, i)[:80])}")

            print(f"\n  Iteration {iteration} loss: {mean_loss:.4f}")

        # ── Summary ──
        print_header("Training Complete")
        print(f"\nLoss trajectory: {[f'{v:.4f}' for v in losses]}")

        print_patch_summary(patch_stats)

        # Save losses
        with open(LOSS_LOG, "w") as f:
            for i, v in enumerate(losses, 1):
                f.write(f"iteration {i}: {v:.6f}\n")
            converged = losses[-1] < losses[0] if len(losses) > 1 else "N/A"
            f.write(f"\nConverged: {converged}\n")
        print(f"\nLosses saved to {LOSS_LOG}")

        if len(losses) > 1 and losses[-1] < losses[0]:
            print("Loss CONVERGED (final < initial)")
        else:
            print("Loss did NOT converge (final >= initial)")


if __name__ == "__main__":
    main()
