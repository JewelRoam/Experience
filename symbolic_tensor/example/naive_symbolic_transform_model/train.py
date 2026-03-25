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

from symbolic_tensor.tensor_util.make_tensor import make_tensor
from symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD
from symbolic_tensor.example.naive_symbolic_transform_model.model import NaiveModel
from symbolic_tensor.function import symbolic_grad_registry

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


def main():
    print("=" * 60)
    print("NaiveModel Training: Translate Python To Viba")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Load dataset ──
        # Input: Python files, Expected: Viba files
        pairs = [
            ("seq.py", "seq.viba"),
            ("branch.py", "branch.viba"),
            ("loop.py", "loop.viba"),
        ]

        py_paths = []
        viba_contents = []
        for py_name, viba_name in pairs:
            py_paths.append(Path(os.path.join(DATASET_DIR, py_name)))
            with open(os.path.join(DATASET_DIR, viba_name)) as f:
                viba_contents.append(f.read())

        # Create input tensor (symlinks to .py files)
        input_tensor = make_tensor(
            [Path(p) for p in py_paths],
            tmpdir,
            symlink=True,
        )

        # Create expected output tensor (.viba content)
        expected_tensor = make_tensor(viba_contents, tmpdir)

        print(f"\nDataset: {len(pairs)} pairs")
        for i, (py_name, viba_name) in enumerate(pairs):
            print(f"  [{i}] {py_name} -> {viba_name}")

        # ── Build experience ──
        # Experience: [query_keywords, key_python, value_viba] per entry
        experience_entries = []
        for i, (py_name, viba_name) in enumerate(pairs):
            experience_entries.append([""] * 3)

        experience_tensor = make_tensor(experience_entries, tmpdir)

        # ── Create model and optimizer ──
        model = NaiveModel(forward_prompt=FORWARD_PROMPT, topk=1)
        model.load_experience(experience_tensor)

        optimizer = SymbolicSGD(
            model.parameters(),
            lr=1.0,
        )

        print(f"\nExperience shape: {list(experience_tensor.shape)}")
        print(f"Input shape: {list(input_tensor.shape)}")
        print(f"Expected shape: {list(expected_tensor.shape)}")

        # ── Training loop ──
        losses = []
        patch_stats = []

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration}/{NUM_ITERATIONS}")
            print(f"{'─' * 60}")

            # Zero gradients
            optimizer.zero_grad()

            # Forward through model (autograd-tracked)
            print("\n  [Forward]")
            output, selected_indexes = model(input_tensor)

            for i in range(output.numel()):
                out_text = read_storage(output, i)
                print(f"    output[{i}] (first 80): {repr(out_text[:80])}")

            # Loss: edit distance ratio via autograd Function
            print("\n  [Loss]")
            loss = get_edit_distance_ratio(output, expected_tensor)
            mean_loss = loss.mean().item()
            losses.append(mean_loss)
            print(f"    Per-sample losses: {[f'{l:.4f}' for l in loss.tolist()]}")
            print(f"    Mean loss: {mean_loss:.4f}")

            # Backward through full autograd graph:
            #   MeanBackward -> GetEditDistanceRatioBackward (produces diff text)
            #   -> SymbolicTransformBackward (LLM computes grad for input & experience)
            print("\n  [Backward]")
            loss.mean().backward()

            # Show gradient info
            grad_experience = model.transform.experience.grad
            # Restore symbolic attributes stripped by autograd (peek by uid, don't consume)
            symbolic_grad = symbolic_grad_registry.peek(model.transform.experience.st_tensor_uid)
            if symbolic_grad is not None:
                grad_experience = symbolic_grad
            if grad_experience is not None and hasattr(grad_experience, "st_tensor_uid"):
                for i in range(min(grad_experience.numel(), 9)):
                    gt = read_storage(grad_experience, i)
                    print(f"    grad_exp[{i}]: {repr(gt[:60])}")
            else:
                print("    (no symbolic gradient on experience)")

            # Optimizer step
            print("\n  [Optimizer Step]")
            exp_before = read_storage(model.transform.experience, 2)

            optimizer.step()

            exp_after = read_storage(model.transform.experience, 2)
            changed = exp_before != exp_after
            print(f"    Experience[0].value changed: {changed}")

            # Patch application stats
            stats = optimizer.get_last_step_stats()
            print(f"    Patch stats: applied={stats['applied']} rejected={stats['rejected']} "
                  f"fuzzed={stats['fuzzed']} skipped={stats['skipped']} rej_files={stats['rej_files']}")

            # Show experience state after step
            print("    Experience state after step:")
            for i in range(experience_tensor.numel()):
                et = read_storage(experience_tensor, i)
                print(f"      exp[{i}]: {repr(et[:80])}")

            print(f"\n  Iteration {iteration} loss: {mean_loss:.4f}")
            patch_stats.append(stats)

        # ── Save losses ──
        print(f"\n{'=' * 60}")
        print("Training Complete")
        print(f"{'=' * 60}")
        print(f"\nLoss trajectory: {[f'{l:.4f}' for l in losses]}")

        # ── Patch rejection analysis ──
        print(f"\n{'─' * 60}")
        print("Patch Application Analysis")
        print(f"{'─' * 60}")
        total_applied = sum(s["applied"] for s in patch_stats)
        total_rejected = sum(s["rejected"] for s in patch_stats)
        total_fuzzed = sum(s["fuzzed"] for s in patch_stats)
        total_skipped = sum(s["skipped"] for s in patch_stats)
        total_rej = sum(s["rej_files"] for s in patch_stats)
        total_attempts = total_applied + total_rejected
        print(f"  Total patches attempted: {total_attempts}")
        print(f"  Applied cleanly: {total_applied - total_fuzzed}")
        print(f"  Applied with fuzz: {total_fuzzed}")
        print(f"  Rejected: {total_rejected}")
        print(f"  Skipped (TODO/empty): {total_skipped}")
        print(f"  .rej files created: {total_rej}")
        if total_attempts > 0:
            print(f"  Success rate: {total_applied / total_attempts * 100:.1f}%")
            print(f"  Rejection rate: {total_rejected / total_attempts * 100:.1f}%")
        print(f"\nPer-iteration breakdown:")
        for i, s in enumerate(patch_stats, 1):
            print(f"  Iter {i}: applied={s['applied']} rejected={s['rejected']} "
                  f"fuzzed={s['fuzzed']} skipped={s['skipped']} rej={s['rej_files']}")

        with open(LOSS_LOG, "w") as f:
            for i, loss_val in enumerate(losses, 1):
                print(f"iteration {i}: {loss_val:.6f}\n")
                f.write(f"iteration {i}: {loss_val:.6f}\n")
            f.write(f"\nConverged: {losses[-1] < losses[0] if len(losses) > 1 else 'N/A'}\n")

        print(f"Losses saved to {LOSS_LOG}")

        # Validate convergence
        if len(losses) > 1 and losses[-1] < losses[0]:
            print("Loss CONVERGED (final < initial)")
        else:
            print("Loss did NOT converge (final >= initial)")


if __name__ == "__main__":
    main()
