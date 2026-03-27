import os
import subprocess
import tempfile
import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.symbolic_tensor.function.st_matmul_forward import st_matmul_forward
from experience.symbolic_tensor.function.st_matmul_backward import st_matmul_backward


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


if __name__ == "__main__":
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

    print("Running test_gain_st_sgd...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # ================================================================
    # Test 1: forward -> backward -> step -> forward (single input)
    # ================================================================
    # Scenario: Experience has a deliberately wrong French translation.
    # The gradient corrects it. After optimizer.step(), the second forward
    # should produce the corrected translation.
    print("Test 1: forward -> backward -> step -> forward (single input, improved output)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world in English"], tmpdir)

        # Experience with a deliberately wrong French translation
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjor la mond"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        optimizer = StSGD([experience_tensor], lr=1.0)

        # ── First forward ──
        print("\n  === Iteration 1: Forward ===")
        output1, selected_indexes1 = st_matmul_forward(
            input_tensor, experience_tensor,
            topk=2,
        )
        output1_text = read_storage(output1, 0)
        print(f"  Output: {repr(output1_text[:120])}")

        # ── Backward with correction signal ──
        print("\n  === Iteration 1: Backward ===")
        grad_output = make_tensor(
            ["The French translation has spelling errors. "
             "Correct: 'Bonjour le monde en francais'. "
             "Fix 'Bonjor' -> 'Bonjour', 'la mond' -> 'le monde en francais'."],
            tmpdir,
        )
        grad_output.data.fill_(1.0)

        grad_input, grad_experience = st_matmul_backward(
            grad_output, input_tensor, output1, experience_tensor,
            selected_experience_qkv_indexes_list=selected_indexes1,
            topk=2,
        )
        experience_tensor.grad = grad_experience

        # Print grad diffs
        for i in range(grad_experience.numel()):
            gt = read_storage(grad_experience, i)
            print(f"  grad_experience[{i}]: {repr(gt[:80])}")

        # ── Optimizer step ──
        print("\n  === Optimizer Step ===")
        exp_val_before = read_storage(experience_tensor, 2)  # value of first entry
        print(f"  experience[0].value before: {repr(exp_val_before)}")
        optimizer.step()
        exp_val_after = read_storage(experience_tensor, 2)
        print(f"  experience[0].value after:  {repr(exp_val_after)}")
        run_test("experience value updated by optimizer", exp_val_after != exp_val_before)

        # ── Second forward with updated experience ──
        print("\n  === Iteration 2: Forward (after optimizer step) ===")
        # Reset coefficient to 1.0 so tensor is usable
        experience_tensor.data.fill_(1.0)
        output2, selected_indexes2 = st_matmul_forward(
            input_tensor, experience_tensor,
            topk=2,
        )
        output2_text = read_storage(output2, 0)
        print(f"  Output: {repr(output2_text[:120])}")

        # ── Verify improvement ──
        print("\n  === Results ===")
        print(f"  Forward 1: {repr(output1_text)}")
        print(f"  Forward 2: {repr(output2_text)}")
        has_bonjor = "Bonjor" in output2_text or "bonjor" in output2_text
        run_test("Second output no longer has 'Bonjor' typo", not has_bonjor)
        run_test("Second output differs from first", output2_text != output1_text)

    # ================================================================
    # Test 2: forward -> backward -> step -> forward (multi-batch: 2 inputs)
    # ================================================================
    print("\n\nTest 2: forward -> backward -> step -> forward (multi-batch: 2 inputs)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(
            ["Hello world in English", "Goodbye my friend"],
            tmpdir,
        )

        # Experience with deliberately wrong translations
        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjor la mond"],
            ["farewell\ngoodbye\nfriend", "Goodbye my friend", "Odieu mon ami"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        optimizer = StSGD([experience_tensor], lr=1.0)

        # ── First forward ──
        print("\n  === Iteration 1: Forward ===")
        output1, selected_indexes1 = st_matmul_forward(
            input_tensor, experience_tensor,
            topk=2,
        )
        output1_texts = [read_storage(output1, i) for i in range(output1.numel())]
        for i, t in enumerate(output1_texts):
            print(f"  Output[{i}]: {repr(t[:120])}")

        # ── Backward with correction signal ──
        print("\n  === Iteration 1: Backward ===")
        grad_output = make_tensor(
            ["Fix spelling: 'Bonjor la mond' -> 'Bonjour le monde en francais'",
             "Fix spelling: 'Odieu mon ami' -> 'Adieu mon ami'"],
            tmpdir,
        )
        grad_output.data.fill_(1.0)

        grad_input, grad_experience = st_matmul_backward(
            grad_output, input_tensor, output1, experience_tensor,
            selected_experience_qkv_indexes_list=selected_indexes1,
            topk=2,
        )
        experience_tensor.grad = grad_experience

        for i in range(grad_experience.numel()):
            gt = read_storage(grad_experience, i)
            print(f"  grad_experience[{i}]: {repr(gt[:80])}")

        # ── Optimizer step ──
        print("\n  === Optimizer Step ===")
        exp_vals_before = [read_storage(experience_tensor, i) for i in range(experience_tensor.numel())]
        optimizer.step()
        exp_vals_after = [read_storage(experience_tensor, i) for i in range(experience_tensor.numel())]
        for i in range(experience_tensor.numel()):
            print(f"  experience[{i}] before: {repr(exp_vals_before[i][:60])}")
            print(f"  experience[{i}] after:  {repr(exp_vals_after[i][:60])}")
        # At least the value entries (index 2 and 5) should have changed
        run_test("at least one experience entry updated",
                 any(a != b for a, b in zip(exp_vals_before, exp_vals_after)))

        # ── Second forward with updated experience ──
        print("\n  === Iteration 2: Forward (after optimizer step) ===")
        experience_tensor.data.fill_(1.0)
        output2, selected_indexes2 = st_matmul_forward(
            input_tensor, experience_tensor,
            topk=2,
        )
        output2_texts = [read_storage(output2, i) for i in range(output2.numel())]
        for i, t in enumerate(output2_texts):
            print(f"  Output[{i}]: {repr(t[:120])}")

        # ── Verify improvement ──
        print("\n  === Results ===")
        for i in range(len(output1_texts)):
            print(f"  Forward 1[{i}]: {repr(output1_texts[i])}")
            print(f"  Forward 2[{i}]: {repr(output2_texts[i])}")
        has_bonjor = any("Bonjor" in t or "bonjor" in t for t in output2_texts)
        run_test("Second output no longer has 'Bonjor' typo", not has_bonjor)
        has_odieu = any("Odieu" in t or "odieu" in t for t in output2_texts)
        run_test("Second output no longer has 'Odieu' typo", not has_odieu)
        run_test("At least one output changed",
                 any(t1 != t2 for t1, t2 in zip(output1_texts, output2_texts)))

    print("\nAll tests completed.")
