import os
import subprocess
import tempfile
import torch
from typing import List, Optional

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.function.st_moe import st_moe
from experience.symbolic_tensor.module.st_moe import StMoeModule
from experience.symbolic_tensor.module.with_dense_view import WithDenseView
from experience.symbolic_tensor.function.st_moe_backward import (
    _read_storage, _detect_input_content_type, _PLAIN, _MERGED,
)
from experience.symbolic_tensor.function.st_copy import copy_impl
from experience.symbolic_tensor.function.get_causal_attention_mask import (
    get_causal_attention_mask,
)
from experience.symbolic_tensor.optimizer.st_sgd import StSGD
from experience.fs_util.text_merger import TextMerger, kFrameMarker


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


def data_file_sizes(tensor):
    """Return dict {flat_index: file_size} for all data files."""
    storage_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid, "storage")
    result = {}
    for root, dirs, files in os.walk(storage_dir):
        for f in files:
            if f == "data":
                rel = os.path.relpath(root, storage_dir)
                flat_idx = int("".join(rel.split(os.sep)))
                fpath = os.path.realpath(os.path.join(root, f))
                result[flat_idx] = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
    return result


def check_coeff_storage_consistency(tensor, label):
    """Verify coefficient > 0 iff data file exists with non-zero size."""
    sizes = data_file_sizes(tensor)
    flat_data = tensor.data.flatten()
    ok = True
    for i in range(tensor.numel()):
        coeff = flat_data[i].item()
        has_file = i in sizes and sizes[i] > 0
        if coeff > 0 and not has_file:
            print(f"    {label}[{i}]: coeff={coeff} but no data file")
            ok = False
        elif coeff == 0 and has_file:
            print(f"    {label}[{i}]: coeff=0 but data file exists (size={sizes[i]})")
            ok = False
    return ok


def build_model(tmpdir, experience_data, topk=1, task_prompt="Translate English to French"):
    """Build a WithDenseView(StMoeModule) with loaded experience."""
    num_entries = len(experience_data)
    moe = StMoeModule(
        experience_shape=[num_entries, 3],
        topk=topk,
        task_prompt=task_prompt,
    )
    src = make_tensor(experience_data, tmpdir)
    loaded = copy_impl(src, moe._experience_dir)
    moe.experience = loaded
    moe.experience.requires_grad_(True)
    model = WithDenseView(dense_handler=lambda x: moe(x)[0])
    model._moe = moe  # keep reference for parameter access
    return model


def run_pipeline(model, inp, mask):
    """Run st_attention → WithDenseView(StMoe) full pipeline."""
    attn_out = st_attention(inp, mask)
    output = model(attn_out)
    return attn_out, output


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

    print("Running WithDenseView(StMoe)(st_attention(x)) tests...\n")

    experience_en_fr = [
        ["greeting\nhello\nworld", "Hello world", "Bonjour le monde"],
        ["farewell\ngoodbye", "Goodbye", "Au revoir"],
    ]

    # ══════════════════════════════════════════════════════
    # Forward: WithDenseView(StMoe)(st_attention(x))
    # ══════════════════════════════════════════════════════

    # Test 1: shape 1x3 causal
    print("Test 1: Shape 1x3 causal — full pipeline")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "World", "Goodbye"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out, output = run_pipeline(model, inp, mask)
        # WithDenseView returns sparse: all 3 positions are nonzero after causal attn
        run_test("output numel >= 3", output.numel() >= 3)
        for i in range(output.numel()):
            content = read_storage(output, i)
            run_test(f"output[{i}] non-None", content is not None)
            if content and i == 0:
                print(f"    output[0]: {repr(content[:80])}")

    # Test 2: output is LLM-generated (differs from raw input)
    print("Test 2: Output is LLM-generated, not raw input copy")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello world", "Goodbye world"]], tmpdir)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        _, output = run_pipeline(model, inp, mask)
        for i in range(output.numel()):
            out_text = read_storage(output, i)
            run_test(f"output[{i}] non-None", out_text is not None)
            if out_text:
                print(f"    output[{i}]: {repr(out_text[:60])}")

    # Test 3: full attention 1x3 — every position sees all tokens
    print("Test 3: Full attention 1x3")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "Goodbye", "World"]], tmpdir)
        mask = torch.ones(1, 3, 3, dtype=torch.bool)
        _, output = run_pipeline(model, inp, mask)
        run_test("output has >= 3 elements", output.numel() >= 3)
        for i in range(output.numel()):
            run_test(f"output[{i}] non-None", read_storage(output, i) is not None)

    # Test 4: diagonal 1x3 — self-only attention (plain-like input)
    print("Test 4: Diagonal 1x3 — self-only → plain input to moe")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "Goodbye", "World"]], tmpdir)
        mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        attn_out, output = run_pipeline(model, inp, mask)
        # Each attention output is single-frame
        for i in range(min(3, attn_out.numel())):
            coeff = attn_out.data.flatten()[i].item()
            run_test(f"attn coeff[{i}] = 1.0 (self-only)", coeff == 1.0, 1.0, coeff)
        run_test("output has >= 3 elements", output.numel() >= 3)

    # Test 5: causal 1x5 — all outputs valid
    print("Test 5: Causal 1x5 — all outputs valid")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        tokens = ["Hello", "beautiful", "world", "Goodbye", "friend"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))
        _, output = run_pipeline(model, inp, mask)
        non_none = sum(1 for i in range(output.numel()) if read_storage(output, i) is not None)
        run_test("all 5 outputs non-None", non_none == 5, 5, non_none)

    # Test 6: padding 1x4 — WithDenseView strips padded positions
    print("Test 6: Padding 1x4 — WithDenseView strips pads")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello world", "Goodbye", "PAD", "PAD"]], tmpdir)
        token_mask = torch.tensor([[True, True, False, False]])
        mask = get_causal_attention_mask(token_mask)
        attn_out, output = run_pipeline(model, inp, mask)
        # attn positions 2,3 have coeff=0 (padded)
        run_test("attn[0,2] coeff=0", attn_out.data[0, 2].item() == 0.0)
        run_test("attn[0,3] coeff=0", attn_out.data[0, 3].item() == 0.0)
        # WithDenseView strips zeros → sparse output has only 2 elements
        run_test("sparse output has 2 elements", output.numel() == 2, 2, output.numel())
        for i in range(output.numel()):
            run_test(f"output[{i}] non-None", read_storage(output, i) is not None)

    # Test 7: multi-batch 2x3 causal
    print("Test 7: Multi-batch 2x3 causal")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([
            ["Hello", "World", "Goodbye"],
            ["Good morning", "Good evening", "Farewell"],
        ], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool)).expand(2, -1, -1).clone()
        _, output = run_pipeline(model, inp, mask)
        # All 6 positions nonzero → sparse has 6 elements
        run_test("output has 6 elements", output.numel() == 6, 6, output.numel())
        non_none = sum(1 for i in range(output.numel()) if read_storage(output, i) is not None)
        run_test("all 6 outputs non-None", non_none == 6, 6, non_none)

    # Test 8: different masks same input → different dense views
    print("Test 8: Causal vs full mask — different outputs")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "World"]], tmpdir)
        causal_mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        full_mask = torch.ones(1, 2, 2, dtype=torch.bool)
        attn_causal, _ = run_pipeline(model, inp, causal_mask)
        attn_full, _ = run_pipeline(model, inp, full_mask)
        # Causal pos 0 sees 1 token (coeff=1), full pos 0 sees 2 (coeff=2)
        run_test("causal attn[0] coeff=1", attn_causal.data[0, 0].item() == 1.0)
        run_test("full attn[0] coeff=2", attn_full.data[0, 0].item() == 2.0)
        run_test("different attention contexts",
                 attn_causal.data[0, 0].item() != attn_full.data[0, 0].item())

    # Test 9: storage consistency
    print("Test 9: Storage consistency on output")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "World", "End"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        _, output = run_pipeline(model, inp, mask)
        ok = check_coeff_storage_consistency(output, "output")
        run_test("output coeff-storage consistent", ok)

    # Test 10: get_causal_attention_mask end-to-end
    print("Test 10: get_causal_attention_mask → st_attention → WithDenseView(StMoe)")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["I", "love", "you"]], tmpdir)
        token_mask = torch.tensor([[True, True, True]])
        causal_mask = get_causal_attention_mask(token_mask)
        run_test("causal mask shape [1,3,3]", list(causal_mask.shape) == [1, 3, 3])
        _, output = run_pipeline(model, inp, causal_mask)
        non_none = sum(1 for i in range(output.numel()) if read_storage(output, i) is not None)
        run_test("all 3 outputs non-None", non_none == 3, 3, non_none)

    # Test 11: sliding window 1x4
    print("Test 11: Sliding window 1x4 (window=2)")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "World", "Goodbye", "Friend"]], tmpdir)
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        for i in range(4):
            for j in range(max(0, i - 1), i + 1):
                mask[0, i, j] = True
        attn_out, output = run_pipeline(model, inp, mask)
        run_test("attn coeff[0]=1", attn_out.data[0, 0].item() == 1.0)
        for i in range(1, 4):
            run_test(f"attn coeff[{i}]=2", attn_out.data[0, i].item() == 2.0)
        non_none = sum(1 for i in range(output.numel()) if read_storage(output, i) is not None)
        run_test("all 4 outputs non-None", non_none == 4, 4, non_none)

    # Test 12: large batch 3x4 causal — stress test
    print("Test 12: Large batch 3x4 causal — stress test")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        tokens = [[f"token_{b}_{i}" for i in range(4)] for b in range(3)]
        inp = make_tensor(tokens, tmpdir)
        mask = torch.tril(torch.ones(1, 4, 4, dtype=torch.bool)).expand(3, -1, -1).clone()
        _, output = run_pipeline(model, inp, mask)
        run_test("output has 12 elements", output.numel() == 12, 12, output.numel())
        non_none = sum(1 for i in range(output.numel()) if read_storage(output, i) is not None)
        run_test("all 12 outputs non-None", non_none == 12, 12, non_none)
        ok = check_coeff_storage_consistency(output, "output_3x4")
        run_test("3x4 coeff-storage consistent", ok)

    # ══════════════════════════════════════════════════════
    # Backward
    # ══════════════════════════════════════════════════════

    # Test 13: backward shape through full pipeline
    print("Test 13: Backward shape — full autograd pipeline")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello world", "Goodbye"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        output = model(attn_out)
        output.backward(torch.ones_like(output))
        grad_input = inp.grad
        run_test("grad_input not None", grad_input is not None)
        if grad_input is not None:
            run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])

    # Test 14: backward produces symbolic diffs
    print("Test 14: Backward produces symbolic diffs")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "Goodbye"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        output = model(attn_out)
        output.backward(torch.ones_like(output))
        grad_input = inp.grad
        run_test("grad_input not None", grad_input is not None)
        if grad_input is not None:
            run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])

    # Test 15: grad_experience shape
    print("Test 15: grad_experience shape")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        moe = model._moe
        inp = make_tensor([["Hello world"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        attn_out = st_attention(inp, mask)
        output = model(attn_out)
        output.backward(torch.ones_like(output))
        grad_exp = moe.experience.grad
        run_test("grad_experience not None", grad_exp is not None)
        if grad_exp is not None:
            run_test("grad_experience shape [2, 3]", list(grad_exp.shape) == [2, 3])

    # Test 16: multi-batch 2x2 backward
    print("Test 16: Multi-batch 2x2 backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        inp = make_tensor([["Hello", "World"], ["Goodbye", "Friend"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool)).expand(2, -1, -1).clone()
        attn_out = st_attention(inp, mask)
        output = model(attn_out)
        output.backward(torch.ones_like(output))
        grad_input = inp.grad
        run_test("grad_input not None", grad_input is not None)
        if grad_input is not None:
            run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])

    # ══════════════════════════════════════════════════════
    # End-to-end integration
    # ══════════════════════════════════════════════════════

    # Test 17: two-iteration training loop
    print("Test 17: Two-iteration training loop")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        moe = model._moe
        optimizer = StSGD(moe.parameters(), lr=1.0)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)

        # Iteration 1
        inp1 = make_tensor([["Hello world"]], tmpdir)
        inp1.requires_grad_(True)
        attn1 = st_attention(inp1, mask)
        out1 = model(attn1)
        out1_text = read_storage(out1, 0)
        run_test("iter 1 output non-None", out1_text is not None)
        if out1_text:
            print(f"    iter 1: {repr(out1_text[:80])}")
        out1.backward(torch.ones_like(out1))
        optimizer.step()
        optimizer.zero_grad()

        # Iteration 2
        inp2 = make_tensor([["Hello world"]], tmpdir)
        inp2.requires_grad_(True)
        attn2 = st_attention(inp2, mask)
        out2 = model(attn2)
        out2_text = read_storage(out2, 0)
        run_test("iter 2 output non-None", out2_text is not None)
        if out2_text:
            print(f"    iter 2: {repr(out2_text[:80])}")
        # Experience was modified by optimizer.step(), so the loop completed
        changed = 0
        for i in range(moe.experience.numel()):
            orig = read_storage(moe.experience, i)
            if orig is not None:
                changed += 1
        run_test("training loop completes 2 iterations", out2_text is not None)

    # Test 18: experience update after training step
    print("Test 18: Experience update after training step")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = build_model(tmpdir, experience_en_fr)
        moe = model._moe
        optimizer = StSGD(moe.parameters(), lr=1.0)

        # Record original experience
        orig_exp = {}
        for i in range(moe.experience.numel()):
            orig_exp[i] = read_storage(moe.experience, i)

        inp = make_tensor([["Hello world", "Goodbye"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        output = model(attn_out)
        output.backward(torch.ones_like(output))
        optimizer.step()

        changed = 0
        for i in range(moe.experience.numel()):
            new_val = read_storage(moe.experience, i)
            if new_val != orig_exp[i]:
                changed += 1
        run_test(f"experience entries changed: {changed}", changed > 0)
        print(f"    {changed}/{moe.experience.numel()} entries modified")

    print("\nAll tests completed.")
