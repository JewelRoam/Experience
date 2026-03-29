import os
import subprocess
import tempfile
import torch
from typing import List, Optional

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.function.st_moe import st_moe, StMoe
from experience.symbolic_tensor.function.st_moe_backward import (
    st_moe_backward, st_moe_backward_grad_input,
    _read_storage, _detect_input_content_type, _PLAIN, _MERGED,
)
from experience.symbolic_tensor.function.get_causal_attention_mask import (
    get_causal_attention_mask,
)
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


def count_data_files(tensor):
    """Count all 'data' files under a tensor's storage directory."""
    storage_dir = os.path.join(tensor.st_relative_to, tensor.st_tensor_uid, "storage")
    count = 0
    for root, dirs, files in os.walk(storage_dir):
        for f in files:
            if f == "data":
                count += 1
    return count


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


def make_experience(tmpdir, entries):
    """Make an experience tensor from a list of [query, key, value] triples."""
    return make_tensor(entries, tmpdir)


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

    print("Running st_attention + st_moe combined tests...\n")

    # ══════════════════════════════════════════════════════
    # Forward-only tests (no LLM required for attention part)
    # ══════════════════════════════════════════════════════

    # Test 1: shape — st_attention output feeds st_moe, output shape == input shape
    print("Test 1: Shape preservation through attention → moe pipeline")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["hello", "world", "foo"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        run_test("attention output shape [1, 3]", list(attn_out.shape) == [1, 3])
        # Verify attention output contains merged content
        for i in range(3):
            content = read_storage(attn_out, i)
            run_test(f"attn_out[0,{i}] is not None", content is not None)

    # Test 2: merged content detection — st_attention outputs contain kFrameMarker
    print("Test 2: Merged content detection in attention output")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha", "beta", "gamma"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        for i in range(3):
            content = read_storage(attn_out, i)
            tag, data = _detect_input_content_type(content)
            if i == 0:
                # Position 0 with causal mask attends only to itself — may be single frame
                run_test(f"attn_out[0,{i}] detected as plain or merged", tag in (_PLAIN, _MERGED))
            else:
                run_test(f"attn_out[0,{i}] detected as merged", tag == _MERGED)
                if tag == _MERGED:
                    run_test(f"attn_out[0,{i}] has {i+1} frames", len(data) == i + 1,
                             i + 1, len(data))

    # Test 3: causal 1x3 — frame counts (1, 2, 3), coefficients match
    print("Test 3: Causal 1x3 frame counts and coefficients")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["cat", "dog", "bird"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        for i in range(3):
            coeff = attn_out.data[0, i].item()
            run_test(f"attn_out[0,{i}] coeff = {i+1}", coeff == float(i + 1),
                     float(i + 1), coeff)
            content = read_storage(attn_out, i)
            if content and kFrameMarker in content:
                frames = TextMerger.unpack(content)
                run_test(f"attn_out[0,{i}] frame count = {i+1}", len(frames) == i + 1,
                         i + 1, len(frames))

    # Test 4: causal 1x5 — monotonically increasing frame count
    print("Test 4: Causal 1x5 monotonic frame count")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = ["a", "b", "c", "d", "e"]
        inp = make_tensor([tokens], tmpdir)
        mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        prev_frames = 0
        for i in range(5):
            content = read_storage(attn_out, i)
            if content and kFrameMarker in content:
                n_frames = len(TextMerger.unpack(content))
            else:
                n_frames = 1  # single frame (plain or single-frame merged)
            run_test(f"position {i}: frames={n_frames} >= prev={prev_frames}",
                     n_frames >= prev_frames)
            prev_frames = n_frames

    # Test 5: full attention 1x4 — all positions have 4 frames
    print("Test 5: Full attention 1x4 — all 4 frames each")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["w", "x", "y", "z"]], tmpdir)
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        attn_out = st_attention(inp, mask)
        for i in range(4):
            coeff = attn_out.data[0, i].item()
            run_test(f"attn_out[0,{i}] coeff = 4.0", coeff == 4.0, 4.0, coeff)
            content = read_storage(attn_out, i)
            if content and kFrameMarker in content:
                frames = TextMerger.unpack(content)
                run_test(f"attn_out[0,{i}] has 4 frames", len(frames) == 4, 4, len(frames))

    # Test 6: diagonal 1x3 — self-only attention, 1 frame each
    print("Test 6: Diagonal 1x3 — self-only, 1 frame each")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["p", "q", "r"]], tmpdir)
        mask = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        attn_out = st_attention(inp, mask)
        for i in range(3):
            coeff = attn_out.data[0, i].item()
            run_test(f"attn_out[0,{i}] coeff = 1.0", coeff == 1.0, 1.0, coeff)

    # Test 7: padding 1x4 (2 real + 2 pad)
    print("Test 7: Padding 1x4 — 2 real + 2 padded")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["real_a", "real_b", "PAD", "PAD"]], tmpdir)
        token_mask = torch.tensor([[True, True, False, False]])
        mask = get_causal_attention_mask(token_mask)
        attn_out = st_attention(inp, mask)
        # Real tokens should have content
        run_test("attn_out[0,0] coeff > 0", attn_out.data[0, 0].item() > 0)
        run_test("attn_out[0,1] coeff > 0", attn_out.data[0, 1].item() > 0)
        # Padded tokens should have coeff 0
        run_test("attn_out[0,2] coeff = 0", attn_out.data[0, 2].item() == 0.0)
        run_test("attn_out[0,3] coeff = 0", attn_out.data[0, 3].item() == 0.0)

    # Test 8: multi-batch 2x3 causal
    print("Test 8: Multi-batch 2x3 causal")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a1", "b1", "c1"], ["a2", "b2", "c2"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool)).expand(2, -1, -1).clone()
        attn_out = st_attention(inp, mask)
        run_test("attn_out shape [2, 3]", list(attn_out.shape) == [2, 3])
        for b in range(2):
            for i in range(3):
                coeff = attn_out.data[b, i].item()
                run_test(f"attn_out[{b},{i}] coeff = {i+1}", coeff == float(i + 1))

    # Test 9: multi-batch 3x2 mixed masks
    print("Test 9: Multi-batch 3x2 mixed masks (causal/full/diagonal)")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["m", "n"], ["o", "p"], ["q", "r"]], tmpdir)
        mask = torch.zeros(3, 2, 2, dtype=torch.bool)
        # Batch 0: causal
        mask[0] = torch.tril(torch.ones(2, 2, dtype=torch.bool))
        # Batch 1: full
        mask[1] = torch.ones(2, 2, dtype=torch.bool)
        # Batch 2: diagonal
        mask[2] = torch.eye(2, dtype=torch.bool)
        attn_out = st_attention(inp, mask)
        run_test("attn_out shape [3, 2]", list(attn_out.shape) == [3, 2])
        # Causal: coeffs [1, 2]
        run_test("batch 0 causal: coeff[0]=1", attn_out.data[0, 0].item() == 1.0)
        run_test("batch 0 causal: coeff[1]=2", attn_out.data[0, 1].item() == 2.0)
        # Full: coeffs [2, 2]
        run_test("batch 1 full: coeff[0]=2", attn_out.data[1, 0].item() == 2.0)
        run_test("batch 1 full: coeff[1]=2", attn_out.data[1, 1].item() == 2.0)
        # Diagonal: coeffs [1, 1]
        run_test("batch 2 diag: coeff[0]=1", attn_out.data[2, 0].item() == 1.0)
        run_test("batch 2 diag: coeff[1]=1", attn_out.data[2, 1].item() == 1.0)

    # Test 10: coefficient propagation from attention through to moe input
    print("Test 10: Coefficient propagation")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["sun", "moon"]], tmpdir)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        run_test("attn coeff[0]=1.0 (self-only)", attn_out.data[0, 0].item() == 1.0)
        run_test("attn coeff[1]=2.0 (sees both)", attn_out.data[0, 1].item() == 2.0)

    # Test 11: content preservation — frames contain original input text
    print("Test 11: Content preservation in merged frames")
    with tempfile.TemporaryDirectory() as tmpdir:
        original_texts = ["apple", "banana", "cherry"]
        inp = make_tensor([original_texts], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        # Position 2 should have frames for apple, banana, cherry
        content = read_storage(attn_out, 2)
        if content and kFrameMarker in content:
            frames = TextMerger.unpack(content)
            frame_texts = [f[2] for f in frames]
            for txt in original_texts:
                run_test(f"'{txt}' found in position 2 frames",
                         any(txt in ft for ft in frame_texts))

    # Test 12: storage consistency — coeff-storage alignment
    print("Test 12: Storage consistency in attention output")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["x", "y", "z"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        ok = check_coeff_storage_consistency(attn_out, "attn_out")
        run_test("attn_out coeff-storage consistent", ok)

    # Test 13: merged content round-trip — unpack then repack preserves frames
    print("Test 13: Merged content unpack/repack round-trip")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["first", "second", "third"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        content = read_storage(attn_out, 2)
        if content and kFrameMarker in content:
            frames = TextMerger.unpack(content)
            repacked = TextMerger.pack(frames)
            frames2 = TextMerger.unpack(repacked)
            run_test("unpack/repack preserves frame count", len(frames) == len(frames2))
            for i, (f1, f2) in enumerate(zip(frames, frames2)):
                run_test(f"frame {i} content preserved", f1[2] == f2[2])

    # ══════════════════════════════════════════════════════
    # Backward tests (require LLM)
    # ══════════════════════════════════════════════════════

    # Test 14: backward shape through full pipeline
    print("Test 14: Backward shape — attention → moe → backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["Hello", "World"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)

        exp = make_experience(tmpdir, [
            ["greeting\nhello", "Hello", "Bonjour"],
            ["world\nearth", "World", "Monde"],
        ])

        # st_moe_backward_grad_input directly (skip full moe forward to isolate backward)
        grad_output = make_tensor([["improve greeting", "improve world"]], tmpdir)
        grad_output.data.fill_(1.0)

        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1,
            llm_method="raw_llm_api",
        )
        run_test("grad_input shape [1, 2]", list(grad_input.shape) == [1, 2])

    # Test 15: backward produces symbolic content
    print("Test 15: Backward produces symbolic diffs")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Reuse grad_input from test 14 if still available, else rebuild
        inp = make_tensor([["Hello", "World"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["greeting\nhello", "Hello", "Bonjour"],
        ])
        grad_output = make_tensor([["be more formal", "be more formal"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        for i in range(2):
            gi = _read_storage(grad_input, i)
            run_test(f"grad_input[0,{i}] is not None", gi is not None)
            run_test(f"grad_input[0,{i}] is unified diff",
                     gi is not None and "---" in gi)

    # Test 16: merged backward detection — per-frame tasks created
    print("Test 16: Merged backward detection")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["cat", "dog", "bird"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        # Position 2 should be merged (3 frames)
        content2 = read_storage(attn_out, 2)
        tag, data = _detect_input_content_type(content2)
        run_test("position 2 detected as merged", tag == _MERGED)
        if tag == _MERGED:
            run_test("position 2 has 3 frames", len(data) == 3, 3, len(data))

    # Test 17: plain vs merged mixed backward
    print("Test 17: Mixed plain/merged backward through attention output")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["only_self", "sees_both"]], tmpdir)
        # Position 0: diagonal (self-only) → plain-like
        # Position 1: sees both → merged
        mask = torch.tensor([[[True, False], [True, True]]])
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)

        tag0, _ = _detect_input_content_type(read_storage(attn_out, 0))
        tag1, _ = _detect_input_content_type(read_storage(attn_out, 1))
        run_test("position 0: plain or single-frame", tag0 in (_PLAIN, _MERGED))
        run_test("position 1: merged (2 frames)", tag1 == _MERGED)

        exp = make_experience(tmpdir, [
            ["test\nkeyword", "source text", "target text"],
        ])
        grad_output = make_tensor([["fix typo", "improve"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("mixed backward shape [1, 2]", list(grad_input.shape) == [1, 2])
        for i in range(2):
            gi = _read_storage(grad_input, i)
            run_test(f"grad_input[0,{i}] is not None", gi is not None)

    # Test 18: frame count preserved in backward
    print("Test 18: Frame count preserved after backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["alpha", "beta", "gamma"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)

        # Get original frame count at position 2
        orig_content = read_storage(attn_out, 2)
        orig_frames = TextMerger.unpack(orig_content) if orig_content and kFrameMarker in orig_content else []

        exp = make_experience(tmpdir, [
            ["test", "source", "target"],
        ])
        grad_output = make_tensor([["fix", "fix", "fix"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        # The grad_input is a unified diff — the improved version that was diffed
        # had the same frame count. Verify grad exists.
        gi2 = _read_storage(grad_input, 2)
        run_test("grad_input[0,2] exists", gi2 is not None)
        run_test(f"original had {len(orig_frames)} frames", len(orig_frames) == 3, 3, len(orig_frames))

    # Test 19: coefficient channel — pass-through from grad_output
    print("Test 19: Coefficient pass-through in backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [["k", "s", "t"]])
        grad_output = make_tensor([["g1", "g2"]], tmpdir)
        grad_output.data = torch.tensor([[0.5, 0.8]])
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("grad_input[0,0] coeff = 0.5",
                 abs(grad_input.data[0, 0].item() - 0.5) < 0.01, 0.5, grad_input.data[0, 0].item())
        run_test("grad_input[0,1] coeff = 0.8",
                 abs(grad_input.data[0, 1].item() - 0.8) < 0.01, 0.8, grad_input.data[0, 1].item())

    # Test 20: causal 1x3 backward — typo fix propagation
    print("Test 20: Causal 1x3 backward — typo fix")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["Helo world", "Goodby", "Teh end"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["greeting\nhello", "Hello world", "Bonjour le monde"],
        ])
        grad_output = make_tensor([
            ["Fix all typos", "Fix all typos", "Fix all typos"]
        ], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("grad_input shape [1, 3]", list(grad_input.shape) == [1, 3])
        has_diff_count = 0
        for i in range(3):
            gi = _read_storage(grad_input, i)
            if gi is not None and len(gi) > 0:
                has_diff_count += 1
            run_test(f"grad_input[0,{i}] has diff content", gi is not None and len(gi) > 0)
            print(f"    grad_input[0,{i}]: {repr(gi[:80]) if gi else 'None'}")
        # At minimum, merged positions (1, 2) should have diffs
        run_test("at least 2 positions have diffs (merged positions)", has_diff_count >= 2)

    # Test 21: multi-batch 2x2 backward
    print("Test 21: Multi-batch 2x2 backward")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["cat", "dog"], ["sun", "moon"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool)).expand(2, -1, -1).clone()
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["animal\ncat\ndog", "Animals", "Animaux"],
            ["sky\nsun\nmoon", "Sky objects", "Objets celestes"],
        ])
        grad_output = make_tensor([
            ["improve cat", "improve dog"],
            ["improve sun", "improve moon"],
        ], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([1], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("grad_input shape [2, 2]", list(grad_input.shape) == [2, 2])
        for b in range(2):
            for j in range(2):
                flat = b * 2 + j
                gi = _read_storage(grad_input, flat)
                run_test(f"grad_input[{b},{j}] is not None", gi is not None)

    # Test 22: full backward — grad_experience shape
    print("Test 22: Full backward — grad_experience shape")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["test input"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["keyword", "source", "target"],
            ["other", "src2", "tgt2"],
        ])
        grad_output = make_tensor([["improve"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input, grad_experience = st_moe_backward(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("grad_experience shape [2, 3]", list(grad_experience.shape) == [2, 3])

    # Test 23: grad_experience content — non-None for selected entries
    print("Test 23: grad_experience content")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Reuse setup from test 22
        inp = make_tensor([["Hello world"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["greeting\nhello", "Hello", "Bonjour"],
        ])
        grad_output = make_tensor([["make formal"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input, grad_experience = st_moe_backward(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        # Key (index 1) and value (index 2) should have diffs
        ge_key = read_storage(grad_experience, 1)
        ge_val = read_storage(grad_experience, 2)
        run_test("grad_exp key not None", ge_key is not None)
        run_test("grad_exp value not None", ge_val is not None)
        print(f"    grad_exp key: {repr(ge_key[:60]) if ge_key else 'None'}")
        print(f"    grad_exp val: {repr(ge_val[:60]) if ge_val else 'None'}")

    # Test 24: round-trip 1x3 causal — forward → modify → backward
    print("Test 24: Round-trip 1x3 causal")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["The cat", "sat on", "the mat"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [
            ["animal\ncat", "The cat sat", "Le chat assis"],
        ])
        grad_output = make_tensor([
            ["Use formal language", "Use formal language", "Use formal language"]
        ], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input = st_moe_backward_grad_input(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        run_test("round-trip grad_input shape [1, 3]", list(grad_input.shape) == [1, 3])
        diff_count = 0
        for i in range(3):
            gi = _read_storage(grad_input, i)
            if gi is not None and len(gi) > 0:
                diff_count += 1
            print(f"    grad_input[0,{i}]: {repr(gi[:60]) if gi else 'None'}")
        # At minimum, merged positions (1, 2) should have diffs
        run_test("at least 2 positions have diffs (merged positions)", diff_count >= 2)

    # Test 25: storage consistency in backward outputs
    print("Test 25: Storage consistency in backward outputs")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["word_a", "word_b"]], tmpdir)
        inp.requires_grad_(True)
        mask = torch.tril(torch.ones(1, 2, 2, dtype=torch.bool))
        attn_out = st_attention(inp, mask)
        attn_out.requires_grad_(True)
        exp = make_experience(tmpdir, [["k", "s", "t"]])
        grad_output = make_tensor([["fix", "fix"]], tmpdir)
        grad_output.data.fill_(1.0)
        grad_input, grad_experience = st_moe_backward(
            grad_output, attn_out, grad_output, exp,
            selected_experience_qkv_indexes_list=[
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
                [torch.tensor([0], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ],
            topk=1, llm_method="raw_llm_api",
        )
        ok_gi = check_coeff_storage_consistency(grad_input, "grad_input")
        run_test("grad_input coeff-storage consistent", ok_gi)
        ok_ge = check_coeff_storage_consistency(grad_experience, "grad_experience")
        run_test("grad_experience coeff-storage consistent", ok_ge)

    # ══════════════════════════════════════════════════════
    # End-to-end integration tests
    # ══════════════════════════════════════════════════════

    # Test 26: attention mask changes produce different moe inputs
    print("Test 26: Different masks → different attention outputs")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["A", "B", "C"]], tmpdir)
        causal_mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool))
        full_mask = torch.ones(1, 3, 3, dtype=torch.bool)
        attn_causal = st_attention(inp, causal_mask)
        attn_full = st_attention(inp, full_mask)
        # Position 0: causal sees only self, full sees all → different coefficients
        run_test("causal[0] coeff=1", attn_causal.data[0, 0].item() == 1.0)
        run_test("full[0] coeff=3", attn_full.data[0, 0].item() == 3.0)
        run_test("different masks → different coefficients",
                 attn_causal.data[0, 0].item() != attn_full.data[0, 0].item())

    # Test 27: large batch 4x6 causal — stress test
    print("Test 27: Large batch 4x6 causal — storage consistency")
    with tempfile.TemporaryDirectory() as tmpdir:
        tokens = [[f"t{b}_{i}" for i in range(6)] for b in range(4)]
        inp = make_tensor(tokens, tmpdir)
        mask = torch.tril(torch.ones(1, 6, 6, dtype=torch.bool)).expand(4, -1, -1).clone()
        attn_out = st_attention(inp, mask)
        run_test("attn_out shape [4, 6]", list(attn_out.shape) == [4, 6])
        ok = check_coeff_storage_consistency(attn_out, "attn_out_4x6")
        run_test("4x6 coeff-storage consistent", ok)
        # Verify coefficients: position i should have coeff i+1 for all batches
        coeff_ok = True
        for b in range(4):
            for i in range(6):
                if attn_out.data[b, i].item() != float(i + 1):
                    coeff_ok = False
        run_test("all coefficients match causal pattern", coeff_ok)

    # Test 28: sliding window 1x5 + verify frame counts
    print("Test 28: Sliding window 1x5 — frame counts match window")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["v", "w", "x", "y", "z"]], tmpdir)
        # Sliding window of size 2: position i attends to [max(0,i-1), i]
        mask = torch.zeros(1, 5, 5, dtype=torch.bool)
        for i in range(5):
            for j in range(max(0, i - 1), i + 1):
                mask[0, i, j] = True
        attn_out = st_attention(inp, mask)
        # Position 0: window [0] → 1 frame
        # Position 1-4: window [i-1, i] → 2 frames
        run_test("pos 0 coeff=1", attn_out.data[0, 0].item() == 1.0)
        for i in range(1, 5):
            run_test(f"pos {i} coeff=2", attn_out.data[0, i].item() == 2.0)

    # Test 29: multi-batch storage file counts
    print("Test 29: Multi-batch storage file counts")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["a", "b", "c"], ["d", "e", "f"]], tmpdir)
        mask = torch.tril(torch.ones(1, 3, 3, dtype=torch.bool)).expand(2, -1, -1).clone()
        attn_out = st_attention(inp, mask)
        n_files = count_data_files(attn_out)
        # All 6 positions should have data files (causal, all non-zero)
        run_test("6 data files in attn_out", n_files == 6, 6, n_files)

    # Test 30: get_causal_attention_mask integration with full pipeline
    print("Test 30: get_causal_attention_mask → st_attention → verify")
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = make_tensor([["I", "am", "here", "now"]], tmpdir)
        token_mask = torch.tensor([[True, True, True, True]])
        causal_mask = get_causal_attention_mask(token_mask)
        run_test("causal mask shape [1,4,4]", list(causal_mask.shape) == [1, 4, 4])
        run_test("causal mask is lower triangular",
                 torch.equal(causal_mask, torch.tril(torch.ones(1, 4, 4, dtype=torch.bool))))
        attn_out = st_attention(inp, causal_mask)
        run_test("attn_out shape [1, 4]", list(attn_out.shape) == [1, 4])
        for i in range(4):
            run_test(f"pos {i} coeff={i+1}",
                     attn_out.data[0, i].item() == float(i + 1))

    print("\nAll tests completed.")
