"""Auto-encoder cloze experiment generated from claude.viba.

Viba DSL specification:
  kMaskedHint := "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

  task_description := void <- $total_batch_size int <- $workspace_dir (str | None)
    <- PrepareDataSet -> BaselineCodingAgentModel -> get_edit_distance
    <- { print $baseline_output.st_uid ground_truth_content_tensor.st_uid }

  BaselineCodingAgentModel[torch.nn.Module] :=
    $baseline_output SymbolicTensor[($total_batch_size,)]
    <- $masked_file_path_tensor SymbolicTensor[($total_batch_size, $num_files)]
    <- $masked_file_content_tensor SymbolicTensor[($total_batch_size, $num_files)]
    <- $ground_truth_content_tensor SymbolicTensor[($total_batch_size,)]

  PrepareDataSet :=
    ($masked_file_path_tensor, $masked_file_content_tensor, $ground_truth_content_tensor)
    <- $total_batch_size int <- $dataset_dir
"""

import os
import random
import tempfile
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.tensor_util.slice_tensor import slice_tensor
from experience.symbolic_tensor.function.st_stack import st_stack_forward
from experience.symbolic_tensor.function.merge_forward import merge_forward
from experience.symbolic_tensor.function.st_attention import st_attention
from experience.symbolic_tensor.function.coding_agent import coding_agent
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

kMaskedHint = "<AUTOENCODER-CLOZE-MASK-PLACEHOLDER>"

EXPERIENCE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "experience")

# Truncate non-masked file contents to this many lines to fit LLM context
_MAX_CONTEXT_LINES = 15


def _read_storage(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the content of a symbolic tensor element from disk."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to, tensor.st_tensor_uid,
        "storage", os.path.join(*digits), "data",
    )
    with open(path) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════════
# PrepareDataSet
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_all_files(root_dir: str) -> List[Tuple[str, str]]:
    """Fetch all .py files as (relative_path, content)."""
    files = []
    for dirpath, _dirs, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if content.strip():
                files.append((rel, content))
    return sorted(files)


def _find_maskable_range(content: str) -> Tuple[int, int]:
    """Find the range of lines eligible for masking: skip comments and __main__ block."""
    lines = content.splitlines(keepends=True)
    main_start = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("if __name__"):
            main_start = i
            break
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("import ") and not stripped.startswith("from "):
            code_start = i
            break
    return code_start, main_start


def _get_random_mask_range(content: str, min_size: int = 20) -> Tuple[int, int, str]:
    """Pick random line range to mask. Skips comments and __main__ block. Range >= min_size."""
    lines = content.splitlines(keepends=True)
    code_start, main_start = _find_maskable_range(content)
    maskable_len = main_start - code_start
    if maskable_len < min_size:
        mask_len = max(1, maskable_len)
        start = code_start
    else:
        mask_len = random.randint(min_size, min(maskable_len, max(min_size, maskable_len // 2)))
        start = random.randint(code_start, main_start - mask_len)
    end = start + mask_len
    return start, end, "".join(lines[start:end])


def _apply_mask(content: str, start: int, end: int) -> str:
    """Replace lines [start, end) with kMaskedHint."""
    lines = content.splitlines(keepends=True)
    return "".join(lines[:start] + [kMaskedHint + "\n"] + lines[end:])


def _truncate_content(content: str, max_lines: int) -> str:
    """Truncate to first max_lines lines, append ellipsis if truncated."""
    lines = content.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return content
    return "".join(lines[:max_lines]) + "...\n"


def prepare_dataset(
    total_batch_size: int,
    dataset_dir: str,
    tmpdir: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """PrepareDataSet from claude.viba.

    Returns:
        masked_file_path_tensor:    Symbolic[Tensor(batch, num_files)]
        masked_file_content_tensor: Symbolic[Tensor(batch, num_files)]
        ground_truth_tensor:        Symbolic[Tensor(batch,)]
        file_info: list of "file_path:start-end" for logging
    """
    # <- ($files list[($file_path str, $file_content)] <- { fetch all files in experience/ })
    files = _fetch_all_files(dataset_dir)
    files = [(fp, fc) for fp, fc in files
             if _find_maskable_range(fc)[1] - _find_maskable_range(fc)[0] >= 20]
    num_files = len(files)
    assert num_files > 0, f"No sufficiently large .py files in {dataset_dir}"

    # <- {num_files = len(files)}
    file_paths = [fp for fp, _ in files]
    file_contents = [fc for _, fc in files]

    all_masked_paths = []
    all_masked_contents = []
    all_ground_truths = []
    file_info = []

    # <- (list[($random_file_index int, $random_range_line_start int, $random_range_line_end int)]
    #       <- { get multiple random ranges ... } <- $total_batch_size <- $num_files)
    for _ in range(total_batch_size):
        file_idx = random.randint(0, num_files - 1)
        fp, fc = files[file_idx]
        start, end, ground_truth = _get_random_mask_range(fc)
        masked_content = _apply_mask(fc, start, end)

        batch_paths = list(file_paths)
        batch_contents = []
        for f_i, fc_i in enumerate(file_contents):
            if f_i == file_idx:
                # Masked file: full content with mask
                batch_contents.append(masked_content)
            else:
                # Non-masked: truncate to save tokens for st_attention merge
                batch_contents.append(_truncate_content(fc_i, _MAX_CONTEXT_LINES))

        all_masked_paths.append(batch_paths)
        all_masked_contents.append(batch_contents)
        all_ground_truths.append(ground_truth)
        file_info.append(f"{fp}:{start+1}-{end}")

    # Create symbolic tensors
    masked_path_tensor = make_tensor(all_masked_paths, tmpdir)
    masked_content_tensor = make_tensor(all_masked_contents, tmpdir)
    gt_tensor = make_tensor(all_ground_truths, tmpdir)

    return masked_path_tensor, masked_content_tensor, gt_tensor, file_info


# ═══════════════════════════════════════════════════════════════════════════════
# BaselineCodingAgentModel[torch.nn.Module]
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineCodingAgentModel(nn.Module):
    """BaselineCodingAgentModel from claude.viba.

    Pipeline:
      1. Stack (path, content) → (batch, num_files, 2)
      2. merge(axis=-1) → (batch, num_files)
      3. st_attention(prefix→last mask) → (batch, num_files)
      4. slice last col → (batch,)
      5. coding_agent → (batch,)
    """

    def __init__(self, llm_method: str = "raw_llm_api"):
        super().__init__()
        self.llm_method = llm_method

    def forward(
        self,
        masked_path_tensor: torch.Tensor,
        masked_content_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass implementing the viba pipeline."""
        batch_size, num_files = masked_path_tensor.shape
        tmpdir = masked_path_tensor.st_relative_to

        # Step 1: st_stack(path, content, dim=-1) → (batch, num_files, 2)
        # $path_and_contents <- Import[function/merge] <- { concat on last axis }
        stacked_tensor = st_stack_forward(
            [masked_path_tensor, masked_content_tensor], dim=-1,
        )
        # shape: (batch, num_files, 2)

        # Step 2: merge(axis=-1) → (batch, num_files)
        # Merges path+content per file via TextMerger.pack
        path_and_contents = merge_forward(stacked_tensor, axis=-1)
        # shape: (batch, num_files)

        # Step 3: st_attention with prefix→last mask
        # $merged_path_and_contents <- { fetch last column }
        #   <- Import[function/st_attention] <- { prepare special attention mask to concat prefixes to last column }
        attention_mask = torch.eye(num_files, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1).clone()
        attention_mask[:, num_files - 1, :] = True
        merged = st_attention(path_and_contents, attention_mask, return_view=True)
        # shape: (batch, num_files) — last col has all files merged

        # Step 4: slice last column → (batch,)
        last_col_idx = [slice(None), num_files - 1]
        merged_context = slice_tensor(merged, last_col_idx)

        # Step 5: coding_agent
        # $prompt <- { autoencoder prompt for baseline model }
        task_prompt = (
            "You are an auto-encoder for source code.\n"
            f"The input contains Python files from a repository packed with frame markers.\n"
            f"One file has a masked region marked by {kMaskedHint}.\n"
            "Your task: reconstruct ONLY the masked lines.\n"
            "Output ONLY the missing source code lines. No explanations."
        )

        # $baseline_output <- Import[function/coding_agent.viba]
        #   <- $merged_path_and_contents <- $prompt <- $llm_method "raw_llm_api"
        output = coding_agent(
            merged_context,
            task_prompt=task_prompt,
            llm_method=self.llm_method,
        )
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# task_description (main experiment entry point)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    total_batch_size: int = 16,
    workspace_dir: Optional[str] = None,
    llm_method: str = "raw_llm_api",
) -> float:
    """task_description from claude.viba.

    Args:
        total_batch_size: default 16
        workspace_dir: None means temp directory
        llm_method: LLM backend to use

    Returns:
        Mean edit distance ratio (lower is better)
    """
    root_dir = os.path.realpath(EXPERIENCE_ROOT)
    print(f"Dataset: {root_dir}")

    # <- $workspace_dir (str | None) # None means temp directory
    if workspace_dir is None:
        tmpdir = tempfile.mkdtemp()
    else:
        tmpdir = workspace_dir
        os.makedirs(tmpdir, exist_ok=True)

    # <- PrepareDataSet
    masked_path_tensor, masked_content_tensor, gt_tensor, file_info = prepare_dataset(
        total_batch_size, root_dir, tmpdir,
    )
    print(f"Batch={total_batch_size}, files={masked_path_tensor.shape[1]}")
    for i, info in enumerate(file_info):
        gt_preview = _read_storage(gt_tensor, i)[:60].replace("\n", "\\n")
        print(f"  [{i}] {info} -> {gt_preview}...")

    # <- BaselineCodingAgentModel
    print(f"\nRunning BaselineCodingAgentModel (llm_method={llm_method})...")
    model = BaselineCodingAgentModel(llm_method=llm_method)
    output = model(masked_path_tensor, masked_content_tensor)

    # <- ($baseline_loss <- Import[function/get_edit_distance] <- $baseline_output <- $ground_truth_content_tensor)
    loss = get_edit_distance_ratio_impl(output, gt_tensor)

    # <- { print $baseline_output.st_uid ground_truth_content_tensor.st_uid }
    print(f"\noutput tensor uid: {output.st_tensor_uid}")
    print(f"ground_truth tensor uid: {gt_tensor.st_tensor_uid}")

    print(f"\n{'='*50}")
    print(f"Results (edit_distance_ratio, lower=better):")
    print(f"{'='*50}")
    for i in range(total_batch_size):
        actual = _read_storage(output, i)
        gt = _read_storage(gt_tensor, i)
        r = loss[i].item()
        print(f"  [{i}] {file_info[i]}  loss={r:.4f}")
        print(f"       gt:     {gt[:80].replace(chr(10), '\\n')}")
        print(f"       actual: {actual[:80].replace(chr(10), '\\n')}")

    mean_loss = loss.float().mean().item()
    print(f"\nMean loss: {mean_loss:.4f}")
    return mean_loss


# ═══════════════════════════════════════════════════════════════════════════════
# __main__ test block
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess

    # Setup environment for LLM API
    result = subprocess.run(
        ["bash", "-c", "source ~/.LLM_config.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    # Run the experiment
    run_experiment(total_batch_size=16, llm_method="raw_llm_api")
