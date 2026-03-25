# Symbolic Tensor

A PyTorch extension that enables **symbolic text operations** within neural network computation graphs. Each tensor element is backed by a file on disk storing arbitrary text (code, translations, etc.), while the numeric tensor coefficients flow through standard autograd. LLM agents perform the actual "computation" on text content during forward/backward passes.

## Core Idea

Traditional tensors store numbers. **Symbolic tensors store text** — each element is a file on disk, and the numeric coefficient (default `1`) acts as a signal strength indicator. This allows building trainable models where the "weights" are natural language mappings (e.g., translation pairs) that are updated by an LLM-based optimizer.

```
Standard Tensor:    [0.5, 0.3, 0.8]           -> numbers
Symbolic Tensor:    ["hello world", "bonjour"]   -> text files + coefficients
```

## Architecture

```
experience/symbolic_tensor/
├── tensor_util/         # Low-level tensor primitives
│   ├── make_tensor          # Create symbolic tensor from nested strings/Paths
│   ├── make_none_tensor     # Create zero-filled symbolic tensor (placeholder)
│   ├── empty_tensor_like    # Create empty-string-filled tensor matching shape
│   ├── todo_tensor_like     # Create TODO-filled tensor matching shape
│   ├── load_tensor          # Restore tensor from dumped directory
│   ├── dump_tensor          # Serialize tensor storage + metadata
│   ├── dump_view            # Create coordinate-based symlink views for LLM
│   ├── slice_view           # Slice via symlinks (shared storage)
│   ├── slice_tensor         # Slice via file copies (independent storage)
│   ├── pack_tensor          # Pack tensor into a single string representation
│   ├── assign_tensor        # Assign values to tensor elements by coordinates
│   ├── get_diff_tensor      # Compute unified diff between two tensors
│   ├── patch_tensor         # Apply patch to tensor using git apply
│   ├── register_tensor_ops  # Register custom ops on torch.Tensor
│   └── st_copy              # Deep copy with autograd support
├── function/            # autograd.Function implementations
│   ├── symbolic_transform              # Dual-channel forward/backward wrapper
│   ├── symbolic_transform_forward      # Forward: input -> output via LLM + experience
│   ├── symbolic_transform_backward     # Backward: compute symbolic gradients via LLM
│   ├── select_qkv_indexes              # Jaccard similarity-based experience retrieval
│   ├── get_input_query_tensor          # LLM-generated query keywords per element
│   ├── get_edit_distance_ratio         # Text similarity loss (Levenshtein-based)
│   ├── symbolic_grad_registry          # Thread-local metadata pass-through between autograd Functions
│   └── test/                          # Benchmarks
│       └── test_transform_method_time_comparison  # coding_agent vs raw_llm_api benchmark
├── module/              # torch.nn.Module wrappers
│   └── symbolic_transform  # SymbolicTransformModule (like nn.Linear for text)
├── optimizer/           # Training optimizers
│   └── symbolic_sgd     # Patch-based SGD: diff experience, apply patches
├── llm_client/          # LLM backend interface (two methods)
│   ├── task_handler              # Dispatches tasks to the selected LLM method
│   ├── agent_task                # AgentTask dataclass: unit of work for LLM
│   ├── coding_agent_query        # Async Claude Agent SDK wrapper
│   ├── coding_agent_task_handler # Dispatches to Claude coding agent (file system access)
│   ├── raw_llm_query             # Async OpenAI-compatible API call
│   ├── raw_llm_task_handler      # Dispatches via raw LLM API (prompt-based)
│   └── pack_dir                  # Packs directory into single string for raw LLM context
├── sparse_util/         # Sparse coordinate utilities
│   ├── group_random_select                                   # Random selection within groups
│   ├── convert_nested_list_coordinates_to_pairs_coordinates # Nested list <-> flat coordinate pairs
│   └── transpose_pairs_coordinates                          # Transpose sparse coordinate matrices
├── fs_util/             # File system utilities
│   └── get_nested_list_file_pathes  # Enumerate file paths matching nested list structure
├── data_loader/         # Dataset utilities
│   └── sole_file_batch_data_loader  # Load files into symbolic tensors
└── example/             # End-to-end example
    └── naive_symbolic_transform_model/
        ├── train.py      # Training loop: Python -> Viba translation
        ├── model.py      # NaiveModel wrapping SymbolicTransformModule
        └── dataset/      # 12 Python/Viba code pairs
```

## Viba: The Spec Language

Each `.py` module has a companion `.viba` file that serves as a **design-time specification** written in the Viba pattern-matching language. These `.viba` files describe the intended logic in a declarative style — they are not executed at runtime, but guide implementation and regeneration.

Viba syntax highlights:
- `<-` for variable binding / return
- `$var` for variable references with type annotations
- `:=` for type/function definitions
- Sum types with `|` for branching
- `Match[condition -> value, ...]` for pattern matching
- `Import[...]` for referencing other modules
- `Object * field type` for dataclass-like structs
- `# inline` for inlining a function body

The `.viba` files in `example/naive_symbolic_transform_model/dataset/` are actual Viba code samples used as **translation targets** in the training demo.

## Dual-Channel Gradient System

Symbolic tensors propagate gradients through **two channels**:

| Channel | What it carries | How it's computed |
|---------|----------------|-------------------|
| **Numeric** (coefficient) | Float values (bfloat16) | Standard autograd / SGD arithmetic |
| **Symbolic** (text) | Unified diffs stored in files | LLM computes `diff -u` between actual and expected |

The `symbolic_grad_registry` (thread-local dictionary) passes symbolic gradient metadata between autograd Function backward calls, since PyTorch autograd strips custom tensor attributes (`st_relative_to`, `st_tensor_uid`) when propagating gradients between Function nodes.

## LLM Backend Methods

Two LLM backends are supported via the `TransformMethod` enum:

### `coding_agent` (default)
Uses Claude Agent SDK with `Read`, `Edit`, `Write` tool access. The agent can directly read context files and modify output files in the workspace. Best for complex tasks requiring file system interaction.

### `raw_llm_api`
Uses OpenAI-compatible API (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` env vars). Packs directory contents into a prompt via `pack_dir`, finds files containing the TODO placeholder, and replaces their content with LLM responses. Lighter weight, no tool access.

Both methods are dispatched through `TaskHandler`, which takes a list of `AgentTask` objects and runs them concurrently via `asyncio.gather`.

## Experience

An **Experience** is a symbolic tensor of shape `[N, 3]` where each row is a `(query, key, value)` triple:
- **Query** (position 0): Semantic keywords (one per line) used for Jaccard similarity retrieval
- **Key** (position 1): Source domain content (e.g., Python code)
- **Value** (position 2): Target domain content (e.g., Viba code)

Formally defined in `tensor_util/experience.viba`:
```viba
Experience[Tensor] := $tensor Tensor * Constraints[
    Assert[$tensor.shape[-1] == 3],
    IsQueryFile[$tensor[..., 0]],
    IsKeyFile[$tensor[..., 1]],
    IsValueFile[$tensor[..., 2]],
]
```

Experience acts as the learnable "weight" of the model. It starts empty and is populated during training — the backward pass computes diffs against the expected output, and the optimizer applies patches to experience entries via `git apply`.

## Demo: Python to Viba Translation

This example trains a model from scratch to translate Python code into Viba. The experience starts **empty** and is learned entirely during training.

### 1. Dataset

12 Python-to-Viba translation pairs covering various patterns:

| Pattern | Python | Viba |
|---------|--------|------|
| Sequential | `seq.py` | `seq.viba` |
| Branching | `branch.py` | `branch.viba` |
| Loop | `loop.py` | `loop.viba` |
| Recursion | `recursion.py` | `recursion.viba` |
| Higher-order | `higher_order.py` | `higher_order.viba` |
| Data structures | `data_struct.py` | `data_struct.viba` |
| Default args | `default_arg.py` | `default_arg.viba` |
| List comprehension | `list_comp.py` | `list_comp.viba` |
| String formatting | `format_str.py` | `format_str.viba` |
| Guard clauses | `guard.py` | `guard.viba` |
| Accumulator | `accumulator.py` | `accumulator.viba` |
| Closure | `closure.py` | `closure.viba` |

### 2. Setup

```python
import tempfile
from pathlib import Path

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio
from experience.symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD
from experience.symbolic_tensor.example.naive_symbolic_transform_model.model import NaiveModel

DATASET_PAIRS = [
    "seq", "branch", "loop",
    "recursion", "higher_order", "data_struct",
    "default_arg", "list_comp", "format_str",
    "guard", "accumulator", "closure",
]
```

### 3. Build Tensors

```python
with tempfile.TemporaryDirectory() as tmpdir:
    # Input: symlinks to Python source files
    py_paths = [Path("dataset") / f"{name}.py" for name in DATASET_PAIRS]
    input_tensor = make_tensor(py_paths, tmpdir, symlink=True)

    # Expected: Viba code as strings
    viba_contents = [(Path("dataset") / f"{name}.viba").read_text() for name in DATASET_PAIRS]
    expected_tensor = make_tensor(viba_contents, tmpdir)

    # Experience: starts EMPTY — learned during training
    n = len(DATASET_PAIRS)
    experience_tensor = make_tensor([[""] * 3 for _ in range(n)], tmpdir)
```

### 4. Training Loop

```python
    model = NaiveModel(forward_prompt="Translate Python To Viba", topk=1)
    model.load_experience(experience_tensor)
    optimizer = SymbolicSGD(model.parameters(), lr=1.0)

    for iteration in range(1, 6):
        optimizer.zero_grad()

        # Forward: LLM translates each Python file to Viba using experience
        output, selected_indexes = model(input_tensor)

        # Loss: Levenshtein edit distance ratio
        loss = get_edit_distance_ratio(output, expected_tensor)
        mean_loss = loss.mean().item()
        print(f"Iteration {iteration} loss: {mean_loss:.4f}")

        # Backward: computes symbolic gradients (diffs) via autograd
        loss.mean().backward()

        # Optimizer step: applies patches to experience via git apply
        optimizer.step()
```

### 5. Run

```bash
# Requires ANTHROPIC_API_KEY or LLM_API_KEY/LLM_BASE_URL/LLM_MODEL env vars
python -m experience.symbolic_tensor.example.naive_symbolic_transform_model.train
```

Output:
```
============================================================
NaiveModel Training: Translate Python To Viba
============================================================

Dataset: 12 pairs
  [0] seq.py -> seq.viba
  [1] branch.py -> branch.viba
  ...
  [11] closure.py -> closure.viba

Experience: [12, 3]
Input:      [12]
Expected:   [12]

────────────────────────────────────────────────────────────
Iteration 1/5
────────────────────────────────────────────────────────────

  [Forward]
    output[0]: 'classify :=\n    | x > 0 -> "positive"\n    | ...'
    ...

  [Loss]
    Per-sample: ['0.4500', '0.3200', ...]
    Mean: 0.3850

  [Backward]
    grad_exp[0]: '--- a/0\n+++ b/0\n@@ -1,3 +1,4 @@\n...'

  [Step]
    Patches: applied=8 rejected=2 fuzzed=1 skipped=3
    Experience after step:
      [0]: 'branch\npython\nviba'
      [1]: 'def classify(x):\n    if x > 0: ...'
      ...

...

============================================================
Training Complete
============================================================

Loss trajectory: ['0.3850', '0.3100', '0.2400', '0.1800', '0.1200']
Loss CONVERGED (final < initial)
```

### 6. How Training Works

1. **Forward**: For each Python input, the LLM retrieves relevant experience entries (by Jaccard similarity), reads the key-value mappings, and translates the input to Viba.
2. **Loss**: Edit distance ratio measures how close the LLM output is to the expected Viba code.
3. **Backward**: The backward pass computes `diff -u` between actual and expected outputs, producing symbolic gradients stored as unified diffs.
4. **Optimizer**: `SymbolicSGD` applies these diffs as patches to experience entries using `git apply`, with stats tracking (applied, rejected, fuzzed).

## How It Works

### Storage Layout

Each symbolic tensor stores its text content on disk:

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # Element at flat index 0
│   ├── 1/data               # Element at flat index 1
│   ├── ...
│   └── 1/1/data             # Multi-digit index 11
```

### Forward Pass Pipeline

1. **Query Generation**: LLM extracts semantic keywords from each input element
2. **Experience Retrieval**: Jaccard similarity selects top-k relevant experience entries (with Gaussian noise for exploration)
3. **Context Assembly**: Dump symlink views of experience (query/key/value) and input
4. **Task Dispatch**: `TaskHandler` creates `AgentTask` objects and dispatches to the selected LLM backend
5. **LLM Translation**: Agent reads context and writes output to mutable copies
6. **Copy-back**: Results propagate through symlinks to parent tensor storage

### Backward Pass Pipeline

1. **Loss Backward**: `get_edit_distance_ratio` produces unified diffs as symbolic gradients
2. **Transform Backward**: Numeric coefficient pass-through + scatter-add to experience; symbolic gradients propagated via `symbolic_grad_registry`
3. Both channels merge into the gradient tensor

### Optimizer Pipeline

1. **Diff**: Compute `get_diff_tensor` between current experience and gradient (which contains target content)
2. **Patch**: Apply `patch_tensor` via `git apply` to update experience storage files
3. **Stats**: Track patch application results (applied, rejected, fuzzed, skipped)

## Key Design Decisions

- **Symlinks for views, copies for mutations**: `slice_view` creates symlinks (shared storage, read-only context); `slice_tensor` creates independent copies (safe for LLM writes)
- **Coordinate-based views**: `dump_view` maps multi-dimensional indices to human-readable paths (e.g., `0/1/data.txt`) for LLM consumption
- **Two LLM backends**: `coding_agent` (Claude Agent SDK with tools) vs `raw_llm_api` (prompt-based, OpenAI-compatible) — switchable via `method` parameter
- **Patch-based optimizer**: Instead of regenerating experience from scratch, `SymbolicSGD` applies `git diff`/`git apply` patches for efficient incremental updates
- **Symbolic grad registry**: Thread-local dict bridges autograd Function calls that lose custom tensor attributes
- **LLM as compute kernel**: The LLM replaces traditional matrix multiplication with semantic reasoning

## Dependencies

- Python 3.13+
- PyTorch
- `claude-agent-sdk` (for `coding_agent` method)
- `openai` (for `raw_llm_api` method)
- `Levenshtein` (edit distance computation)

## Installation

```bash
pip install torch claude-agent-sdk openai Levenshtein
```

## Quick Start

```python
from experience.symbolic_tensor import tensor, none

# Create a symbolic tensor
t = tensor(["hello world", "bonjour le monde"], "/tmp/my_tensors")

print(t.shape)           # torch.Size([2])
print(t.data)            # tensor([1., 1.], dtype=torch.bfloat16)
print(t.st_relative_to)  # '/tmp/my_tensors'
print(t.st_tensor_uid)   # 'a3f2...'

# Read text content
import os
path = os.path.join(t.st_relative_to, t.st_tensor_uid, "storage", "0", "data")
with open(path) as f:
    print(f.read())      # "hello world"
```
