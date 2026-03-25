# Experience

A PyTorch extension that replaces numeric matrix multiplication with **LLM-powered text computation** inside neural network training loops. Each tensor element stores arbitrary text (code, translations, etc.) in files on disk, while numeric coefficients flow through standard autograd. ExperienceTensor — a key-value store shaped as `[N, 3]` — serves as the learnable "weight" of the model, starting empty and being built up entirely by a patch-based optimizer across training iterations. The LLM acts as the compute kernel: during forward pass it reads experience entries to produce outputs; during backward pass it computes diffs; during the optimizer step `git apply` patches experience entries incrementally.

The result is a model that can **learn to perform tasks it was never trained on** by accumulating experience at runtime, demonstrated by translating Python into Viba — a novel DSL that does not exist in the training corpus of any existing LLM.

## Architecture

```
experience/
├── symbolic_tensor/     # Core tensor primitives, autograd functions, module, optimizer
├── llm_client/          # LLM backends: coding agent (Claude SDK) and raw API (OpenAI-compatible)
├── sparse_util/         # Sparse coordinate operations
├── fs_util/             # File system utilities (directory packing, path enumeration)
├── test/                # Integration tests and benchmarks
└── example/             # End-to-end training demo
```

## Dual-Channel Gradient System

Gradients propagate through two channels simultaneously:

| Channel | What it carries | How it's computed |
|---------|----------------|-------------------|
| **Numeric** (coefficient) | Float values (bfloat16) | Standard autograd / SGD arithmetic |
| **Symbolic** (text) | Unified diffs stored in files | LLM computes `diff -u` between actual and expected |

The `symbolic_grad_registry` (thread-local dictionary) passes symbolic gradient metadata between autograd Function backward calls, since PyTorch autograd strips custom tensor attributes (`st_relative_to`, `st_tensor_uid`) when propagating gradients between Function nodes.

## LLM Backend Methods

Two backends are supported via the `TransformMethod` enum:

### `raw_llm_api` (default)
OpenAI-compatible API (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`). Packs directory contents into a prompt, finds files containing the TODO placeholder, and replaces them with LLM responses. Lightweight, no tool access.

### `coding_agent`
Claude Agent SDK with `Read`, `Edit`, `Write` tool access. The agent can directly read and modify files in the workspace. Best for complex tasks requiring file system interaction.

Both are dispatched through `TaskHandler`, which takes `AgentTask` objects and runs them concurrently via `asyncio.gather`.

## ExperienceTensor

An **ExperienceTensor** is a symbolic tensor of shape `[N, 3]` where each row is a `(query, key, value)` triple:
- **Query** (position 0): Semantic keywords (one per line) used for Jaccard similarity retrieval
- **Key** (position 1): Source domain content (e.g., Python code)
- **Value** (position 2): Target domain content (e.g., code in another language)

It acts as the learnable "weight" of the model — starts empty and is populated during training. The backward pass computes diffs against expected output, and the optimizer applies patches to experience entries via `git apply`.

## Demo: Python to Viba Translation

This example trains a model from scratch to translate Python code into **Viba** — a novel domain-specific language invented for this project that does not exist in the training corpus of any existing LLM. The LLM must learn to generate syntactically and semantically correct code in a language it has never seen, purely from the experience entries built up during training.

### Why Viba?

Existing LLMs have been trained on billions of lines of Python, JavaScript, Haskell, etc. Translating between these tests whether the model can *recall* patterns from pre-training. Viba eliminates this confound — any correct translation demonstrates genuine *generalization* driven by the experience mechanism.

Viba features:
- `<-` for variable binding / return
- `$var` for variable references with type annotations
- `:=` for type/function definitions
- Sum types with `|` for branching (replacing if/elif/else)
- `Match[condition -> value, ...]` for pattern matching
- List comprehensions: `list[$doubled <- $item * 2]`

### 1. Dataset

12 Python-to-Viba translation pairs:

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
python -m experience.symbolic_tensor.example.naive_symbolic_transform_model.train
```

## How It Works

### Storage Layout

```
{relative_to}/{tensor_uid}/
├── shape                    # JSON: [2, 3]
├── storage/
│   ├── 0/data               # Element at flat index 0
│   ├── 1/data               # Element at flat index 1
│   └── 1/1/data             # Multi-digit index 11
```

### Forward Pass

1. **Query Generation**: LLM extracts semantic keywords from each input element
2. **Experience Retrieval**: Jaccard similarity selects top-k relevant experience entries
3. **Context Assembly**: Dump symlink views of experience and input
4. **Task Dispatch**: `TaskHandler` dispatches `AgentTask` objects to the LLM backend
5. **Copy-back**: Results propagate through symlinks to parent storage

### Backward Pass

1. `get_edit_distance_ratio` produces unified diffs as symbolic gradients
2. Numeric coefficients pass through; symbolic gradients propagated via `symbolic_grad_registry`

### Optimizer

`SymbolicSGD` computes `get_diff_tensor` then applies `patch_tensor` via `git apply` — incremental patch-based updates instead of full regeneration.

## Key Design Decisions

- **LLM as compute kernel**: Replaces matrix multiplication with semantic reasoning
- **Patch-based optimizer**: `git diff`/`git apply` for efficient incremental experience updates
- **Two LLM backends**: `raw_llm_api` (default, lightweight) and `coding_agent` (tool access)
- **Symlinks for views, copies for mutations**: Shared storage for context, independent copies for LLM writes
- **Experience starts empty**: Learned entirely at runtime, not pre-seeded

## Dependencies

- Python 3.13+
- PyTorch
- `openai` (default LLM backend)
- `claude-agent-sdk` (alternative LLM backend)
- `Levenshtein` (edit distance loss)

## Installation

```bash
pip install torch openai claude-agent-sdk Levenshtein
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
