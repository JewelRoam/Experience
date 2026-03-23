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
symbolic_tensor/
├── tensor_util/         # Low-level tensor primitives
│   ├── make_tensor      # Create symbolic tensor from nested strings/Paths
│   ├── make_none_tensor # Create zero-filled symbolic tensor (placeholder)
│   ├── load_tensor      # Restore tensor from dumped directory
│   ├── dump_tensor      # Serialize tensor storage + metadata
│   ├── dump_view        # Create coordinate-based symlink views for LLM
│   ├── slice_view       # Slice via symlinks (shared storage)
│   ├── slice_tensor     # Slice via file copies (independent storage)
│   ├── todo_tensor_like # Create TODO-filled tensor matching input shape
│   └── copy             # Deep copy with autograd support
├── function/            # autograd.Function implementations
│   ├── symbolic_transform         # Dual-channel forward/backward wrapper
│   ├── symbolic_transform_forward # Forward: input -> output via LLM + experience
│   ├── symbolic_transform_backward# Backward: compute symbolic gradients via LLM
│   ├── select_qkv_indexes         # Jaccard similarity-based experience retrieval
│   ├── get_input_query_tensor     # LLM-generated query keywords per element
│   ├── get_edit_distance_ratio    # Text similarity loss (Levenshtein-based)
│   └── copy                      # Tensor copy with gradient passthrough
├── module/              # torch.nn.Module wrappers
│   └── symbolic_transform # SymbolicTransformModule (like nn.Linear for text)
├── optimizer/           # Training optimizers
│   └── symbolic_sgd     # Two-channel SGD: numeric coefficient + LLM text update
├── llm_client/          # LLM backend interface
│   ├── coding_agent_query  # Async Claude Agent SDK wrapper
│   └── _call_claude        # OpenAI-compatible API fallback
├── data_loader/         # Dataset utilities
│   └── sole_file_batch_data_loader  # Load files into symbolic tensors
└── example/             # End-to-end example
    └── naive_symbolic_transform_model/
        ├── train.py      # Training loop: Python -> Viba translation
        ├── model.py      # NaiveModel wrapping SymbolicTransformModule
        ├── init_dataset.py # Dataset initialization (3 code pairs)
        └── dataset/      # Python and Viba code samples
```

## Dual-Channel Gradient System

Symbolic tensors propagate gradients through **two channels**:

| Channel | What it carries | How it's computed |
|---------|----------------|-------------------|
| **Numeric** (coefficient) | Float values | Standard autograd / SGD arithmetic |
| **Symbolic** (text) | Text diffs stored in files | LLM generates diff descriptions |

## Demo: Python to Viba Translation

This example trains a model to translate Python code into Viba (a pattern-matching language) using experience entries and an LLM.

### 1. Initialize Dataset

```python
from symbolic_tensor.example.naive_symbolic_transform_model.init_dataset import init_dataset
init_dataset()
```

Creates 3 translation pairs:
- `seq.py` (sequential) -> `seq.viba`
- `branch.py` (if/else) -> `branch.viba`
- `loop.py` (for-loop) -> `loop.viba`

### 2. Build Tensors

```python
from symbolic_tensor.tensor_util.make_tensor import make_tensor

# Input: Python source code
input_tensor = make_tensor(
    [Path("dataset/branch.py"), Path("dataset/loop.py")],
    tmpdir, symlink=True
)

# Experience: [query_keywords, key_python, value_viba]
experience = make_tensor([
    ["branch\npython\nviba", "def classify(x):\n    if x > 0: ...", "classify :=\n    | 'positive'\n    ..."],
    ["loop\npython\nviba",   "def double_all(items):\n    for item in items: ...", "double_all :=\n    list[$doubled int]\n    ..."],
], tmpdir)

# Expected output: Viba code
expected_tensor = make_tensor([
    open("dataset/branch.viba").read(),
    open("dataset/loop.viba").read(),
], tmpdir)
```

### 3. Forward Pass

```python
from symbolic_tensor.function.symbolic_transform_forward import symbolic_transform_forward

output, selected_indexes = symbolic_transform_forward(
    input_tensor, experience,
    forward_prompt="Translate Python To Viba",
    topk=2,
)
# LLM reads experience entries (key->value mappings) and
# translates each input Python file to Viba code.
```

### 4. Compute Loss

```python
from symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl

loss = get_edit_distance_ratio_impl(output, expected_tensor)
mean_loss = loss.mean().item()
# Levenshtein-based edit distance ratio per element
```

### 5. Backward Pass

```python
from symbolic_tensor.function.symbolic_transform_backward import symbolic_transform_backward

# Construct symbolic gradient: text describing how output should change
grad_output = make_tensor([
    "The output does not match expected.\nExpected: ...\nActual: ...\n"
    "Please update experience to improve translations.",
    ...
], tmpdir)
grad_output.data.fill_(1.0)

grad_input, grad_experience = symbolic_transform_backward(
    grad_output, input_tensor, output, experience,
    selected_experience_qkv_indexes_list=selected_indexes,
    forward_prompt="Translate Python To Viba",
    topk=2,
)
# LLM computes text diffs for both input and experience gradients
```

### 6. Optimizer Step

```python
from symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD

optimizer = SymbolicSGD(
    model.parameters(), lr=1.0,
    step_prompt="You are updating Python-to-Viba translation experience entries."
)

experience.grad = grad_experience
optimizer.step()
# Numeric: experience.data -= lr * grad.data
# Symbolic: LLM applies text diffs to experience storage files
```

### 7. Full Training Loop

```bash
# Requires ANTHROPIC_API_KEY or LLM_API_KEY/LLM_BASE_URL/LLM_MODEL env vars
python -m symbolic_tensor.example.naive_symbolic_transform_model.train
```

Output:
```
============================================================
NaiveModel Training: Translate Python To Viba
============================================================

Dataset: 2 pairs
  [0] branch.py -> branch.viba
  [1] loop.py -> loop.viba

Experience shape: [2, 3]
Input shape: [2]

Iteration 1/5
  [Forward]
    output[0]: "classify := ..."
  [Loss]
    Per-sample losses: ['0.4500', '0.3200']
    Mean loss: 0.3850
  [Backward]
  [Optimizer Step]
    Experience[0].value changed: True

...
```

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
2. **Experience Retrieval**: Jaccard similarity selects top-k relevant experience entries
3. **Context Assembly**: Dump symlink views of experience (query/key/value) and input
4. **LLM Translation**: Coding agent reads context and writes output to mutable copies
5. **Copy-back**: Results propagate through symlinks to parent tensor storage

### Backward Pass Pipeline

1. **Numeric Gradient**: Standard coefficient pass-through + scatter-add to experience
2. **Symbolic Gradient**: LLM computes text diffs describing how input/experience should change
3. Both channels merge into the gradient tensor

## Key Design Decisions

- **Symlinks for views, copies for mutations**: `slice_view` creates symlinks (shared storage, read-only context); `slice_tensor` creates independent copies (safe for LLM writes)
- **Coordinate-based views**: `dump_view` maps multi-dimensional indices to human-readable paths (e.g., `0/1/data.txt`) for LLM consumption
- **Cached experience views**: QKV data views are computed once and reused across forward passes
- **LLM as compute kernel**: The LLM coding agent (Claude Agent SDK) replaces traditional matrix multiplication with semantic reasoning

## Dependencies

- Python 3.13+
- PyTorch
- `claude-agent-sdk` (Claude Agent SDK for LLM coding agent)
- `openai` (fallback LLM client)
- `Levenshtein` (edit distance computation)

## Installation

```bash
pip install torch claude-agent-sdk openai Levenshtein
```

## Quick Start

```python
from symbolic_tensor import tensor, none

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
