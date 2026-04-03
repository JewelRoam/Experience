"""Run auto-encoder experiment multiple times to check baseline stability."""
import os
import subprocess
import sys

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

sys.path.insert(0, "/workspace/Experience")

# Hyphen in dir name — use importlib
import importlib
mod = importlib.import_module("demo.auto-encoder.claude")

num_iterations = 1

losses = []
for i in range(num_iterations):
    print(f"\n{'='*50}")
    print(f"Run {i+1}/num_iterations")
    print(f"{'='*50}")
    loss = mod.run_experiment(total_batch_size=16, workspace_dir="/workspace/code-auto-encoder/", llm_method="raw_llm_api")
    losses.append(loss)
    print(f">>> Run {i+1} loss: {loss:.4f}")

print(f"\n{'='*50}")
print("Summary (10 runs):")
print(f"{'='*50}")
print(f"Losses: {['%.4f' % l for l in losses]}")
mean = sum(losses) / len(losses)
std = (sum((l - mean)**2 for l in losses) / len(losses))**0.5
print(f"Mean: {mean:.4f}")
print(f"Min:  {min(losses):.4f}")
print(f"Max:  {max(losses):.4f}")
print(f"Std:  {std:.4f}")
