import os
import time
import subprocess
import tempfile
import torch

from experience.symbolic_tensor.tensor_util.make_tensor import make_tensor
from experience.symbolic_tensor.function.st_matmul_forward import st_matmul_forward


def run_benchmark(input_data, experience_data, topk, llm_method):
    """Run a single benchmark, return (time_seconds, outputs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(input_data, tmpdir)
        experience_tensor = make_tensor(experience_data, tmpdir)

        t0 = time.time()
        output, selected_indexes = st_matmul_forward(
            input_tensor, experience_tensor,
            topk=topk,
            llm_method=llm_method,
        )
        t1 = time.time()
        elapsed = t1 - t0

        outputs = []
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        for i in range(output.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            with open(path) as f:
                content = f.read()
            outputs.append(content)

        return elapsed, outputs


if __name__ == "__main__":
    # Source env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    experience_data = [
        ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
        ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ["counting\nnumbers", "One two three in English", "Un deux trois en francais"],
    ]
    topk = 2
    methods = ["raw_llm_api", "coding_agent"]

    # Define scale tests: 1x, 2x, 3x inputs
    scale_inputs = {
        "1x": ["Hello world in English"],
        "2x": ["Hello world in English", "Goodbye my friend in English"],
        "3x": ["Hello world in English", "Goodbye my friend in English", "One two three in English"],
    }

    # results[scale][method] = (time, outputs)
    results = {}

    print("=" * 70)
    print("Transform Method Time Comparison: 1x, 2x, 3x Scale")
    print("=" * 70)

    for scale_name, input_data in scale_inputs.items():
        results[scale_name] = {}
        print(f"\n{'='*70}")
        print(f"  Scale: {scale_name} ({len(input_data)} input(s))")
        print(f"{'='*70}")

        for method in methods:
            print(f"\n  --- {method} ---")
            elapsed, outputs = run_benchmark(
                input_data, experience_data, topk, method
            )
            results[scale_name][method] = (elapsed, outputs)
            for i, out in enumerate(outputs):
                print(f"    Output {i}: {repr(out.strip())}")
            print(f"    Time: {elapsed:.2f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scale':<8} {'coding_agent':>14} {'raw_llm_api':>14} {'Speedup':>10}")
    print("-" * 50)

    for scale_name in scale_inputs:
        ca_time = results[scale_name]["coding_agent"][0]
        rl_time = results[scale_name]["raw_llm_api"][0]
        if rl_time > 0:
            speedup = ca_time / rl_time
        else:
            speedup = float("inf")
        print(f"{scale_name:<8} {ca_time:>12.2f}s {rl_time:>12.2f}s {speedup:>9.1f}x")

    # Scaling analysis
    print("\n" + "-" * 50)
    print("Scaling (time per element):")
    print(f"{'Scale':<8} {'coding_agent':>14} {'raw_llm_api':>14}")
    print("-" * 40)
    for scale_name, input_data in scale_inputs.items():
        n = len(input_data)
        ca_per = results[scale_name]["coding_agent"][0] / n
        rl_per = results[scale_name]["raw_llm_api"][0] / n
        print(f"{scale_name:<8} {ca_per:>12.2f}s {rl_per:>12.2f}s")
