import os
import asyncio
from typing import List

from symbolic_tensor.llm_client.agent_task import AgentTask
from symbolic_tensor.llm_client.pack_dir import pack_dir


def _flatten_nested(nested) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested(item))
    return result


def _grep_by_file_content_hint(root_dir: str, todo_file_content_hint: str) -> List[str]:
    """Find all files under root_dir whose content contains the hint string."""
    todo_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if todo_file_content_hint in content:
                    todo_files.append(file_path)
            except (UnicodeDecodeError, OSError):
                continue
    return todo_files


class RawLlmTaskHandler:
    def __call__(self, all_tasks) -> None:
        flat_tasks = _flatten_nested(all_tasks)

        # Collect all async jobs: (todo_file_path, coroutine)
        jobs = []
        for task in flat_tasks:
            workspace_dir = task.workspace_dir
            prompt = task.prompt
            output_relative_dir = task.output_relative_dir
            todo_file_content_hint = task.todo_file_content_hint

            packed_workspace = pack_dir(workspace_dir)

            output_root = os.path.join(workspace_dir, output_relative_dir)
            todo_file_paths = _grep_by_file_content_hint(output_root, todo_file_content_hint)

            for todo_file_path in todo_file_paths:
                raw_llm_prompt = (
                    f"{prompt}\n\n"
                    f"The Directories are packed as following like repomix:\n\n"
                    f"{packed_workspace}\n\n"
                    f"Your output will replace the {todo_file_content_hint} placeholder in file {todo_file_path}. \n"
                    f"Do not generate unrelated content."
                )
                jobs.append((todo_file_path, raw_llm_prompt))

        async def _do_one(todo_file_path: str, raw_llm_prompt: str):
            from symbolic_tensor.llm_client.raw_llm_query import raw_llm_query
            output_content = await raw_llm_query(raw_llm_prompt)
            with open(todo_file_path, "w", encoding="utf-8") as f:
                f.write(output_content)

        async def _run_all():
            await asyncio.gather(*[_do_one(path, prompt) for path, prompt in jobs])

        asyncio.run(_run_all())


if __name__ == "__main__":
    import subprocess
    import tempfile

    # Source LLM env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val

    print("Running RawLlmTaskHandler tests...\n")

    # Test 1: _grep_by_file_content_hint
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.txt"), "w") as f:
            f.write("this has TODO in it")
        with open(os.path.join(tmpdir, "b.txt"), "w") as f:
            f.write("this does not")
        os.makedirs(os.path.join(tmpdir, "sub"))
        with open(os.path.join(tmpdir, "sub", "c.txt"), "w") as f:
            f.write("another TODO here")

        results = _grep_by_file_content_hint(tmpdir, "TODO")
        basenames = sorted([os.path.basename(p) for p in results])
        assert basenames == ["a.txt", "c.txt"], f"got {basenames}"
        print("  ok: _grep_by_file_content_hint finds correct files")

    # Test 2: _flatten_nested
    nested = [[1, 2], [3, [4, 5]]]
    assert _flatten_nested(nested) == [1, 2, 3, 4, 5]
    print("  ok: _flatten_nested works")

    # Test 3: Handler instantiation
    handler = RawLlmTaskHandler()
    assert callable(handler)
    print("  ok: RawLlmTaskHandler is callable")

    # Test 4: Integration test with real LLM
    print("\n  Test 4: Integration test (LLM call)")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create workspace with input and TODO output
        os.makedirs(os.path.join(tmpdir, "const_input"))
        with open(os.path.join(tmpdir, "const_input", "data.txt"), "w") as f:
            f.write("Hello world in English")
        os.makedirs(os.path.join(tmpdir, "mutable_output"))
        with open(os.path.join(tmpdir, "mutable_output", "data.txt"), "w") as f:
            f.write("TODO")
        task = AgentTask(
            workspace_dir=tmpdir,
            output_relative_dir="mutable_output",
            prompt="Translate the English text in const_input to French.",
        )
        handler([task])

        with open(os.path.join(tmpdir, "mutable_output", "data.txt"), "r") as f:
            output = f.read()
        assert "TODO" not in output, f"Output still contains TODO: {output}"
        print(f"  ok: LLM replaced TODO with: {repr(output)}")

    print("\nAll tests passed.")
