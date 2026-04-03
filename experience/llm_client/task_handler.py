from experience.llm_client.agent_task import AgentTask
from experience.llm_client.coding_agent_task_handler import CodingAgentTaskHandler
from experience.llm_client.raw_llm_task_handler import RawLlmTaskHandler
import time
import sys
from typing import Dict, Optional

class TaskHandler:
    def __call__(self, all_tasks, llm_method: str, llm_env: Optional[Dict[str, str]] = None) -> None:
        start = time.time()
        if llm_method == "coding_agent":
            CodingAgentTaskHandler()(all_tasks, llm_env=llm_env)
        elif llm_method == "raw_llm_api":
            RawLlmTaskHandler()(all_tasks, llm_env=llm_env)
        else:
            raise ValueError(f"Unknown llm_method: {llm_method}")
        end = time.time()
        # print(f"Finished processing tasks in {end-start:.2f} seconds", file=sys.stderr)


if __name__ == "__main__":
    import os
    import subprocess
    import tempfile

    # Source LLM env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.LLM_config.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running TaskHandler tests...\n")

    # Test 1: Instantiation
    handler = TaskHandler()
    assert callable(handler)
    print("  ok: TaskHandler is callable")

    # Test 2: Invalid method raises
    try:
        handler([], "nonexistent")
        assert False, "Should have raised"
    except ValueError as e:
        assert "nonexistent" in str(e)
        print("  ok: invalid method raises ValueError")

    # Test 3: Integration with raw_llm_api
    print("\n  Test 3: Integration (raw_llm_api)")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "input"))
        with open(os.path.join(tmpdir, "input", "data.txt"), "w") as f:
            f.write("Hello world")
        os.makedirs(os.path.join(tmpdir, "output"))
        with open(os.path.join(tmpdir, "output", "data.txt"), "w") as f:
            f.write("TODO")

        task = AgentTask(
            workspace_dir=tmpdir,
            output_relative_dir="output",
            prompt="Translate the text in input/ to French.",
        )
        handler([task], "raw_llm_api")

        with open(os.path.join(tmpdir, "output", "data.txt")) as f:
            content = f.read()
        assert "TODO" not in content
        print(f"  ok: raw_llm_api produced: {repr(content)}")

    print("\nAll tests passed.")
