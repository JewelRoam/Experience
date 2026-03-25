from dataclasses import dataclass
from typing import Union, List


@dataclass
class AgentTask:
    workspace_dir: str
    output_relative_dir: Union[str, List[str]]
    prompt: str
    todo_file_content_hint: str = "TODO"


if __name__ == "__main__":
    print("Running AgentTask tests...\n")

    # Test 1: Basic construction with str
    task = AgentTask(workspace_dir="/tmp/test", output_relative_dir="output", prompt="do something")
    assert task.workspace_dir == "/tmp/test"
    assert task.output_relative_dir == "output"
    assert task.prompt == "do something"
    assert task.todo_file_content_hint == "TODO"
    print("  ok: basic construction with str output_relative_dir")

    # Test 2: Construction with list[str]
    task_list = AgentTask(workspace_dir="/tmp/test", output_relative_dir=["out1", "out2"], prompt="do something")
    assert task_list.output_relative_dir == ["out1", "out2"]
    print("  ok: construction with list[str] output_relative_dir")

    # Test 3: Custom todo_file_content_hint
    task2 = AgentTask(workspace_dir="/tmp/w", output_relative_dir="out", prompt="p", todo_file_content_hint="PLACEHOLDER")
    assert task2.todo_file_content_hint == "PLACEHOLDER"
    print("  ok: custom todo_file_content_hint")

    # Test 4: Equality
    task3 = AgentTask(workspace_dir="/tmp/test", output_relative_dir="output", prompt="do something")
    assert task == task3
    print("  ok: dataclass equality")

    print("\nAll tests passed.")