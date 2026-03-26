import os
import asyncio
from typing import Dict, List, Optional

from experience.llm_client.agent_task import AgentTask
from experience.llm_client.coding_agent_query import coding_agent_query


def _flatten_nested(nested) -> list:
    """Flatten a nested list structure into a flat list."""
    if not isinstance(nested, list):
        return [nested]
    result = []
    for item in nested:
        result.extend(_flatten_nested(item))
    return result


class CodingAgentTaskHandler:
    def __call__(self, all_tasks, llm_env: Optional[Dict[str, str]] = None) -> None:
        flat_tasks = _flatten_nested(all_tasks)

        async def _do_one(task: AgentTask):
            workspace_dir = task.workspace_dir
            prompt = task.prompt
            env_backup = os.environ.pop("CLAUDECODE", None)
            try:
                async for _ in coding_agent_query(prompt=prompt, cwd=workspace_dir, allowed_tools=["Read", "Edit", "Write"], llm_env=llm_env):
                    pass
            finally:
                if env_backup is not None:
                    os.environ["CLAUDECODE"] = env_backup

        async def _run_all():
            await asyncio.gather(*[_do_one(task) for task in flat_tasks])

        asyncio.run(_run_all())
