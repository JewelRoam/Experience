import asyncio
import sys
from claude_agent_sdk import query, ClaudeAgentOptions

async def coding_agent_query(
    prompt: str,
    cwd: str | None = None,
    allowed_tools: list[str] = None,
    permission_mode: str = "acceptEdits",
):
    if allowed_tools is None:
        allowed_tools = ["Read", "Edit"]
    async for item in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            permission_mode=permission_mode,
            cwd=cwd,
        ),
    ):
        # print(item, file=sys.stderr)
        yield item

def async_run(handlers : list):
    async def func():
        await asyncio.gather(*handlers)
    asyncio.run(func())

if __name__ == '__main__':

    async def foo():
        async for message in coding_agent_query(
            prompt="hello, save your greeting response to a file named './foo.txt'",
            cwd='/tmp/demo',
        ):
            print(message)  # Claude reads the file, finds the bug, edits it
    import os
    os.makedirs('/tmp/demo', exist_ok=True)
    async_run([foo()])
