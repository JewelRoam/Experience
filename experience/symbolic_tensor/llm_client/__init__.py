import os

from openai import OpenAI
from pathlib import Path


def _call_claude(prompt: str) -> str:
    """
    Call the LLM via OpenAI-compatible API.
    Requires LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL
    environment variables.
    """
    env = os.environ
    with Path("/tmp/a.log").open("a", encoding="utf-8") as f:
        print(f"ENTER _call_claude, model={env.get('LLM_MODEL')}\n{prompt}", file=f)
    client = OpenAI(
        api_key=env.get('LLM_API_KEY'),
        base_url=env.get('LLM_BASE_URL'),
    )
    response = client.chat.completions.create(
        model=env.get('LLM_MODEL'),
        messages=[
            {'role': 'user', 'content': prompt},
        ],
    )
    ret = response.choices[0].message.content.strip()
    with Path("/tmp/a.log").open("a", encoding="utf-8") as f:
        print(f"EXIT _call_claude, model={env.get('LLM_MODEL')}\n{ret}", file=f)
    return ret
