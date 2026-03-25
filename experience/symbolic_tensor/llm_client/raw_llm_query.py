# 安装：pip3 install openai
import asyncio
import os
from openai import AsyncOpenAI  # 导入异步客户端


async def raw_llm_query(prompt: str):
    # 每次调用创建客户端，避免跨 asyncio.run() 复用已关闭的连接池
    client = AsyncOpenAI(
        api_key=os.environ.get('LLM_API_KEY'),
        base_url=os.environ.get('LLM_BASE_URL')
    )
    # 发送异步请求
    try:
        response = await client.chat.completions.create(
            model=os.environ.get('LLM_MODEL'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    finally:
        await client.close()

if __name__ == "__main__":
    ret = asyncio.run(raw_llm_query("Hello"))
    print(ret)