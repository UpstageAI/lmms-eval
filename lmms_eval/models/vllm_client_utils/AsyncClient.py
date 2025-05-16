from itertools import cycle
import asyncio
import openai
from typing import List, Union

from lmms_eval.models.vllm_client_utils.BaseClient import BaseClient

class AsyncOpenAIPool(BaseClient):
    def __init__(self, model_name: str, host_list: List[str], port_list: List[int], use_system_prompt: bool = False, prompt_image_type: int = 1, max_completion_tokens: int = 4096):
        """Create a list of OpenAI clients with the given base URLs and API keys"""
        if isinstance(host_list, str) or isinstance(host_list, int):
            host_list = [host_list]
        if isinstance(port_list, str) or isinstance(port_list, int):
            port_list = [port_list]

        base_urls = [f"http://{host}:{port}/v1" for host, port in zip(host_list, port_list)]
        print(base_urls)
        api_key="token-abc123"
        self.clients = [openai.AsyncOpenAI(base_url=base_url, api_key=api_key) for base_url in base_urls]
        self.client_cycle = cycle(self.clients)
        self.lock = asyncio.Lock()  # 비동기 락 추가

        self.MODEL = model_name
        self.use_system_prompt = False
        self.prompt_image_type = 1
        self.max_completion_tokens = 16384

    async def get(self):
        """Get the next OpenAI client in the cycle in a thread-safe manner"""
        async with self.lock:  # 락을 사용하여 동시 접근 제어
            return next(self.client_cycle)

    async def _chat_completion(self, system_prompt:str, user_prompt:str, image_url_list:List[str], guided_json:Union[dict, None]=None):
        messages, extra_body = self._make_prompt(system_prompt, user_prompt, image_url_list, guided_json)

        client = await self.get()  # 비동기 메서드 호출로 변경
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=self.MODEL,
            max_completion_tokens=self.max_completion_tokens,
            temperature=0,
            extra_body=extra_body
        )

        return chat_completion.choices[0].message.content